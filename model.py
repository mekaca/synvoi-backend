import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiResolutionFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mfcc=20):
        super().__init__()
        self.mel_512 = T.MelSpectrogram(sample_rate=sample_rate, n_fft=512, hop_length=128, n_mels=64, f_min=20, f_max=8000)
        self.mel_1024 = T.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=128, f_min=20, f_max=8000)
        self.mfcc = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={'n_fft': 1024, 'hop_length': 256, 'n_mels': 64})
    def forward(self, waveform):
        return {'mel_512': torch.log1p(self.mel_512(waveform)), 'mel_1024': torch.log1p(self.mel_1024(waveform)), 'mfcc': self.mfcc(waveform)}

class TemporalStatisticsPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Linear(d_model * 4, d_model)
    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        min_pool, _ = torch.min(x, dim=1, keepdim=True)
        return self.projection(torch.cat([mean, std, max_pool, min_pool], dim=-1)).squeeze(1)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, conv_kernel_size=31, dropout=0.15):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4*d_model), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_module = nn.Sequential(nn.Conv1d(d_model, 2*d_model, 1), nn.GLU(dim=1), nn.Conv1d(d_model, d_model, conv_kernel_size, padding=conv_kernel_size//2, groups=d_model), nn.BatchNorm1d(d_model), nn.SiLU(), nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout))
        self.ff2 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4*d_model), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        attn_out = self.attn_norm(x)
        attn_out, _ = self.attn(attn_out, attn_out, attn_out)
        x = x + self.dropout(attn_out)
        x = x + self.conv_module(self.conv_norm(x).transpose(1,2)).transpose(1,2)
        x = x + 0.5 * self.ff2(x)
        return x

class ConformerVoiceDetector(nn.Module):
    def __init__(self, num_classes=2, d_model=384, n_conformer_blocks=6, n_heads=12, num_segments=12, rnn_hidden=384, rnn_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        
        self.mel_encoder_512 = nn.Sequential(nn.Conv2d(1, 32, (5,5), (2,1), 2), nn.BatchNorm2d(32), nn.LeakyReLU(0.3), nn.Conv2d(32, 64, (3,3), (2,1), 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.3))
        self.mel_encoder_1024 = nn.Sequential(nn.Conv2d(1, 64, (7,7), (2,1), 3), nn.BatchNorm2d(64), nn.LeakyReLU(0.3), nn.MaxPool2d((3,3), (2,2), 1), nn.Conv2d(64, 128, (5,5), (2,1), 2), nn.BatchNorm2d(128), nn.LeakyReLU(0.3))
        self.mfcc_encoder = nn.Sequential(nn.Conv2d(1, 32, (3,3), (1,1), 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.3), nn.MaxPool2d((2,2)))
        
        self._calculate_fusion_size()
        self.pos_encoding = PositionalEncoding(d_model)
        self.conformer_blocks = nn.ModuleList([ConformerBlock(d_model, n_heads, dropout=0.15) for _ in range(n_conformer_blocks)])
        
        # ✅ GRU (checkpoint has GRU weights!)
        self.rnn = nn.GRU(d_model, rnn_hidden, rnn_layers, batch_first=True, bidirectional=True, dropout=0.3)
        
        self.temporal_pool = TemporalStatisticsPooling(rnn_hidden * 2)
        self.segment_attention = nn.Sequential(nn.Linear(rnn_hidden*2, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Linear(rnn_hidden*2, 256), nn.LeakyReLU(0.3), nn.Dropout(0.5), nn.Linear(256, 128), nn.LeakyReLU(0.3), nn.Dropout(0.4), nn.Linear(128, num_classes))

    def _calculate_fusion_size(self):
        with torch.no_grad():
            mel_512 = self.mel_encoder_512(torch.randn(1,1,64,100)).permute(0,3,1,2).flatten(2)
            mel_1024 = self.mel_encoder_1024(torch.randn(1,1,128,100)).permute(0,3,1,2).flatten(2)
            mfcc = self.mfcc_encoder(torch.randn(1,1,20,100)).permute(0,3,1,2).flatten(2)
            min_time = min(mel_512.size(1), mel_1024.size(1), mfcc.size(1))
            total_size = mel_512[:,:min_time,:].size(-1) + mel_1024[:,:min_time,:].size(-1) + mfcc[:,:min_time,:].size(-1)
        self.feature_fusion = nn.Sequential(nn.LayerNorm(total_size), nn.Linear(total_size, 384), nn.LeakyReLU(0.3), nn.Dropout(0.3), nn.Linear(384, self.d_model))

    def forward(self, x):
        mel_512 = self.mel_encoder_512(x['mel_512']).permute(0,3,1,2).flatten(2)
        mel_1024 = self.mel_encoder_1024(x['mel_1024']).permute(0,3,1,2).flatten(2)
        mfcc = self.mfcc_encoder(x['mfcc']).permute(0,3,1,2).flatten(2)
        min_time = min(mel_512.size(1), mel_1024.size(1), mfcc.size(1))
        x = self.feature_fusion(torch.cat([mel_512[:,:min_time,:], mel_1024[:,:min_time,:], mfcc[:,:min_time,:]], dim=-1))
        x = self.pos_encoding(x)
        for block in self.conformer_blocks:
            x = block(x)
        time_steps, segment_size, segments = x.size(1), max(1, x.size(1) // self.num_segments), []
        for i in range(self.num_segments):
            start_idx, end_idx = i * segment_size, min((i+1) * segment_size, time_steps)
            if start_idx < time_steps and end_idx > start_idx:
                rnn_out, _ = self.rnn(x[:, start_idx:end_idx, :])
                segments.append(self.temporal_pool(rnn_out))
        if not segments:
            rnn_out, _ = self.rnn(x)
            final_repr = self.temporal_pool(rnn_out)
        else:
            segments = torch.stack(segments, dim=1)
            final_repr = torch.sum(segments * F.softmax(self.segment_attention(segments), dim=1), dim=1)
        return self.classifier(final_repr)

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    model = ConformerVoiceDetector(num_classes=2, d_model=config['d_model'], n_conformer_blocks=config['n_conformer_blocks'], n_heads=config['n_heads'], num_segments=config['num_segments'], rnn_hidden=config['rnn_hidden'], rnn_layers=config['rnn_layers'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ GRU model - %83.7 accuracy tamamen yüklendi!")
    return model, config
