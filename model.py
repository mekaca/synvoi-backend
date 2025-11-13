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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiResolutionFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mfcc=20):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Multi-resolution Mel Spectrograms
        self.mel_512 = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, hop_length=128,
            n_mels=64, f_min=20, f_max=8000
        )
        self.mel_1024 = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256,
            n_mels=128, f_min=20, f_max=8000
        )
        
        # MFCC
        self.mfcc = T.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc,
            melkwargs={'n_fft': 1024, 'hop_length': 256, 'n_mels': 64}
        )

    def forward(self, waveform):
        features = {}
        
        # Apply log compression
        mel_512 = torch.log1p(self.mel_512(waveform))
        mel_1024 = torch.log1p(self.mel_1024(waveform))
        mfcc = self.mfcc(waveform)
        
        features['mel_512'] = mel_512
        features['mel_1024'] = mel_1024
        features['mfcc'] = mfcc
        
        return features

class TemporalStatisticsPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.projection = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        min_pool, _ = torch.min(x, dim=1, keepdim=True)
        
        stats = torch.cat([mean, std, max_pool, min_pool], dim=-1)
        return self.projection(stats).squeeze(1)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, conv_kernel_size=31, dropout=0.15):
        super().__init__()
        
        # Feed-forward 1
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Convolution module
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size,
                     padding=conv_kernel_size//2, groups=d_model),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # Feed-forward 2
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # First feed-forward (half residual)
        x = x + 0.5 * self.ff1(x)
        
        # Multi-head self-attention
        attn_out = self.attn_norm(x)
        attn_out, _ = self.attn(attn_out, attn_out, attn_out)
        x = x + self.dropout(attn_out)
        
        # Convolution
        conv_in = self.conv_norm(x).transpose(1, 2)
        conv_out = self.conv_module(conv_in).transpose(1, 2)
        x = x + conv_out
        
        # Second feed-forward (half residual)
        x = x + 0.5 * self.ff2(x)
        
        return x

class ConformerVoiceDetector(nn.Module):
    def __init__(self, num_classes=2, d_model=384, n_conformer_blocks=6, n_heads=12,
                 num_segments=8, rnn_hidden=384, rnn_layers=2):
        super().__init__()
        self.d_model = d_model
        self.num_segments = num_segments
        
        # Multi-resolution encoders
        self.mel_encoder_512 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 1), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3)
        )
        
        self.mel_encoder_1024 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 1), padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 1), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3)
        )
        
        self.mfcc_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.3),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Calculate fusion size
        self._calculate_fusion_size()
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dropout=0.15) 
            for _ in range(n_conformer_blocks)
        ])
        
        # Bidirectional GRU
        self.rnn = nn.GRU(
            d_model, rnn_hidden, rnn_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        
        # Temporal pooling
        self.temporal_pool = TemporalStatisticsPooling(rnn_hidden * 2)
        
        # Segment attention
        self.segment_attention = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 256),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def _calculate_fusion_size(self):
        """Calculate fusion layer size with dummy pass"""
        dummy_mel_512 = torch.randn(1, 1, 64, 100)
        dummy_mel_1024 = torch.randn(1, 1, 128, 100)
        dummy_mfcc = torch.randn(1, 1, 20, 100)
        
        with torch.no_grad():
            mel_512 = self.mel_encoder_512(dummy_mel_512)
            mel_512 = mel_512.permute(0, 3, 1, 2).flatten(2)
            
            mel_1024 = self.mel_encoder_1024(dummy_mel_1024)
            mel_1024 = mel_1024.permute(0, 3, 1, 2).flatten(2)
            
            mfcc = self.mfcc_encoder(dummy_mfcc)
            mfcc = mfcc.permute(0, 3, 1, 2).flatten(2)
            
            min_time = min(mel_512.size(1), mel_1024.size(1), mfcc.size(1))
            
            total_size = (mel_512[:, :min_time, :].size(-1) +
                         mel_1024[:, :min_time, :].size(-1) +
                         mfcc[:, :min_time, :].size(-1))
        
        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(total_size),
            nn.Linear(total_size, 384),
            nn.LeakyReLU(0.3),
            nn.Dropout(0.3),
            nn.Linear(384, self.d_model)
        )

    def forward(self, x):
        # Process all feature types
        mel_512 = self.mel_encoder_512(x['mel_512'])
        mel_512 = mel_512.permute(0, 3, 1, 2).flatten(2)
        
        mel_1024 = self.mel_encoder_1024(x['mel_1024'])
        mel_1024 = mel_1024.permute(0, 3, 1, 2).flatten(2)
        
        mfcc = self.mfcc_encoder(x['mfcc'])
        mfcc = mfcc.permute(0, 3, 1, 2).flatten(2)
        
        # Align temporal dimensions
        min_time = min(mel_512.size(1), mel_1024.size(1), mfcc.size(1))
        
        features = [
            mel_512[:, :min_time, :],
            mel_1024[:, :min_time, :],
            mfcc[:, :min_time, :]
        ]
        
        # Concatenate and fuse
        x = torch.cat(features, dim=-1)
        x = self.feature_fusion(x)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)
        
        # Segment processing
        time_steps = x.size(1)
        segment_size = max(1, time_steps // self.num_segments)
        segments = []
        
        for i in range(self.num_segments):
            start_idx = i * segment_size
            end_idx = min(start_idx + segment_size, time_steps)
            
            if start_idx < time_steps:
                segment = x[:, start_idx:end_idx, :]
                if segment.size(1) > 0:
                    rnn_out, _ = self.rnn(segment)
                    segment_repr = self.temporal_pool(rnn_out)
                    segments.append(segment_repr)
        
        if len(segments) == 0:
            rnn_out, _ = self.rnn(x)
            final_repr = self.temporal_pool(rnn_out)
        else:
            segments = torch.stack(segments, dim=1)
            attn_weights = self.segment_attention(segments)
            attn_weights = F.softmax(attn_weights, dim=1)
            final_repr = torch.sum(segments * attn_weights, dim=1)
        
        return self.classifier(final_repr)
