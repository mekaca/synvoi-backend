import torch

MODEL_PATH = '/Users/mekaca/Desktop/45/demo_model_v3.83.7.pth'
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

state_dict = checkpoint['model_state_dict']

print("=" * 80)
print("MODEL MİMARİSİ ANALİZİ (Checkpoint'ten)")
print("=" * 80)

# Gruplanmış layer analizi
layers = {}
for key, tensor in state_dict.items():
    base = key.split('.')[0]
    if base not in layers:
        layers[base] = []
    layers[base].append((key, tensor.shape))

# Önemli katmanları yazdır
for base in ['mel_encoder_512', 'mel_encoder_1024', 'mfcc_encoder', 
             'feature_fusion', 'rnn', 'classifier']:
    if base in layers:
        print(f"\n📦 {base.upper()}:")
        for key, shape in layers[base][:8]:  # İlk 8 layer
            print(f"   {key}: {shape}")

# RNN hidden size hesapla
if 'rnn' in layers:
    for key, shape in layers['rnn']:
        if 'weight_ih_l0' in key:
            hidden_size = shape[0] // 4  # LSTM = 4 * hidden_size
            print(f"\n✅ RNN Hidden Size: {hidden_size}")
            break

print("\n" + "=" * 80)
