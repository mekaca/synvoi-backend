import torch

MODEL_PATH = '/Users/mekaca/Desktop/45/demo_model_v3.83.7.pth'

checkpoint = torch.load(MODEL_PATH, map_location='cpu')

print("=" * 60)
print("MODEL CONFIG ANALİZİ")
print("=" * 60)

config = checkpoint['config']

print("\n📋 CONFIG İÇERİĞİ:")
print(config)

print("\n" + "=" * 60)
