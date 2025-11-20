import torch
import os

# Paths
MODEL_PATH = '/Users/mekaca/Desktop/45/demo_model_v3.83.7.pth'
OUTPUT_PATH = '/Users/mekaca/Desktop/45/demo_model_v3.83.7_fp16.pth'

print("=" * 60)
print("FP16 QUANTIZATION")
print("=" * 60)

# Model yükle
print("\n[1/4] Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

# Dosya yapısını kontrol et
if 'model_state_dict' in checkpoint:
    print("✓ Format: State dict with metadata")
    state_dict = checkpoint['model_state_dict']
    
    # FP16'ya çevir
    print("\n[2/4] Converting to FP16...")
    converted = 0
    for key in state_dict:
        if state_dict[key].dtype == torch.float32:
            state_dict[key] = state_dict[key].half()
            converted += 1
    
    checkpoint['model_state_dict'] = state_dict
    print(f"✓ Converted {converted} tensors to FP16")
    
else:
    print("✓ Format: Direct state dict")
    
    # FP16'ya çevir
    print("\n[2/4] Converting to FP16...")
    converted = 0
    for key in checkpoint:
        if isinstance(checkpoint[key], torch.Tensor) and checkpoint[key].dtype == torch.float32:
            checkpoint[key] = checkpoint[key].half()
            converted += 1
    
    print(f"✓ Converted {converted} tensors to FP16")

# Kaydet
print("\n[3/4] Saving quantized model...")
torch.save(checkpoint, OUTPUT_PATH)

# Boyut karşılaştırması
original_size = os.path.getsize(MODEL_PATH) / (1024**2)
quantized_size = os.path.getsize(OUTPUT_PATH) / (1024**2)

print("\n[4/4] Results:")
print(f"✓ Original:  {original_size:.2f} MB")
print(f"✓ Quantized: {quantized_size:.2f} MB")
print(f"✓ Reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
print(f"\n✅ Quantized model saved to:")
print(f"   {OUTPUT_PATH}")
print("=" * 60)
