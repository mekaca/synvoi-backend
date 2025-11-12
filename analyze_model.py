import torch
import os

MODEL_PATH = '/Users/mekaca/Desktop/45/demo_model_v3.83.7.pth'

print("=" * 60)
print("MODEL YAPISI ANALİZİ")
print("=" * 60)

# Dosya kontrolü
if not os.path.exists(MODEL_PATH):
    print(f"\n❌ HATA: Model dosyası bulunamadı!")
    print(f"   Aranılan: {MODEL_PATH}")
    exit(1)

print(f"\n✓ Model dosyası bulundu")
print(f"  Boyut: {os.path.getsize(MODEL_PATH) / (1024**2):.2f} MB")

# Model yükle
print(f"\n📦 Model yükleniyor...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

print(f"\n1️⃣  TİP: {type(checkpoint)}")

# Analiz
if isinstance(checkpoint, dict):
    print(f"\n2️⃣  YAPISI: Dictionary (Sözlük)")
    print(f"   Keys: {list(checkpoint.keys())}")
    
    if 'state_dict' in checkpoint:
        print(f"\n3️⃣  SONUÇ: ❌ STATE_DICT formatı")
        print(f"   → Backend'de MİMARİ tanımı gerekli!")
        state_dict = checkpoint['state_dict']
    else:
        print(f"\n3️⃣  SONUÇ: ❌ Direkt STATE_DICT")
        print(f"   → Backend'de MİMARİ tanımı gerekli!")
        state_dict = checkpoint
    
    print(f"\n📋 Layer İsimleri (ilk 10):")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"   {i+1}. {key}")
    
    print(f"\n💡 ÇÖZÜMLERİN:")
    print(f"   A) Eğitim kodundaki model class'ını bul")
    print(f"   B) Veya ONNX'e çevir")
    
else:
    print(f"\n2️⃣  YAPISI: {type(checkpoint).__name__} (Model Objesi)")
    print(f"   Module: {type(checkpoint).__module__}")
    
    try:
        checkpoint.eval()
        print(f"\n3️⃣  SONUÇ: ✅ TAM MODEL")
        print(f"   → Backend ÇALIŞIR, mimari gerekmez!")
        
        # Model detayları
        print(f"\n📋 Model Mimarisi:")
        print(checkpoint)
        
    except Exception as e:
        print(f"\n3️⃣  SONUÇ: ⚠️  Custom Class Eksik")
        print(f"   Hata: {e}")
        print(f"   → Eğitim kodundaki class'ı import etmek gerek")

print("\n" + "=" * 60)
