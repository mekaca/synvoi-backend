from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchaudio
import os
from model import load_model, MultiResolutionFeatureExtractor

app = Flask(__name__)
CORS(app)

MODEL_PATH = '/Users/mekaca/Desktop/45/demo_model_v3.83.7.pth'

print("\n" + "="*60)
print("🚀 SYNVOI BACKEND")
print("="*60)

device = torch.device("cpu")
model, config = load_model(MODEL_PATH)
model = model.to(device)

feature_extractor = MultiResolutionFeatureExtractor()

os.makedirs("tmp", exist_ok=True)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "accuracy": "83.7%"}), 200

@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": True, "message": "No audio file"}), 400
    
    audio_file = request.files["audio"]
    filename = "".join(c for c in (audio_file.filename or "upload.wav") if c.isalnum() or c in "._-")
    temp_path = f"tmp/{filename}"
    
    try:
        audio_file.save(temp_path)
        print(f"\n📥 Analyzing: {filename}")
        
        # Load audio
        waveform, sr = torchaudio.load(temp_path)
        
        # Preprocess: 16kHz, mono, 3 seconds
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        target_len = int(16000 * 3.0)
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        elif waveform.shape[1] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        
        # Extract features
        features = feature_extractor(waveform.unsqueeze(0))
        
        # Add batch dimension and unsqueeze
        features = {k: v.unsqueeze(1).to(device) for k, v in features.items()}
        
        # Inference
        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            
            prediction = "FAKE" if pred_class == 0 else "REAL"
        
        print(f"✅ Result: {prediction} ({confidence:.1%})")
        
        return jsonify({
            "prediction": prediction,
            "confidence": float(confidence),
            "fake_prob": float(probs[0][0]),
            "real_prob": float(probs[0][1])
        })
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": True, "message": str(e)}), 500
    
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("✅ Backend ready - Model accuracy: 83.7%")
    print("   http://localhost:5001")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=True)
