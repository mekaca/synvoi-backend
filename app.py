from flask import Flask, request, jsonify
from flask_cors import CORS
import os, sys, math, time, hashlib, urllib.request, traceback
import torch
import torchaudio

from model import load_model, MultiResolutionFeatureExtractor

app = Flask(__name__)
CORS(app)

# =========================
# Ortam değişkenleri
# =========================
MODEL_URL    = os.environ.get("MODEL_URL", "https://github.com/mekaca/synvoi-backend/releases/download/v0.1/demo_model_v3.83.7.pth").strip()
MODEL_SHA256 = os.environ.get("MODEL_SHA256", "").strip()  # opsiyonel
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()  # private release ise

WEIGHTS_DIR  = "weights"
MODEL_LOCAL  = os.path.join(WEIGHTS_DIR, "model.pth")
TMP_DIR      = "tmp"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

print("\n" + "="*60)
print("SYNVOI BACKEND")
print("="*60)

# =========================
# İndirme + doğrulama
# =========================
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _download_with_headers(url: str, dst: str):
    headers = {"User-Agent": "synvoi-backend"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        ctype = r.info().get_content_type()
        total = r.length or 0
        if ctype.startswith("text/"):
            raise RuntimeError(f"Beklenmeyen içerik türü: {ctype} (muhtemelen 403/404 HTML)")
        got = 0
        block = 1024 * 1024
        while True:
            chunk = r.read(block)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = math.floor(got * 100 / total)
                print(f"\r[bootstrap] indiriliyor: {pct}% ({got//(1<<20)}/{total//(1<<20)} MB)", end="")
                sys.stdout.flush()
    print()

def ensure_weights():
    if os.path.exists(MODEL_LOCAL):
        if MODEL_SHA256:
            got = _sha256(MODEL_LOCAL)
            if got != MODEL_SHA256:
                raise RuntimeError(f"Model hash mismatch: {got} != {MODEL_SHA256}")
        return
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL boş. Releases indirme linkini ayarla.")
    print(f"[bootstrap] model indiriliyor: {MODEL_URL}")
    _download_with_headers(MODEL_URL, MODEL_LOCAL)
    if MODEL_SHA256:
        got = _sha256(MODEL_LOCAL)
        if got != MODEL_SHA256:
            raise RuntimeError(f"Model hash mismatch: {got} != {MODEL_SHA256}")

# =========================
# Modeli yükle
# =========================
device = torch.device("cpu")
feature_extractor = MultiResolutionFeatureExtractor()

try:
    ensure_weights()
    model, _cfg = load_model(MODEL_LOCAL, device=device)
    model = model.to(device)
    print("[bootstrap] model yüklendi.")
except Exception as e:
    print(f"[bootstrap] model yükleme hatası: {e}", file=sys.stderr)
    traceback.print_exc()
    # Servisi ayakta tutalım; /analyze çağrısında hata dönecek.
    model = None

# =========================
# Health
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "has_weights": os.path.exists(MODEL_LOCAL),
        "model_loaded": model is not None,
        "accuracy": "83.7%"
    }), 200

# =========================
# Inference
# =========================
@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": True, "message": "No audio file"}), 400

    if model is None:
        return jsonify({"error": True, "message": "Model yüklenemedi. Logs'u kontrol edin."}), 503

    audio_file = request.files["audio"]
    filename = "".join(c for c in (audio_file.filename or "upload.wav") if c.isalnum() or c in "._-")
    temp_path = os.path.join(TMP_DIR, filename)

    try:
        audio_file.save(temp_path)

        # Load audio
        waveform, sr = torchaudio.load(temp_path)

        # Preprocess: 16kHz, mono, 3s
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        target_len = int(16000 * 3.0)
        T = waveform.shape[1]
        if T > target_len:
            waveform = waveform[:, :target_len]
        elif T < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - T))

        # Features
        feats = feature_extractor(waveform.unsqueeze(0))  # dict
        feats = {k: v.unsqueeze(1).to(device) for k, v in feats.items()}  # (kendi model sözleşmene göre)

        # Inference
        with torch.no_grad():
            output = model(feats)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            prediction = "FAKE" if pred_class == 0 else "REAL"

        return jsonify({
            "prediction": prediction,
            "confidence": float(confidence),
            "fake_prob": float(probs[0][0]),
            "real_prob": float(probs[0][1])
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": True, "message": str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

# =========================
# Run
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Running on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
