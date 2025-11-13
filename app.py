import os
import re
import io
import time
import base64
import hashlib
import urllib.request
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS

# PyTorch/Torchaudio imports
import torch
import torchaudio
from torchaudio.functional import resample

# Model imports
from model import ConformerVoiceDetector, MultiResolutionFeatureExtractor


# =========================
# Konfig
# =========================
APP_NAME = "synvoi-backend"

# Model konumu
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/mekaca/synvoi-backend/releases/download/v0.1/demo_model_v3.83.7.pth",
).strip()

WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/opt/render/project/src/weights").strip()
MODEL_FILENAME = os.getenv("MODEL_FILENAME", os.path.basename(MODEL_URL)).strip()
MODEL_LOCAL = os.path.join(WEIGHTS_DIR, MODEL_FILENAME)

# Private GitHub ise PAT girin
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()

# SHA256 doğrulaması
RAW_SHA256 = os.getenv("MODEL_SHA256", "").strip()
MODEL_SHA256 = RAW_SHA256 if re.fullmatch(r"[0-9a-fA-F]{64}", RAW_SHA256 or "") else ""

# Kontrol env'leri
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0").strip() == "1"
DELAY_MODEL_LOAD = int(os.getenv("DELAY_MODEL_LOAD", "0").strip() or "0")

# API ayarları
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*").strip()
API_KEY = os.getenv("API_KEY", "").strip()
MAX_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "20"))
TARGET_SR = 16000


# =========================
# Yardımcılar
# =========================
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_headers(url: str, dst: str, sha256: Optional[str] = None,
                           retries: int = 3, timeout: int = 120) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    headers = {"User-Agent": f"{APP_NAME}/1.0 (+python-urllib)", "Accept": "*/*"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    req = urllib.request.Request(url, headers=headers)
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
                f.write(r.read())
            break
        except Exception as e:
            last_err = e
            time.sleep(min(10, 2 * attempt))
    else:
        raise last_err or RuntimeError("unknown download error")

    if sha256:
        got = _sha256(dst)
        if got.lower() != sha256.lower():
            os.remove(dst)
            raise RuntimeError(f"Checksum mismatch for {dst}: got {got}, want {sha256}")


def ensure_weights() -> None:
    print(f"[bootstrap] ENV MODEL_URL: {MODEL_URL}")
    print(f"[bootstrap] TARGET FILE : {MODEL_LOCAL}")
    if os.path.exists(MODEL_LOCAL) and os.path.getsize(MODEL_LOCAL) > 0:
        print("[bootstrap] model mevcut, indirme atlandı")
        return
    print(f"[bootstrap] model indiriliyor: {MODEL_URL} -> {MODEL_LOCAL}")
    _download_with_headers(MODEL_URL, MODEL_LOCAL, sha256=MODEL_SHA256 or None)
    print("[bootstrap] model indirildi:", MODEL_LOCAL)


# =========================
# Model yükleme
# =========================
model = None
model_ready = False
feature_extractor = None


def initialize_feature_extractor():
    """Feature extractor'ı başlat"""
    global feature_extractor
    feature_extractor = MultiResolutionFeatureExtractor(sample_rate=TARGET_SR)
    print("[bootstrap] Feature extractor başlatıldı")


def load_model_safe() -> None:
    """Model yüklemesini güvenli yap"""
    global model, model_ready

    if SKIP_MODEL_LOAD:
        print("[bootstrap] SKIP_MODEL_LOAD=1, model yüklenmeyecek")
        model = None
        model_ready = False
        return

    if DELAY_MODEL_LOAD > 0:
        print(f"[bootstrap] DELAY_MODEL_LOAD={DELAY_MODEL_LOAD}s")
        time.sleep(DELAY_MODEL_LOAD)

    try:
        ensure_weights()

        # Gerçek model yükleme
        print("[bootstrap] Conformer model yükleniyor...")
        
        # Model oluştur
        model = ConformerVoiceDetector(
            num_classes=2,
            d_model=384,
            n_conformer_blocks=6,
            n_heads=12,
            num_segments=8,  # Deployment için 8 kullan
            rnn_hidden=384,
            rnn_layers=2
        )
        
        # CPU'da çalıştır
        device = torch.device("cpu")
        
        # Checkpoint yükle
        checkpoint = torch.load(MODEL_LOCAL, map_location=device)
        
        # State dict'i yükle
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("[bootstrap] model_state_dict kullanıldı")
        else:
            # Eğer doğrudan state_dict ise
            model.load_state_dict(checkpoint)
            print("[bootstrap] Doğrudan state_dict yüklendi")
        
        model.to(device)
        model.eval()
        
        model_ready = True
        print("[bootstrap] ✅ Conformer model başarıyla yüklendi")
        
        # Feature extractor'ı da başlat
        initialize_feature_extractor()
        
    except Exception as e:
        print(f"[bootstrap] ❌ Model yükleme hatası: {e}")
        model = None
        model_ready = False


# =========================
# Audio yardımcıları
# =========================
def _to_mono_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = resample(wav, sr, TARGET_SR)
    return wav


def _load_wav_bytes(b: bytes) -> torch.Tensor:
    t, sr = torchaudio.load(io.BytesIO(b))
    return _to_mono_16k(t, sr)


def _load_wav_url(url: str) -> torch.Tensor:
    with urllib.request.urlopen(url, timeout=20) as r:
        data = r.read()
    return _load_wav_bytes(data)


# =========================
# Inference
# =========================
@torch.inference_mode()
def run_inference(model, wav_16k: torch.Tensor) -> dict:
    """Gerçek Conformer model inference"""
    
    # Waveform'u 3 saniyeye ayarla
    target_length = TARGET_SR * 3  # 3 saniye
    current_length = wav_16k.shape[-1]
    
    # Uzunluk ayarlaması
    if current_length > target_length:
        # Ortadan kes
        start = (current_length - target_length) // 2
        wav_16k = wav_16k[:, start:start + target_length]
    elif current_length < target_length:
        # Padding ekle
        pad_amount = target_length - current_length
        pad_left = pad_amount // 2
        pad_right = pad_amount - pad_left
        wav_16k = torch.nn.functional.pad(wav_16k, (pad_left, pad_right))
    
    # Feature extraction
    if feature_extractor is None:
        initialize_feature_extractor()
    
    features = feature_extractor(wav_16k)
    
    # Normalize features
    for key in ['mel_512', 'mel_1024', 'mfcc']:
        if key in features:
            feat = features[key]
            mean = feat.mean(dim=(1, 2), keepdim=True)
            std = feat.std(dim=(1, 2), keepdim=True).clamp(min=1e-5)
            features[key] = (feat - mean) / std
    
    # Batch dimension ekle
    for key in features:
        features[key] = features[key].unsqueeze(0)
    
    # Model inference
    output = model(features)
    
    # Softmax uygula
    probs = torch.nn.functional.softmax(output, dim=1)
    
    # Sonuçları al
    fake_prob = probs[0, 1].item()  # Fake class probability
    real_prob = probs[0, 0].item()  # Real class probability
    
    # Karar ver
    label = "fake" if fake_prob > real_prob else "real"
    confidence = max(fake_prob, real_prob)
    
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4)
    }


# =========================
# Flask app
# =========================
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
CORS(app, resources={r"/infer": {"origins": FRONTEND_ORIGIN}}, supports_credentials=False)

# Gunicorn altında modül import edilir edilmez modeli yükle
load_model_safe()


def _check_api_key() -> bool:
    if not API_KEY:
        return True
    return request.headers.get("x-api-key") == API_KEY


@app.route("/", methods=["GET", "HEAD"])
def root():
    return jsonify(
        status="ok",
        service=APP_NAME,
        model_ready=model_ready,
        model_file=os.path.basename(MODEL_LOCAL) if MODEL_LOCAL else "none",
    ), 200


@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200


@app.route("/version", methods=["GET"])
def version():
    commit = os.getenv("RENDER_GIT_COMMIT", "")[:7]
    return jsonify(
        service=APP_NAME,
        commit=commit,
        python=os.sys.version.split()[0],
        model_url=MODEL_URL,
        model_file=os.path.basename(MODEL_LOCAL),
        sha256_enabled=bool(MODEL_SHA256),
        skip=SKIP_MODEL_LOAD,
        delay=DELAY_MODEL_LOAD,
        cors_origin=FRONTEND_ORIGIN,
        model_ready=model_ready,
    ), 200


@app.route("/infer", methods=["POST"])
def infer():
    global model_ready, model
    if not model_ready or model is None:
        return jsonify(error="Model not loaded"), 503
    if not _check_api_key():
        return jsonify(error="Unauthorized"), 401

    try:
        wav: Optional[torch.Tensor] = None

        # 1) multipart: "file"
        if "file" in request.files:
            f = request.files["file"]
            wav = _load_wav_bytes(f.read())

        # 2) JSON body (wav_url veya wav_base64)
        if wav is None:
            data = request.get_json(silent=True) or {}
            if "wav_url" in data and data["wav_url"]:
                wav = _load_wav_url(data["wav_url"])
            elif "wav_base64" in data and data["wav_base64"]:
                b64 = data["wav_base64"]
                if b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                wav = _load_wav_bytes(base64.b64decode(b64))

        if wav is None:
            return jsonify(error="No audio provided. Use multipart 'file', or JSON 'wav_url'/'wav_base64'."), 400

        # Inference yap
        result = run_inference(model, wav)
        
        # Duration hesapla
        dur_s = round(wav.shape[-1] / TARGET_SR, 3)
        
        return jsonify(
            ok=True,
            **result,
            sample_rate=TARGET_SR,
            duration_sec=dur_s,
        ), 200

    except Exception as e:
        print(f"[ERROR] Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify(error="inference_failed", detail=str(e)), 500


if __name__ == "__main__":
    # Lokal denemelerde de çalışsın
    if not model_ready:
        load_model_safe()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
