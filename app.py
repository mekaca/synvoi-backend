import os
import time
import hashlib
import urllib.request
from typing import Optional

from flask import Flask, jsonify, request

APP_NAME = "synvoi-backend"

# ENV’den oku; yoksa sizin doğruladığınız asset’a düş
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/mekaca/synvoi-backend/releases/download/v0.1/demo_model_v3.83.7.pth",
).strip()

WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/opt/render/project/src/weights").strip()
MODEL_FILENAME = os.getenv("MODEL_FILENAME", os.path.basename(MODEL_URL)).strip()
MODEL_LOCAL = os.path.join(WEIGHTS_DIR, MODEL_FILENAME)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
MODEL_SHA256 = os.getenv("MODEL_SHA256", "").strip()

model = None
model_ready = False


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_headers(url: str, dst: str, sha256: Optional[str] = None, retries: int = 3, timeout: int = 120):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    headers = {
        "User-Agent": f"{APP_NAME}/1.0 (+python-urllib)",
        "Accept": "*/*",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"

    req = urllib.request.Request(url, headers=headers)
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
                f.write(r.read())
            break
        except Exception as e:
            last_err = e
            time.sleep(min(10, 2 * attempt))
    else:
        raise last_err

    if sha256:
        got = _sha256(dst)
        if got.lower() != sha256.lower():
            os.remove(dst)
            raise RuntimeError(f"Checksum mismatch for {dst}: got {got}, want {sha256}")


def ensure_weights():
    print(f"[bootstrap] ENV MODEL_URL: {MODEL_URL}")
    print(f"[bootstrap] TARGET FILE : {MODEL_LOCAL}")
    if os.path.exists(MODEL_LOCAL) and os.path.getsize(MODEL_LOCAL) > 0:
        print("[bootstrap] model mevcut, indirme atlandı")
        return
    print(f"[bootstrap] model indiriliyor: {MODEL_URL} -> {MODEL_LOCAL}")
    _download_with_headers(MODEL_URL, MODEL_LOCAL, sha256=MODEL_SHA256 or None)
    if os.path.exists(MODEL_LOCAL):
        print("[bootstrap] model indirildi:", MODEL_LOCAL)


def load_model_safe():
    global model, model_ready
    try:
        ensure_weights()

        try:
            import torch  # noqa: F401
        except Exception as e:
            print(f"[bootstrap] torch import edilemedi: {e}")
            model = None
            model_ready = False
            return

        # Burada kendi model yüklemenizi yapın.
        # örn:
        # import torch
        # model = torch.load(MODEL_LOCAL, map_location="cpu")
        # model.eval()

        model = object()  # placeholder
        model_ready = True
        print("[bootstrap] model yüklendi (placeholder)")
    except Exception as e:
        print(f"[bootstrap] model yükleme hatası: {e}")
        model = None
        model_ready = False


app = Flask(__name__)

@app.route("/", methods=["GET", "HEAD"])
def root():
    return jsonify(status="ok", service=APP_NAME, model_ready=model_ready, model_file=os.path.basename(MODEL_LOCAL)), 200

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

@app.route("/version", methods=["GET"])
def version():
    commit = os.getenv("RENDER_GIT_COMMIT", "")[:7]
    return jsonify(service=APP_NAME, commit=commit, python=os.sys.version.split()[0], model_url=MODEL_URL, model_file=os.path.basename(MODEL_LOCAL)), 200

@app.route("/infer", methods=["POST"])
def infer():
    if not model_ready or model is None:
        return jsonify(error="Model not loaded"), 503
    data = request.get_json(silent=True) or {}
    # result = run_inference(model, data)
    result = {"ok": True, "detail": "placeholder"}
    return jsonify(result), 200


if __name__ == "__main__":
    print("\n" + "=" * 60 + f"\n{APP_NAME.upper()}\n" + "=" * 60)
    load_model_safe()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
