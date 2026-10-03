# GenAI Restoration Lab — Generative AI Assignment 1

Four generative systems, trained with PyTorch + Optuna, tracked with Weights & Biases, exported to ONNX and served
through one React + Tailwind / FastAPI web application running in Docker.

| Task | Workspace | Model |
|---|---|---|
| 1 | Universal Restoration | Convolutional denoising autoencoder with a compressed latent bottleneck (no skip connections) |
| 2 | Hard-Routed Restoration | CNN corruption classifier → one of three specialist autoencoders (clean → identity bypass) |
| 3 | Soft Mixture-of-Experts Restoration | Gate (initialised from the classifier) + identity + 3 experts, jointly fine-tuned |
| 4 | Face-to-Sketch Generator | Style-conditioned pix2pix: U-Net generator + PatchGAN discriminator, learned style embedding |

## 1. Run the application (evaluator quick start)

Requirements: Docker Desktop (with Docker Compose v2), Python 3 (only to download the models), ~2 GB disk.

```bash
git clone https://github.com/ibrahimmlk/Generative-AI-Assignment-1.git
cd Generative-AI-Assignment-1
python scripts/download_models.py      # downloads the 7 ONNX models into ./models
docker compose up --build              # one command starts backend + frontend
```

Open **http://localhost:8080**. API documentation is at http://localhost:8000/docs.

If Python is not available, download every file from the
[v1.0 release](https://github.com/ibrahimmlk/Generative-AI-Assignment-1/releases/tag/v1.0) into the `models/` folder.
Stop with `docker compose down`.

### API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | health check, loaded models, ONNX Runtime version |
| GET | `/api/system` | model metadata + PyTorch↔ONNX verification results |
| GET | `/api/samples` | unseen test images bundled with the app |
| POST | `/api/restore/universal` | Task 1 |
| POST | `/api/restore/hard` | Task 2 (classifier probabilities, predicted class, expert) |
| POST | `/api/restore/moe` | Task 3 (four routing weights, dominant branch) |
| POST | `/api/sketch` | Task 4 (`style` = 1, 2 or 3) |

Restoration endpoints take multipart form fields `file` (or `sample`), `corruption`
(`none|salt_pepper|blur|occlusion`), `severity` (`low|medium|high`) and optional `seed`.
Use `corruption=none` for an image that is already corrupted.

## 2. Repository layout

```
genai/                      training package
  corruptions.py            runtime corruptions + fixed test severities (also used by the backend)
  data.py                   Oxford-IIIT Pet / FS2K loading, splits, manifests, datasets, balanced sampler
  models.py                 autoencoder, classifier, soft MoE, U-Net generator, PatchGAN discriminator
  metrics.py                SSIM, PSNR, losses, routing-balance loss
  train_utils.py            training loops, W&B tracker, Optuna helpers
  evaluate_restoration.py   test evaluation, tables and figures for Tasks 1-3
  export.py                 ONNX export + PyTorch/ONNX Runtime consistency check
scripts/
  run_restoration.py        Tasks 1-3: prep -> Optuna -> final training -> eval -> ONNX
  run_sketch.py             Task 4: Optuna -> final training -> test -> ONNX
  download_models.py        fetches the released ONNX models
kaggle/train_all.ipynb      runs both pipelines in parallel on Kaggle (2 x T4)
results/                    Optuna studies (SQLite + JSON), manifests, metrics, figures, training histories
backend/                    FastAPI + ONNX Runtime service (Dockerfile)
frontend/                   React + Tailwind (Vite) app served by nginx (Dockerfile)
design/stitch/              original Google Stitch design export
report/                     IEEE LaTeX report
docker-compose.yml
```

## 3. Reproduce training

```bash
pip install -r requirements-train.txt
wandb login                                   # optional; logs offline without a key
python scripts/run_restoration.py --data data --out outputs --budget full
python scripts/run_sketch.py --fs2k data/FS2K --data data --out outputs_sketch --budget full
python scripts/run_restoration.py --budget smoke --data data --out out_smoke   # 10-minute CPU sanity check
```

* Oxford-IIIT Pet is downloaded automatically via torchvision; FS2K is downloaded from the authors' Google Drive
  (`gdown`). Datasets are not stored in the repository.
* Split: official trainval → 80 % train / 20 % validation (seed 42); official test set used only for final evaluation.
  FS2K: official train/test; 15 % of train held out for validation, stratified by style (seed 42).
* Training corruptions are sampled on the fly inside the `Dataset`; validation/test corruptions come from the stored
  manifests in `results/manifests/` (type, severity, seed, mask coordinates, blur settings).
* Every stage is resumable: re-running skips stages whose checkpoints exist; Optuna studies resume from the SQLite DB.
* Recommended: open `kaggle/train_all.ipynb` on Kaggle (GPU T4 ×2, Internet on, `WANDB_API_KEY` secret) and
  *Save & Run All*. Total time ≈ 2–3 h.

## 4. Experiment tracking

All Optuna trials, final runs, losses, validation metrics, sample images, checkpoints (artifacts) and test results are
logged to Weights & Biases: https://wandb.ai/m-ibrahim-malik-national-university-of-computer-and-eme/genai-a1
