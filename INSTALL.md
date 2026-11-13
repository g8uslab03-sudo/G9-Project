# G9 Autonomous Multi-AI System - Installation & Deployment Guide

## Architecture

- **GCP e2-micro** (24x7 controller): Flask API + LangGraph macro planner + Main/Assistant/Audit/Shadow brains + Dream/Magpie/Research modules
- **Kaggle 2xT4** (compute worker): Anchor-Guided MCTS + sandbox + low-precision GPU inference + evaporation + 12h disconnect resume

## Prerequisites

- Python 3.11+
- GCP project with Cloud Run (or Compute Engine e2-micro)
- Kaggle account with GPU-enabled notebooks
- Docker (for GCP deployment)

## Directory Layout

```
shared/          # Domain contracts, application logic, infrastructure, persistence
gcp/             # GCP controller (Flask API, LangGraph, deploy)
kaggle/          # Kaggle T4x2 worker (MCTS, sandbox)
configs/         # system.yaml, market-sources.yaml
tests/           # Unit tests
docs/            # Documentation
```

## Local Development

```bash
# Install GCP controller deps
pip install -r gcp/requirements.txt

# Install Kaggle worker deps (on Kaggle T4x2)
pip install -r kaggle/requirements.txt

# Run GCP controller locally
cd gcp && python app.py
# API: http://localhost:8080

# Endpoints:
# POST /jobs          - Submit a problem (runs macro planner, evaporates, hands to Kaggle)
# GET  /jobs/{id}     - Get job checkpoint
# GET  /checkpoints/{id}
# POST /research      - Run 7-step research loop
# POST /dream         - Generate 5 daily hypotheses
# POST /magpie/failure - Record a failure for Magpie training
# POST /magpie/finetune - Schedule fine-tuning
# POST /evaporate/{id} - Manual workspace evaporation
# GET  /skills        - List 250 skill genes
# GET  /health        - Health check
```

## Kaggle Worker

On the Kaggle 2xT4 notebook:
```python
from kaggle.worker.worker import worker
import shared  # ensure shared/ is on PYTHONPATH

job = Job(...)  # from shared.contracts
result = worker.dispatch_from_gcp(job, gcp_state)
```

## Deployment

### GCP Cloud Run
```bash
cd gcp/deploy
gcloud builds submit --config cloudbuild.yaml
gcloud run deploy g9-orchestrator --image gcr.io/$PROJECT_ID/g9-orchestrator:latest \
  --region us-central1 --platform managed --allow-unauthenticated --memory 512Mi --cpu 1
```

### Docker
```bash
docker build -t g9-orchestrator -f gcp/deploy/Dockerfile .
docker run -p 8080:8080 g9-orchestrator
```

## Key Policies

1. **Open-source only**: All models/skills/data downloaded from Hugging Face, ModelScope, GitHub, arXiv, DuckDuckGo, WorldMonitor. No proprietary/closed sources.
2. **Evaporation**: Workspace fully cleared between pipeline handoffs; checkpoints saved before evaporation for 12h disconnect resume.
3. **GPU VRAM**: FP16/BF16 low-precision loading; `clear_vram()` between steps; no cross-GPU large object transfer.
4. **TurboQuant**: 1-bit vector index compression available as candidate backend.
5. **Security audit**: Sandbox code audited for malicious patterns before execution.

## Testing

```bash
pip install pytest
pytest tests/
```

## License

MIT
