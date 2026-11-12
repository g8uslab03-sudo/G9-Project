# Multi-Agent Autonomous Problem Solving System

A production-ready, multi-AI core autonomous problem solving system designed for extreme cloud constraints (GCP Free Tier + Kaggle 2xT4) with 24x7 operation capability.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Orchestrator                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ Main    │ │ Sub     │ │ Audit   │ │ Shadow  │ │ Dream   │       │
│  │ Brain   │ │ Brain   │ │ Brain   │ │ Brain   │ │ Module  │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
│       │           │           │           │           │             │
│       └───────────┼───────────┼───────────┼───────────┘             │
│                   ▼           ▼           ▼                         │
│        ┌─────────────────────────────────────────┐                 │
│        │           Magpie Training               │                 │
│        └─────────────────────────────────────────┘                 │
│                   │                                                 │
│                   ▼                                                 │
│        ┌─────────────────────────────────────────┐                 │
│        │        Research Pipeline (7 phases)     │                 │
│        └─────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

## 🧠 Six Brain Modules

| Module | Role | Key Responsibilities |
|--------|------|---------------------|
| **Main Brain** | Central Coordination | Problem analysis, strategy selection, resource allocation, bottleneck detection |
| **Sub Brain** | Task Distribution | Task decomposition, agent assignment, load balancing, progress monitoring |
| **Audit Brain** | Quality Gate | Result approval, compliance check, risk assessment, quality validation |
| **Shadow Brain** | Adversarial Analysis | Blindspot detection, counterargument generation, assumption challenging |
| **Dream Module** | Hypothesis Generation | Daily 5 hypotheses from open sources, audit-gated, work order creation |
| **Magpie Module** | Failure Distillation | Collect failures → distill patterns → create training data → fine-tune agents |

## 🔬 Research Pipeline (7 Phases)

1. **Analyze & Plan** - Problem analysis and work division
2. **Literature Review** - Open source paper collection (arXiv, Semantic Scholar, PubMed, etc.)
3. **Hypothesis Generation** - Anchor-based hypotheses from literature
4. **Experiment Execution** - Sandbox-verified experiments (Python/Jupyter/Docker)
5. **Data Analysis** - Statistical analysis and conclusion generation
6. **Paper Generation** - Full research paper with LaTeX-ready sections
7. **Artifact Production** - Algorithms, tools, skills (non-physical artifacts)

## 🎯 250 Elite Skills (5 Categories × 50)

| Category | Skills | Examples |
|----------|--------|----------|
| **Information Retrieval** | 50 | Real-time web search, RAG, fact-checking, knowledge graphs |
| **Logical Reasoning** | 50 | Text-to-SQL, statistical analysis, causal inference, decision matrices |
| **Content Creation** | 50 | SEO writing, social media, ad copy, technical documentation |
| **Visual Design** | 50 | Image generation, video creation, 3D assets, voice synthesis |
| **Programming** | 50 | Code generation, debugging, CI/CD, architecture design, multi-agent orchestration |

**Skill Acquisition Strategy**: Parallel open-source download + research synthesis → audit comparison → promote best

## 💾 Memory System (Tiered Storage)

```
Persistent Memory (Reconstructible)
├── A. Raw Objects: Documents, work orders, experiments, audit evidence
├── B. Atomic Memory: Minimum retrievable units with embeddings
├── C. Manifest: Atomic ordering, source & object relationships
├── D. Graph: Relationships between atoms
└── E. Vector Index: Semantic retrieval (RaBitQ compressed)

Working Memory (Ephemeral)
├── GPU Vector Index Cache
└── Current Task Memory Fragments
```

## ⚡ Compression & Optimization

| Technique | Purpose | Configuration |
|-----------|---------|---------------|
| **FP16/BF16** | Inference precision | Default for all models |
| **Mixed Precision** | Training | BF16 + gradient scaling |
| **RaBitQ** | Vector index compression | 75% size reduction, 95%+ recall |
| **TurboQuant** | Vector quantization backend | Optimized for similarity search |
| **1-bit Adam** | Gradient compression | Distributed training synchronization |
| **Gradient Checkpointing** | Activation memory | 75% activation memory reduction |
| **CPU Offloading** | Large model support | Tiered: GPU → CPU RAM → Disk |

## 🚀 Deployment Targets

### GCP Free Tier (Cloud Run)
- **Services**: 9 microservices (api-gateway, orchestrator, memory, rag, skills, research, dream, magpie, monitoring)
- **Resources**: 0.5-2 vCPU, 0.5-4 GiB RAM per service
- **Scaling**: 0 to N instances, concurrency 1-80
- **Cost**: ~$0/month within free tier limits

### Kaggle 2×T4 (30 hrs/week)
- **GPU 0**: Main Brain + Magpie Training (scheduled)
- **GPU 1**: Sub Brain + Research Experiments (scheduled)
- **CPU**: Audit/Shadow Brains, Dream Module, Memory/RAG, Skill Registry
- **Memory**: 16 GiB per GPU, 16 GiB system RAM
- **Storage**: 20 GiB persistent, 50 GiB ephemeral

### Local Development
- **Single container**: All services in one (docker-compose)
- **Hot reload**: Source code mounted for development
- **Full resources**: Up to 8 vCPU, 32 GiB RAM, 1×GPU

## 📦 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA Container Toolkit (for GPU)
- Python 3.11+ (for local development)

### Local Development
```bash
# Clone and enter
git clone <repo>
cd multi-agent-system

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Test API
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"user_query": "How to optimize a Python loop?", "domain": "programming"}'
```

### Kaggle Deployment
```bash
# In Kaggle notebook
!git clone <repo>
%cd multi-agent-system

# Start services
!docker-compose -f docker-compose.kaggle.yml up -d

# Check GPU allocation
!docker exec kaggle-main-brain nvidia-smi
!docker exec kaggle-sub-brain nvidia-smi
```

### GCP Cloud Run
```bash
# Make deploy script executable
chmod +x deploy/gcp/cloudrun_deploy.sh

# Deploy all services
./deploy/gcp/cloudrun_deploy.sh

# Or deploy individually
gcloud run deploy multi-agent-api-gateway --source . --region us-central1 --allow-unauthenticated
```

## 🔧 API Usage

### Solve a Problem
```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "Design a caching strategy for a high-traffic API",
    "domain": "architecture",
    "constraints": {"latency_ms": 100, "memory_mb": 512},
    "priority": 1
  }'
```

### Acquire a Skill
```bash
curl -X POST http://localhost:8000/skills/acquire \
  -H "Content-Type: application/json" \
  -d '{"gene_key": "SKILL_PROGRAMMING_001"}'
```

### Start Research
```bash
curl -X POST http://localhost:8000/research/start \
  -H "Content-Type: application/json" \
  -d '{
    "problem_id": "uuid-here",
    "context": {"user_query": "Novel transformer architecture", "domain": "ml"}
  }'
```

### System Health
```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

## 🏃 Pipeline Execution Flow

```
User Query
    │
    ▼
┌─────────────────────┐
│ 1. Receive Problem  │ ──► ProblemAggregate created
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 2. Understand       │ ──► Main Brain analyzes & extracts constraints
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 3. Decompose        │ ──► Skill-based sub-problem creation
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 4. Dispatch Experts │ ──► Sub Brain assigns agents (skill-matched)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 5. Parallel Reason  │ ──► Agents execute skills (LLM + sandbox)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 6. Mutual Verify    │ ──► Cross-agent verification
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 7. Debate Evaluate  │ ──► Shadow Brain generates counterarguments
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 8. Integrate Answer │ ──► Synthesis with confidence scoring
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 9. Continuous Learn │ ──► Failures → Magpie → Training data
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 10. Optimize Future │ ──► Agent performance updates
└─────────────────────┘
```

## 🔒 Security & Safety

- **Sandbox Isolation**: Full network isolation, resource limits, timeout enforcement
- **Audit Gates**: Every research phase requires Audit Brain approval
- **Skill Validation**: Static analysis + sandbox execution + audit review
- **No Network Access**: Research sandboxes run without internet
- **Resource Quotas**: Per-sandbox CPU, memory, disk, time limits

## 📊 Monitoring & Observability

- **Health Checks**: Per-component health with status aggregation
- **Metrics**: Prometheus-format metrics (counters, gauges, histograms)
- **Tracing**: OpenTelemetry-ready (Jaeger, Zipkin compatible)
- **Logging**: Structured JSON logging with correlation IDs
- **Checkpointing**: Automatic pipeline state persistence every 5 minutes

## 🔄 Resilience Features

- **Checkpoint/Resume**: Pipeline state saved at each stage, resumable after crash
- **Circuit Breakers**: Automatic failure isolation with exponential backoff
- **Graceful Degradation**: CPU fallback when GPU unavailable
- **Auto-scaling**: Cloud Run scales to zero when idle
- **24x7 Operation**: Dream/Magpie cycles scheduled, self-healing workers

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_system.py::TestProblemAggregate -v
pytest tests/test_system.py::TestSkillRegistry -v
pytest tests/test_system.py::TestMemoryConfig -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📁 Project Structure

```
multi-agent-system/
├── src/
│   ├── main.py                    # Application entry point
│   ├── shared/                    # Shared kernel (CQRS, Event Sourcing, DDD)
│   │   ├── kernel/               # Base classes, commands, queries, events
│   │   ├── observability/        # Logging, metrics, tracing, health
│   │   ├── exceptions.py         # Domain exceptions
│   │   ├── constants.py          # System constants
│   │   └── utils.py              # Utilities
│   ├── domain/                   # Domain layer (Clean Architecture)
│   │   ├── entities.py           # Domain entities & value objects
│   │   ├── aggregates.py         # Aggregate roots
│   │   ├── services.py           # Domain services
│   │   └── repositories.py       # Repository interfaces (ports)
│   ├── application/              # Application layer
│   │   ├── commands.py           # Command definitions
│   │   ├── queries.py            # Query definitions
│   │   ├── handlers.py           # Command handlers
│   │   ├── query_handlers.py     # Query handlers
│   │   ├── services.py           # Application services
│   │   ├── dtos.py               # Data transfer objects
│   │   └── ports.py              # Hexagonal architecture ports
│   └── infrastructure/           # Infrastructure layer
│       ├── orchestrator/
│       │   └── brains.py         # 6 brain module implementations
│       ├── memory/
│       │   └── memory_system.py  # Tiered memory with compression
│       ├── rag/
│       │   └── rag_system.py     # RAG pipeline + open source search
│       ├── skills/
│       │   └── skill_system.py   # 250 skill registry & execution
│       ├── research/
│       │   └── research_pipeline.py # 7-phase research pipeline
│       ├── compression/
│       │   └── compression.py    # FP16, RaBitQ, TurboQuant, 1-bit Adam
│       └── deployment/
│           └── deployment_config.py # GCP/Kaggle configs
├── tests/
│   └── test_system.py            # Comprehensive test suite
├── deploy/
│   ├── gcp/                      # GCP Cloud Run deployment
│   └── kaggle/                   # Kaggle-specific configs
├── docker-compose.yml            # Local development
├── docker-compose.kaggle.yml     # Kaggle 2×T4 deployment
├── Dockerfile                    # Production image
├── Dockerfile.kaggle             # Kaggle-optimized image
├── requirements.txt              # Core dependencies
├── requirements-ml.txt           # ML dependencies
└── README.md                     # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with Clean Architecture, Hexagonal Architecture, CQRS, Event Sourcing
- Inspired by multi-agent systems research (AutoGPT, BabyAGI, MetaGPT)
- Optimized for constrained environments (GCP Free Tier, Kaggle)
- Uses open-source models and data exclusively