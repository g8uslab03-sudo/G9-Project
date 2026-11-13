# G9-Project---KILO---

Multi-AI Autonomous Problem-Solving System for dual-cloud (GCP + Kaggle 2xT4) deployment.

## Architecture

Multi-Agent Orchestrator + Event-Driven + Microservices + CQRS + Event Sourcing + RAG + Clean Architecture + Hexagonal Architecture + LangGraph + Pydantic.

## Modules

- Main Brain / Assistant Brain
- Audit Brain / Shadow Brain
- Dream Module (5 daily hypotheses)
- Magpie Training Module (failure → fine-tune)
- Research Module (7-core closed loop)
- 250 open-source skill genes (5 categories × 50)

## Clouds

- GCP e2-micro: controller (Flask API, LangGraph)
- Kaggle 2xT4: Anchor-Guided MCTS worker, sandbox, GPU FP16, evaporation, 12h disconnect resume

## Open Sources

Hugging Face, ModelScope, GitHub, arXiv, DuckDuckGo News, WorldMonitor.

See INSTALL.md for deployment.
