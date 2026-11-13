# G9 System Architecture

## Overview

Multi-AI autonomous problem-solving system spanning GCP e2-micro (24x7 controller) and Kaggle 2xT4 (compute worker).

## Architecture Patterns

- Multi-Agent Orchestrator
- Event-Driven (CQRS commands/queries + Event Sourcing)
- Microservices (separate GCP controller & Kaggle worker)
- CQRS (commands vs queries separated)
- Event Sourcing (all state changes stored as events)
- Clean Architecture (domain/application/infrastructure separation)
- Hexagonal Architecture (ports & adapters)
- LangGraph state graph (macro planner)
- Pydantic validation (all contracts)

## Core Modules

1. **Main Brain** - central coordinator, requirement understanding, problem decomposition, skill selection
2. **Assistant Brain** - work allocation & release coordination
3. **Audit Brain** - result/artifact approval gatekeeper
4. **Shadow Brain** - reverse/blind-spot thinking, challenges decisions
5. **Dream Module** - 5 daily hypotheses from open sources, requires audit approval
6. **Magpie Module** - failure capture, distillation, fine-tuning schedules
7. **Research Module** - 7-core closed loop (planner, literature, hypothesis, sandbox, analysis, paper, artifact)

## Data Flow

1. User submits problem → GCP `/jobs`
2. Main Brain decomposes → selects from 250 skill genes
3. Assistant Brain allocates work
4. Audit + Shadow review
5. Workspace evaporates (checkpoint saved)
6. Handoff to Kaggle T4x2 via Anchor-Guided MCTS
7. Kaggle executes in sandbox, clears VRAM, evaporates, saves checkpoint
8. Results persisted; failures → Magpie → fine-tuning

## Persistence Layers

- Atomic memory (dedup by SHA-256 fingerprint)
- Manifest (ordering + relationships)
- Merkle graph (integrity)
- TurboQuant vector index (1-bit compression candidate)
- Checkpoint store (ZIP + SHA-256, 12h disconnect resume)

## Open-Source Sources

Hugging Face, ModelScope, GitHub, arXiv, DuckDuckGo News, WorldMonitor.

## Skill Catalog

250 elite skill genes: 5 categories × 50 keys. Each maps to a downloadable open-source implementation.
