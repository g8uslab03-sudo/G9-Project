"""
System constants and configuration defaults.
"""
from __future__ import annotations
from enum import Enum
from typing import Final
from dataclasses import dataclass


# System Constants
SYSTEM_NAME: Final[str] = "Multi-Agent Autonomous Problem Solving System"
SYSTEM_VERSION: Final[str] = "1.0.0"

# Cloud Constraints
GCP_FREE_TIER_LIMITS: Final[dict] = {
    "compute_engine": {
        "f1_micro_hours_per_month": 720,
        "regions": ["us-east1", "us-west1", "us-central1"]
    },
    "cloud_run": {
        "requests_per_month": 2_000_000,
        "cpu_seconds_per_month": 360_000,
        "memory_gb_seconds_per_month": 180_000
    },
    "cloud_storage": {
        "storage_gb": 5,
        "class_a_operations": 5_000,
        "class_b_operations": 50_000
    },
    "pub_sub": {
        "messages_per_month": 10_000_000
    }
}

KAGGLE_LIMITS: Final[dict] = {
    "gpu": {
        "type": "T4",
        "count": 2,
        "vram_gb": 16,
        "weekly_hours": 30
    },
    "cpu": {
        "cores": 4,
        "ram_gb": 16
    },
    "storage": {
        "persistent_gb": 20,
        "ephemeral_gb": 50
    },
    "network": {
        "egress_gb_per_week": 20
    }
}

# Memory Management
MEMORY_CONSTANTS: Final[dict] = {
    "vector_index_compression": {
        "algorithm": "RaBitQ",
        "target_compression_ratio": 0.25,
        "min_recall": 0.95
    },
    "model_quantization": {
        "inference": "FP16",
        "training": "BF16_mixed",
        "turboquant_enabled": True
    },
    "gradient_compression": {
        "algorithm": "1bit_Adam",
        "enabled_for_distributed": True
    },
    "offloading": {
        "cpu_ram_threshold_gb": 12,
        "disk_threshold_gb": 50,
        "prefetch_batches": 2
    }
}

# Pipeline Constants
PIPELINE_CONSTANTS: Final[dict] = {
    "stages": [
        "receive_problem",
        "understand_requirements",
        "decompose_problem",
        "dispatch_experts",
        "parallel_reasoning",
        "mutual_verification",
        "debate_evaluation",
        "integrate_answer",
        "continuous_learning",
        "optimize_future"
    ],
    "handoff_policy": "clean_workspace",
    "checkpoint_interval_seconds": 300,
    "max_stage_duration_seconds": 3600,
    "retry_attempts": 3,
    "circuit_breaker_threshold": 5
}

# Skill System
SKILL_CONSTANTS: Final[dict] = {
    "categories": 5,
    "skills_per_category": 50,
    "total_skills": 250,
    "gene_key_format": "SKILL_{category}_{index:03d}",
    "acquisition_strategy": "parallel_download_and_research",
    "audit_before_promotion": True
}

# Research Module
RESEARCH_CONSTANTS: Final[dict] = {
    "phases": 7,
    "phase_names": [
        "analyze_and_plan",
        "literature_review",
        "hypothesis_generation",
        "experiment_execution",
        "data_analysis",
        "paper_generation",
        "artifact_production"
    ],
    "sandbox_isolation": True,
    "audit_required": True,
    "max_iterations_per_phase": 3
}

# Dream Module
DREAM_CONSTANTS: Final[dict] = {
    "hypotheses_per_day": 5,
    "audit_required": True,
    "source_databases": [
        "arxiv",
        "pubmed",
        "semantic_scholar",
        "openalex",
        "huggingface_papers",
        "papers_with_code",
        "github_trending",
        "hacker_news",
        "lobsteers",
        "rss_feeds"
    ],
    "rejection_retry": True
}

# Magpie Training
MAGPIE_CONSTANTS: Final[dict] = {
    "collection_scope": "all_modules",
    "distillation_method": "failure_pattern_extraction",
    "training_schedule": "daily_batch",
    "fine_tune_per_core": True,
    "checkpoint_retention": 10
}

# Brain Modules
BRAIN_MODULES: Final[dict] = {
    "main_brain": {
        "role": "coordination_and_strategy",
        "responsibilities": [
            "problem_analysis",
            "resource_allocation",
            "strategy_selection",
            "bottleneck_detection",
            "algorithm_selection"
        ]
    },
    "sub_brain": {
        "role": "task_distribution_and_coordination",
        "responsibilities": [
            "task_decomposition",
            "agent_assignment",
            "progress_monitoring",
            "load_balancing"
        ]
    },
    "audit_brain": {
        "role": "result_approval",
        "responsibilities": [
            "output_validation",
            "quality_gate",
            "compliance_check",
            "risk_assessment"
        ]
    },
    "shadow_brain": {
        "role": "adversarial_analysis",
        "responsibilities": [
            "blindspot_detection",
            "counter_argument_generation",
            "risk_identification",
            "assumption_challenging"
        ]
    }
}

# Event Types
class EventCategory(str, Enum):
    PROBLEM = "problem"
    AGENT = "agent"
    SKILL = "skill"
    MEMORY = "memory"
    RESEARCH = "research"
    DREAM = "dream"
    MAGPIE = "magpie"
    AUDIT = "audit"
    SYSTEM = "system"


# Model Configuration
MODEL_DEFAULTS: Final[dict] = {
    "default_precision": "FP16",
    "max_context_length": 32768,
    "batch_size_inference": 1,
    "batch_size_training": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "max_grad_norm": 1.0
}

# Vector Search
VECTOR_SEARCH_DEFAULTS: Final[dict] = {
    "index_type": "HNSW",
    "ef_construction": 200,
    "ef_search": 100,
    "m": 16,
    "compression": "RaBitQ",
    "dimension": 768
}

# Timeouts (seconds)
TIMEOUTS: Final[dict] = {
    "model_load": 300,
    "inference": 60,
    "training_step": 300,
    "vector_search": 10,
    "external_api": 30,
    "database_query": 15,
    "pipeline_stage": 3600,
    "health_check": 5
}

# Retry Configuration
RETRY_CONFIG: Final[dict] = {
    "max_attempts": 3,
    "base_delay": 1.0,
    "max_delay": 60.0,
    "exponential_base": 2,
    "jitter": True
}

# Circuit Breaker
CIRCUIT_BREAKER: Final[dict] = {
    "failure_threshold": 5,
    "success_threshold": 2,
    "timeout": 60,
    "half_open_requests": 3
}