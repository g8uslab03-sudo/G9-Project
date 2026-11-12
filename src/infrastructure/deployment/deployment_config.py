"""
Infrastructure - Deployment configurations for GCP and Kaggle.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class DeploymentTarget(str, Enum):
    GCP_FREE_TIER = "gcp_free_tier"
    KAGGLE_2xT4 = "kaggle_2xt4"
    LOCAL_DEV = "local_dev"


@dataclass
class ResourceLimits:
    """Resource limits for deployment target."""
    cpu_cores: float
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: float
    disk_gb: float
    network_gb_per_month: float
    max_runtime_hours: Optional[float] = None


@dataclass
class ServiceConfig:
    """Configuration for a service."""
    name: str
    replicas: int
    resources: ResourceLimits
    environment: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    health_check: Optional[str] = None


# GCP Free Tier Configuration
GCP_FREE_TIER_LIMITS = ResourceLimits(
    cpu_cores=0.6,  # f1-micro burst
    memory_gb=0.6,
    gpu_count=0,
    gpu_memory_gb=0,
    disk_gb=5,
    network_gb_per_month=1,  # Egress
    max_runtime_hours=720  # Monthly
)

GCP_CLOUD_RUN_LIMITS = ResourceLimits(
    cpu_cores=2,
    memory_gb=2,
    gpu_count=0,
    gpu_memory_gb=0,
    disk_gb=10,
    network_gb_per_month=100,
    max_runtime_hours=None
)

GCP_SERVICES = {
    "api_gateway": ServiceConfig(
        name="api-gateway",
        replicas=1,
        resources=GCP_CLOUD_RUN_LIMITS,
        environment={
            "PORT": "8080",
            "ENV": "production",
            "LOG_LEVEL": "INFO"
        },
        health_check="/health"
    ),
    "orchestrator": ServiceConfig(
        name="orchestrator",
        replicas=1,
        resources=GCP_CLOUD_RUN_LIMITS,
        environment={
            "PORT": "8081",
            "ENV": "production",
            "LOG_LEVEL": "INFO"
        },
        depends_on=["api_gateway"],
        health_check="/health"
    ),
    "memory_service": ServiceConfig(
        name="memory-service",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=1,
            memory_gb=2,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=5,
            network_gb_per_month=10
        ),
        environment={
            "PORT": "8082",
            "ENV": "production"
        },
        depends_on=["orchestrator"],
        health_check="/health"
    ),
    "rag_service": ServiceConfig(
        name="rag-service",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=1,
            memory_gb=2,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=5,
            network_gb_per_month=10
        ),
        environment={
            "PORT": "8083",
            "ENV": "production"
        },
        depends_on=["memory_service"],
        health_check="/health"
    ),
    "skill_service": ServiceConfig(
        name="skill-service",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=1,
            memory_gb=1,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=2,
            network_gb_per_month=5
        ),
        environment={
            "PORT": "8084",
            "ENV": "production"
        },
        depends_on=["orchestrator"],
        health_check="/health"
    ),
    "research_service": ServiceConfig(
        name="research-service",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=2,
            memory_gb=4,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=10,
            network_gb_per_month=20
        ),
        environment={
            "PORT": "8085",
            "ENV": "production"
        },
        depends_on=["orchestrator", "memory_service", "rag_service"],
        health_check="/health"
    ),
    "dream_service": ServiceConfig(
        name="dream-service",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=0.5,
            memory_gb=1,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=2,
            network_gb_per_month=5
        ),
        environment={
            "PORT": "8086",
            "ENV": "production",
            "SCHEDULE": "0 2 * * *"  # Daily at 2 AM
        },
        depends_on=["orchestrator", "memory_service", "rag_service"],
        health_check="/health"
    ),
    "magpie_service": ServiceConfig(
        name="magpie-service",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=0.5,
            memory_gb=1,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=5,
            network_gb_per_month=5
        ),
        environment={
            "PORT": "8087",
            "ENV": "production",
            "SCHEDULE": "0 3 * * *"  # Daily at 3 AM
        },
        depends_on=["orchestrator", "memory_service"],
        health_check="/health"
    ),
    "monitoring": ServiceConfig(
        name="monitoring",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=0.25,
            memory_gb=0.5,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=2,
            network_gb_per_month=1
        ),
        environment={
            "PORT": "9090",
            "ENV": "production"
        },
        health_check="/health"
    )
}


# Kaggle 2xT4 Configuration
KAGGLE_LIMITS = ResourceLimits(
    cpu_cores=4,
    memory_gb=16,
    gpu_count=2,
    gpu_memory_gb=16,  # Per GPU
    disk_gb=20,  # Persistent
    network_gb_per_month=20,  # Weekly egress limit
    max_runtime_hours=30  # Weekly
)

KAGGLE_SERVICES = {
    "main_brain": ServiceConfig(
        name="main-brain",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=2,
            memory_gb=8,
            gpu_count=1,
            gpu_memory_gb=16,
            disk_gb=5,
            network_gb_per_month=5
        ),
        environment={
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "MODEL_PRECISION": "fp16",
            "BATCH_SIZE": "1"
        },
        health_check="/health"
    ),
    "sub_brain": ServiceConfig(
        name="sub-brain",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=1,
            memory_gb=4,
            gpu_count=1,
            gpu_memory_gb=16,
            disk_gb=5,
            network_gb_per_month=5
        ),
        environment={
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "MODEL_PRECISION": "fp16",
            "BATCH_SIZE": "2"
        },
        health_check="/health"
    ),
    "audit_shadow_brains": ServiceConfig(
        name="audit-shadow-brains",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=1,
            memory_gb=4,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=2,
            network_gb_per_month=2
        ),
        environment={
            "MODEL_PRECISION": "fp16"
        },
        health_check="/health"
    ),
    "dream_module": ServiceConfig(
        name="dream-module",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=0.5,
            memory_gb=2,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=2,
            network_gb_per_month=2
        ),
        environment={
            "SCHEDULE": "daily",
            "HYPOTHESIS_COUNT": "5"
        },
        health_check="/health"
    ),
    "magpie_module": ServiceConfig(
        name="magpie-module",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=1,
            memory_gb=4,
            gpu_count=1,  # Uses GPU for training
            gpu_memory_gb=16,
            disk_gb=10,
            network_gb_per_month=2
        ),
        environment={
            "CUDA_VISIBLE_DEVICES": "0",  # Shares with main-brain, scheduled
            "TRAINING_SCHEDULE": "weekly",
            "PRECISION": "bf16"
        },
        health_check="/health"
    ),
    "research_module": ServiceConfig(
        name="research-module",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=2,
            memory_gb=8,
            gpu_count=1,
            gpu_memory_gb=16,
            disk_gb=10,
            network_gb_per_month=10
        ),
        environment={
            "CUDA_VISIBLE_DEVICES": "1",  # Shares with sub-brain
            "SANDBOX_TIMEOUT": "3600",
            "MAX_EXPERIMENTS": "10"
        },
        health_check="/health"
    ),
    "memory_rag": ServiceConfig(
        name="memory-rag",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=1,
            memory_gb=4,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=5,
            network_gb_per_month=5
        ),
        environment={
            "VECTOR_INDEX_TYPE": "HNSW",
            "COMPRESSION": "RaBitQ",
            "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
        },
        health_check="/health"
    ),
    "skill_registry": ServiceConfig(
        name="skill-registry",
        replicas=1,
        resources=ResourceLimits(
            cpu_cores=0.5,
            memory_gb=2,
            gpu_count=0,
            gpu_memory_gb=0,
            disk_gb=2,
            network_gb_per_month=2
        ),
        environment={
            "TOTAL_SKILLS": "250",
            "CATEGORIES": "5"
        },
        health_check="/health"
    )
}


# Local Development Configuration
LOCAL_LIMITS = ResourceLimits(
    cpu_cores=8,
    memory_gb=32,
    gpu_count=1,
    gpu_memory_gb=24,
    disk_gb=100,
    network_gb_per_month=1000,
    max_runtime_hours=None
)

LOCAL_SERVICES = {
    "all_in_one": ServiceConfig(
        name="all-in-one",
        replicas=1,
        resources=LOCAL_LIMITS,
        environment={
            "ENV": "development",
            "LOG_LEVEL": "DEBUG",
            "CUDA_VISIBLE_DEVICES": "0",
            "MODEL_PRECISION": "fp16",
            "BATCH_SIZE": "4"
        },
        health_check="/health"
    )
}


def get_deployment_config(target: DeploymentTarget) -> Dict[str, Any]:
    """Get deployment configuration for target."""
    if target == DeploymentTarget.GCP_FREE_TIER:
        return {
            "target": target.value,
            "limits": GCP_FREE_TIER_LIMITS.__dict__,
            "services": {k: v.__dict__ for k, v in GCP_SERVICES.items()},
            "cloud_run": True,
            "cloud_sql": True,
            "cloud_storage": True,
            "pub_sub": True,
            "secret_manager": True
        }
    elif target == DeploymentTarget.KAGGLE_2xT4:
        return {
            "target": target.value,
            "limits": KAGGLE_LIMITS.__dict__,
            "services": {k: v.__dict__ for k, v in KAGGLE_SERVICES.items()},
            "notebook": True,
            "gpu_accelerator": "T4",
            "gpu_count": 2,
            "persistent_disk": "20GB",
            "internet": True
        }
    elif target == DeploymentTarget.LOCAL_DEV:
        return {
            "target": target.value,
            "limits": LOCAL_LIMITS.__dict__,
            "services": {k: v.__dict__ for k, v in LOCAL_SERVICES.items()},
            "docker_compose": True,
            "hot_reload": True
        }
    else:
        raise ValueError(f"Unknown deployment target: {target}")


def generate_docker_compose(target: DeploymentTarget) -> str:
    """Generate docker-compose.yml for target."""
    config = get_deployment_config(target)
    services = config["services"]
    
    compose = ["version: '3.8'", "", "services:"]
    
    for name, svc in services.items():
        compose.append(f"  {name}:")
        compose.append(f"    image: multi-agent-system/{name}:latest")
        compose.append(f"    container_name: {name}")
        
        # Resources
        resources = svc["resources"]
        compose.append("    deploy:")
        compose.append("      resources:")
        compose.append("        limits:")
        compose.append(f"          cpus: '{resources['cpu_cores']}'")
        compose.append(f"          memory: {resources['memory_gb']}G")
        
        if resources["gpu_count"] > 0:
            compose.append("        reservations:")
            compose.append("          devices:")
            compose.append("            - driver: nvidia")
            compose.append(f"              count: {resources['gpu_count']}")
            compose.append("              capabilities: [gpu]")
        
        # Environment
        if svc["environment"]:
            compose.append("    environment:")
            for key, value in svc["environment"].items():
                compose.append(f"      {key}: '{value}'")
        
        # Depends on
        if svc["depends_on"]:
            compose.append("    depends_on:")
            for dep in svc["depends_on"]:
                compose.append(f"      - {dep}")
        
        # Health check
        if svc["health_check"]:
            compose.append("    healthcheck:")
            compose.append(f"      test: ['CMD', 'curl', '-f', 'http://localhost:{svc['environment'].get('PORT', '8080')}{svc['health_check']}']")
            compose.append("      interval: 30s")
            compose.append("      timeout: 10s")
            compose.append("      retries: 3")
            compose.append("      start_period: 40s")
        
        compose.append("")
    
    # Networks
    compose.append("networks:")
    compose.append("  default:")
    compose.append("    name: multi-agent-network")
    
    # Volumes
    compose.append("volumes:")
    compose.append("  model_cache:")
    compose.append("  data:")
    compose.append("  logs:")
    
    return "\n".join(compose)


def generate_kubernetes_manifests(target: DeploymentTarget) -> Dict[str, str]:
    """Generate Kubernetes manifests for GCP Cloud Run / GKE."""
    if target != DeploymentTarget.GCP_FREE_TIER:
        return {}
    
    config = get_deployment_config(target)
    services = config["services"]
    
    manifests = {}
    
    for name, svc in services.items():
        resources = svc["resources"]
        
        # Deployment
        deployment = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
spec:
  replicas: {svc['replicas']}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
      - name: {name}
        image: gcr.io/PROJECT_ID/{name}:latest
        ports:
        - containerPort: {svc['environment'].get('PORT', '8080')}
        resources:
          requests:
            cpu: "{resources['cpu_cores']}"
            memory: {resources['memory_gb']}Gi
          limits:
            cpu: "{resources['cpu_cores']}"
            memory: {resources['memory_gb']}Gi
        env:
"""
        for key, value in svc["environment"].items():
            deployment += f"        - name: {key}\n          value: \"{value}\"\n"
        
        if svc["health_check"]:
            port = svc['environment'].get('PORT', '8080')
            deployment += f"""        livenessProbe:
          httpGet:
            path: {svc['health_check']}
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: {svc['health_check']}
            port: {port}
          initialDelaySeconds: 10
          periodSeconds: 5
"""
        deployment += """      restartPolicy: Always
"""
        
        manifests[f"{name}-deployment.yaml"] = deployment
        
        # Service
        service = f"""apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector:
    app: {name}
  ports:
  - port: 80
    targetPort: {svc['environment'].get('PORT', '8080')}
  type: ClusterIP
"""
        manifests[f"{name}-service.yaml"] = service
    
    return manifests