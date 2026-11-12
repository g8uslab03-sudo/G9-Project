"""
Ports - Interfaces for external adapters (Hexagonal Architecture).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from uuid import UUID
from ..shared.kernel.result import Result
from ..domain.entities import (
    ProblemId, AgentId, SkillId, MemoryId, ResearchId,
    GeneKey, SkillSpec, AgentCapability, ProblemContext, SubProblem,
    VerificationResult, DebateArgument, IntegratedAnswer,
    MemoryAtom, MemoryManifest, MemoryGraphEdge,
    ResearchPlan, Hypothesis, ExperimentResult, ResearchPaper,
    Artifact, DreamHypothesis, FailureRecord, TrainingSample
)


# Primary Ports (Driving Adapters -> Application)

class ProblemSolvingPort(ABC):
    """Primary port for problem solving operations."""
    
    @abstractmethod
    async def solve_problem(
        self,
        user_query: str,
        domain: str = "general",
        constraints: Optional[Dict[str, Any]] = None,
        available_resources: Optional[Dict[str, Any]] = None,
        priority: int = 1
    ) -> Result[IntegratedAnswer]:
        pass
    
    @abstractmethod
    async def resume_pipeline(self, pipeline_id: str) -> Result[IntegratedAnswer]:
        pass
    
    @abstractmethod
    async def get_pipeline_status(self, pipeline_id: str) -> Result[Dict[str, Any]]:
        pass


class SkillManagementPort(ABC):
    """Primary port for skill management."""
    
    @abstractmethod
    async def initialize_genes(self, genes: List[GeneKey]) -> Result[None]:
        pass
    
    @abstractmethod
    async def ensure_skill(self, gene_key: GeneKey) -> Result[SkillSpec]:
        pass
    
    @abstractmethod
    async def promote_skill(self, skill_id: SkillId, benchmarks: Dict[str, float]) -> Result[None]:
        pass
    
    @abstractmethod
    async def get_skill_stats(self) -> Result[Dict[str, Any]]:
        pass


class AgentManagementPort(ABC):
    """Primary port for agent management."""
    
    @abstractmethod
    async def register_agent(self, capability: AgentCapability) -> Result[AgentId]:
        pass
    
    @abstractmethod
    async def get_agent_load(self) -> Result[Dict[str, Any]]:
        pass


class ResearchManagementPort(ABC):
    """Primary port for research management."""
    
    @abstractmethod
    async def start_research(
        self,
        problem_id: ProblemId,
        problem_context: ProblemContext
    ) -> Result[ResearchId]:
        pass
    
    @abstractmethod
    async def get_research_progress(self, research_id: ResearchId) -> Result[Dict[str, Any]]:
        pass


class MemoryManagementPort(ABC):
    """Primary port for memory management."""
    
    @abstractmethod
    async def store_atom(self, atom: MemoryAtom) -> Result[MemoryId]:
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> Result[List[MemoryAtom]]:
        pass
    
    @abstractmethod
    async def get_by_manifest(self, object_type: str, object_id: str) -> Result[List[MemoryAtom]]:
        pass


class DreamManagementPort(ABC):
    """Primary port for dream module."""
    
    @abstractmethod
    async def generate_hypotheses(self, count: int = 5) -> Result[List[DreamHypothesis]]:
        pass
    
    @abstractmethod
    async def audit_hypothesis(self, hypothesis_id: UUID, approved: bool, notes: str) -> Result[None]:
        pass


class MagpieManagementPort(ABC):
    """Primary port for Magpie training."""
    
    @abstractmethod
    async def record_failure(self, failure: FailureRecord) -> Result[None]:
        pass
    
    @abstractmethod
    async def distill_failures(self, module: Optional[str] = None) -> Result[List[TrainingSample]]:
        pass
    
    @abstractmethod
    async def fine_tune_agent(self, agent_id: AgentId, sample_ids: List[UUID]) -> Result[None]:
        pass


# Secondary Ports (Application -> Driven Adapters)

class LLMProviderPort(ABC):
    """Port for LLM providers."""
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        model: Optional[str] = None
    ) -> Result[str]:
        pass
    
    @abstractmethod
    async def complete_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Result[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str], model: Optional[str] = None) -> Result[List[List[float]]]:
        pass


class VectorSearchPort(ABC):
    """Port for vector search."""
    
    @abstractmethod
    async def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> Result[None]:
        pass
    
    @abstractmethod
    async def search(
        self,
        vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> Result[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> Result[None]:
        pass


class OpenSourceSearchPort(ABC):
    """Port for open source search and download."""
    
    @abstractmethod
    async def search_papers(
        self,
        query: str,
        sources: List[str],
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def search_code(
        self,
        query: str,
        languages: List[str],
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def search_models(
        self,
        query: str,
        tasks: List[str],
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def download_resource(self, resource_id: str, resource_type: str) -> Result[bytes]:
        pass
    
    @abstractmethod
    async def get_trending(self, source: str, limit: int = 10) -> Result[List[Dict[str, Any]]]:
        pass


class SandboxPort(ABC):
    """Port for code execution sandbox."""
    
    @abstractmethod
    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int = 300,
        resources: Optional[Dict[str, Any]] = None
    ) -> Result[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def create_session(self, config: Dict[str, Any]) -> Result[str]:
        pass
    
    @abstractmethod
    async def close_session(self, session_id: str) -> Result[None]:
        pass


class ModelTrainingPort(ABC):
    """Port for model training."""
    
    @abstractmethod
    async def fine_tune(
        self,
        model_id: str,
        training_data: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> Result[str]:
        pass
    
    @abstractmethod
    async def evaluate(
        self,
        model_id: str,
        eval_data: List[Dict[str, Any]]
    ) -> Result[Dict[str, float]]:
        pass


class CompressionPort(ABC):
    """Port for compression algorithms."""
    
    @abstractmethod
    async def compress_vectors(
        self,
        vectors: List[List[float]],
        algorithm: str = "RaBitQ"
    ) -> Result[List[bytes]]:
        pass
    
    @abstractmethod
    async def decompress_vectors(
        self,
        compressed: List[bytes],
        algorithm: str = "RaBitQ",
        original_shape: Optional[tuple] = None
    ) -> Result[List[List[float]]]:
        pass
    
    @abstractmethod
    async def quantize_model(
        self,
        model_path: str,
        quantization: str = "FP16"
    ) -> Result[str]:
        pass


class MonitoringPort(ABC):
    """Port for monitoring and observability."""
    
    @abstractmethod
    async def record_metric(self, name: str, value: float, labels: Dict[str, str]) -> None:
        pass
    
    @abstractmethod
    async def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    async def health_check(self) -> Result[Dict[str, Any]]:
        pass


class StoragePort(ABC):
    """Port for persistent storage."""
    
    @abstractmethod
    async def save(self, key: str, data: bytes, metadata: Dict[str, Any]) -> Result[str]:
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Result[bytes]:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> Result[None]:
        pass
    
    @abstractmethod
    async def list(self, prefix: str) -> Result[List[str]]:
        pass


class MessageQueuePort(ABC):
    """Port for message queue."""
    
    @abstractmethod
    async def publish(self, topic: str, message: Dict[str, Any]) -> Result[None]:
        pass
    
    @abstractmethod
    async def subscribe(self, topic: str, handler: callable) -> Result[str]:
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> Result[None]:
        pass


class CachePort(ABC):
    """Port for caching."""
    
    @abstractmethod
    async def get(self, key: str) -> Result[Optional[Any]]:
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> Result[None]:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> Result[None]:
        pass
    
    @abstractmethod
    async def clear(self) -> Result[None]:
        pass