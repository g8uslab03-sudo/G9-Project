"""
Repository Interfaces - Ports for persistence (Hexagonal Architecture).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, AsyncIterator
from uuid import UUID
from ..shared.kernel.base import AggregateRoot
from ..shared.kernel.result import Result
from .entities import (
    ProblemId, AgentId, SkillId, MemoryId, ResearchId,
    ProblemStatus, AgentRole, SkillCategory, GeneKey
)
from .aggregates import ProblemAggregate, AgentAggregate, SkillAggregate, ResearchAggregate


class ProblemRepository(ABC):
    """Repository for Problem aggregates."""
    
    @abstractmethod
    async def save(self, aggregate: ProblemAggregate) -> Result[None]:
        pass
    
    @abstractmethod
    async def get(self, problem_id: ProblemId) -> Result[ProblemAggregate]:
        pass
    
    @abstractmethod
    async def get_by_status(self, status: ProblemStatus, limit: int = 100) -> Result[List[ProblemAggregate]]:
        pass
    
    @abstractmethod
    async def get_recent(self, limit: int = 100) -> Result[List[ProblemAggregate]]:
        pass


class AgentRepository(ABC):
    """Repository for Agent aggregates."""
    
    @abstractmethod
    async def save(self, aggregate: AgentAggregate) -> Result[None]:
        pass
    
    @abstractmethod
    async def get(self, agent_id: AgentId) -> Result[AgentAggregate]:
        pass
    
    @abstractmethod
    async def get_available(self, role: Optional[AgentRole] = None) -> Result[List[AgentAggregate]]:
        pass
    
    @abstractmethod
    async def get_by_role(self, role: AgentRole) -> Result[List[AgentAggregate]]:
        pass
    
    @abstractmethod
    async def get_all(self) -> Result[List[AgentAggregate]]:
        pass


class SkillRepository(ABC):
    """Repository for Skill aggregates."""
    
    @abstractmethod
    async def save(self, aggregate: SkillAggregate) -> Result[None]:
        pass
    
    @abstractmethod
    async def get(self, skill_id: SkillId) -> Result[SkillAggregate]:
        pass
    
    @abstractmethod
    async def get_by_gene_key(self, gene_key: GeneKey) -> Result[Optional[SkillAggregate]]:
        pass
    
    @abstractmethod
    async def get_by_category(self, category: SkillCategory) -> Result[List[SkillAggregate]]:
        pass
    
    @abstractmethod
    async def get_available(self) -> Result[List[SkillAggregate]]:
        pass
    
    @abstractmethod
    async def get_all(self) -> Result[List[SkillAggregate]]:
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> Result[List[SkillAggregate]]:
        pass


class ResearchRepository(ABC):
    """Repository for Research aggregates."""
    
    @abstractmethod
    async def save(self, aggregate: ResearchAggregate) -> Result[None]:
        pass
    
    @abstractmethod
    async def get(self, research_id: ResearchId) -> Result[ResearchAggregate]:
        pass
    
    @abstractmethod
    async def get_by_problem(self, problem_id: ProblemId) -> Result[List[ResearchAggregate]]:
        pass
    
    @abstractmethod
    async def get_in_progress(self) -> Result[List[ResearchAggregate]]:
        pass
    
    @abstractmethod
    async def get_completed(self, limit: int = 100) -> Result[List[ResearchAggregate]]:
        pass


class MemoryRepository(ABC):
    """Repository for Memory system."""
    
    @abstractmethod
    async def store_atom(self, atom: Any) -> Result[MemoryId]:
        pass
    
    @abstractmethod
    async def get_atom(self, atom_id: MemoryId) -> Result[Optional[Any]]:
        pass
    
    @abstractmethod
    async def store_manifest(self, manifest: Any) -> Result[UUID]:
        pass
    
    @abstractmethod
    async def get_manifest(self, manifest_id: UUID) -> Result[Optional[Any]]:
        pass
    
    @abstractmethod
    async def add_edge(self, edge: Any) -> Result[None]:
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int, filters: Optional[Dict[str, Any]]) -> Result[List[Any]]:
        pass
    
    @abstractmethod
    async def get_by_manifest(self, object_type: str, object_id: str) -> Result[List[Any]]:
        pass


class EventStoreRepository(ABC):
    """Repository for Event Store (Event Sourcing)."""
    
    @abstractmethod
    async def append(self, aggregate_id: UUID, events: List[Any], expected_version: int) -> Result[None]:
        pass
    
    @abstractmethod
    async def get_events(self, aggregate_id: UUID, from_version: int = 0) -> Result[List[Any]]:
        pass
    
    @abstractmethod
    async def get_all_events(self, from_position: int = 0, batch_size: int = 100) -> AsyncIterator[Any]:
        pass


class SnapshotRepository(ABC):
    """Repository for Aggregate Snapshots."""
    
    @abstractmethod
    async def save(self, aggregate_type: str, aggregate_id: UUID, version: int, state: Dict[str, Any]) -> Result[None]:
        pass
    
    @abstractmethod
    async def get(self, aggregate_type: str, aggregate_id: UUID) -> Result[Optional[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def delete(self, aggregate_type: str, aggregate_id: UUID) -> Result[None]:
        pass


class VectorIndexRepository(ABC):
    """Repository for Vector Index (RAG)."""
    
    @abstractmethod
    async def add(self, ids: List[str], vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> Result[None]:
        pass
    
    @abstractmethod
    async def search(self, vector: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None) -> Result[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def delete(self, ids: List[str]) -> Result[None]:
        pass
    
    @abstractmethod
    async def rebuild(self) -> Result[None]:
        pass
    
    @abstractmethod
    async def get_stats(self) -> Result[Dict[str, Any]]:
        pass


class ModelRepository(ABC):
    """Repository for ML Models."""
    
    @abstractmethod
    async def load(self, model_id: str, config: Dict[str, Any]) -> Result[Any]:
        pass
    
    @abstractmethod
    async def save(self, model_id: str, model: Any, config: Dict[str, Any]) -> Result[None]:
        pass
    
    @abstractmethod
    async def list_models(self) -> Result[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def delete(self, model_id: str) -> Result[None]:
        pass


class DatasetRepository(ABC):
    """Repository for Datasets."""
    
    @abstractmethod
    async def store(self, dataset_id: str, data: Any, metadata: Dict[str, Any]) -> Result[None]:
        pass
    
    @abstractmethod
    async def load(self, dataset_id: str) -> Result[Any]:
        pass
    
    @abstractmethod
    async def list_datasets(self) -> Result[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def delete(self, dataset_id: str) -> Result[None]:
        pass


class CheckpointRepository(ABC):
    """Repository for Pipeline Checkpoints (Resume capability)."""
    
    @abstractmethod
    async def save_checkpoint(
        self,
        pipeline_id: str,
        stage: str,
        state: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Result[None]:
        pass
    
    @abstractmethod
    async def get_latest_checkpoint(self, pipeline_id: str) -> Result[Optional[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def get_checkpoint(self, pipeline_id: str, stage: str) -> Result[Optional[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def list_checkpoints(self, pipeline_id: str) -> Result[List[Dict[str, Any]]]:
        pass


class SkillGeneRepository(ABC):
    """Repository for Skill Gene Keys (250 elite skills)."""
    
    @abstractmethod
    async def initialize_genes(self, genes: List[GeneKey]) -> Result[None]:
        pass
    
    @abstractmethod
    async def get_gene(self, gene_key: GeneKey) -> Result[Optional[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def get_all_genes(self) -> Result[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def get_genes_by_category(self, category: SkillCategory) -> Result[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    async def update_gene_status(self, gene_key: GeneKey, status: str, metadata: Dict[str, Any]) -> Result[None]:
        pass