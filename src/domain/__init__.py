"""
Domain Layer - Core business logic.
"""
from .entities import *
from .aggregates import *
from .services import *
from .repositories import *

__all__ = [
    # Entities
    "ProblemId", "AgentId", "SkillId", "MemoryId", "ResearchId", "GeneKey",
    "SkillSpec", "AgentCapability", "ProblemContext", "SubProblem",
    "VerificationResult", "DebateArgument", "IntegratedAnswer",
    "MemoryAtom", "MemoryManifest", "MemoryGraphEdge",
    "ResearchPhase", "ResearchPlan", "Hypothesis", "ExperimentResult",
    "ResearchPaper", "Artifact", "DreamHypothesis", "FailureRecord", "TrainingSample",
    "ProblemStatus", "AgentRole", "SkillCategory", "VerificationStatus",
    # Events
    "ProblemReceived", "ProblemUnderstood", "ProblemDecomposed", "ExpertsDispatched",
    "ReasoningCompleted", "VerificationCompleted", "DebateCompleted", "AnswerIntegrated",
    "SkillAcquired", "SkillPromoted", "MemoryStored",
    "ResearchStarted", "ResearchPhaseCompleted", "HypothesisGenerated",
    "ExperimentCompleted", "PaperGenerated", "ArtifactProduced",
    "DreamHypothesisGenerated", "DreamHypothesisAudited",
    "FailureRecorded", "TrainingSampleCreated", "ModelFineTuned",
    # Aggregates
    "ProblemAggregate", "AgentAggregate", "SkillAggregate", "ResearchAggregate",
    # Services
    "ProblemDecompositionService", "AgentAssignmentService", "VerificationService",
    "DebateService", "IntegrationService", "SkillAcquisitionService",
    "MemoryManagementService", "ResearchPlanningService", "DreamGenerationService",
    "MagpieDistillationService",
    # Repositories
    "ProblemRepository", "AgentRepository", "SkillRepository", "ResearchRepository",
    "MemoryRepository", "EventStoreRepository", "SnapshotRepository",
    "VectorIndexRepository", "ModelRepository", "DatasetRepository",
    "CheckpointRepository", "SkillGeneRepository",
]