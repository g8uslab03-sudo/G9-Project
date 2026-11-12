"""
Application Layer - Use cases and orchestration.
"""
from .commands import *
from .queries import *
from .handlers import *
from .query_handlers import *
from .services import *
from .dtos import *
from .ports import *

__all__ = [
    # Commands
    "ReceiveProblemCommand", "UnderstandProblemCommand", "DecomposeProblemCommand",
    "DispatchExpertsCommand", "CompleteReasoningCommand", "VerifyResultCommand",
    "ConductDebateCommand", "IntegrateAnswerCommand",
    "RegisterAgentCommand", "AssignTaskCommand", "CompleteTaskCommand",
    "UpdateAgentPerformanceCommand", "AddAgentSkillCommand",
    "RegisterSkillCommand", "AcquireSkillCommand", "PromoteSkillCommand",
    "DeprecateSkillCommand", "RecordSkillUsageCommand",
    "StoreMemoryAtomCommand", "StoreMemoryManifestCommand", "AddMemoryRelationshipCommand",
    "StartResearchCommand", "CompleteResearchPhaseCommand", "AddHypothesisCommand",
    "AddExperimentResultCommand", "SetResearchPaperCommand", "AddArtifactCommand",
    "GenerateDreamHypothesesCommand", "AuditDreamHypothesisCommand",
    "RecordFailureCommand", "DistillFailuresCommand", "FineTuneModelCommand",
    "ExecutePipelineCommand", "SaveCheckpointCommand", "ResumePipelineCommand",
    # Queries
    "GetProblemQuery", "GetProblemsByStatusQuery", "GetRecentProblemsQuery",
    "GetProblemStatusQuery", "GetProblemAnswerQuery",
    "GetAgentQuery", "GetAvailableAgentsQuery", "GetAgentsByRoleQuery",
    "GetAllAgentsQuery", "GetAgentCapabilitiesQuery",
    "GetSkillQuery", "GetSkillByGeneKeyQuery", "GetSkillsByCategoryQuery",
    "GetAvailableSkillsQuery", "GetAllSkillsQuery", "SearchSkillsQuery",
    "GetSkillGeneKeysQuery",
    "GetResearchQuery", "GetResearchByProblemQuery", "GetInProgressResearchQuery",
    "GetCompletedResearchQuery",
    "SearchMemoryQuery", "GetMemoryByManifestQuery", "GetMemoryAtomQuery",
    "GetPipelineStatusQuery", "GetPipelineCheckpointsQuery", "GetLatestCheckpointQuery",
    "GetSystemHealthQuery", "GetSystemMetricsQuery", "GetResourceUsageQuery",
    # Handlers
    "ProblemCommandHandlers", "AgentCommandHandlers", "SkillCommandHandlers",
    "ResearchCommandHandlers", "register_command_handlers",
    "ProblemQueryHandlers", "AgentQueryHandlers", "SkillQueryHandlers",
    "ResearchQueryHandlers", "MemoryQueryHandlers", "PipelineQueryHandlers",
    "SystemQueryHandlers", "register_query_handlers",
    # Services
    "PipelineOrchestrator", "SkillManagementService", "AgentManagementService",
    "ResearchManagementService", "PipelineStage", "PipelineState",
    # DTOs
    "ProblemDTO", "SubProblemDTO", "AgentDTO", "SkillDTO", "ResearchDTO",
    "HypothesisDTO", "ExperimentResultDTO", "ResearchPaperDTO", "ArtifactDTO",
    "DreamHypothesisDTO", "PipelineStatusDTO", "CheckpointDTO",
    "SystemHealthDTO", "SystemMetricsDTO", "ResourceUsageDTO", "SkillGeneDTO",
    "problem_to_dto", "sub_problem_to_dto", "agent_to_dto", "skill_to_dto",
    "research_to_dto", "hypothesis_to_dto", "experiment_to_dto",
    "paper_to_dto", "artifact_to_dto", "dream_hypothesis_to_dto",
    # Ports
    "ProblemSolvingPort", "SkillManagementPort", "AgentManagementPort",
    "ResearchManagementPort", "MemoryManagementPort", "DreamManagementPort",
    "MagpieManagementPort", "LLMProviderPort", "VectorSearchPort",
    "OpenSourceSearchPort", "SandboxPort", "ModelTrainingPort",
    "CompressionPort", "MonitoringPort", "StoragePort", "MessageQueuePort", "CachePort",
]