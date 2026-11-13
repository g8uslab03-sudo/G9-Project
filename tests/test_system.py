"""Unit tests for G9 autonomous system."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from shared.contracts import (  # noqa: E402
    EventType,
    Job,
    JobStatus,
    SkillCategory,
)
from shared.persistence import (  # noqa: E402
    AtomicMemory,
    PersistenceManager,
    TurboQuantVectorIndex,
    WorkspaceEvaporator,
    FingerprintStore,
)
from shared.skill_genome import (  # noqa: E402
    SKILL_CATALOG,
    get_skill,
    list_skills_by_category,
    search_skills,
)
from shared.application.system import (  # noqa: E402
    AutonomousSystem,
    AuditBrain,
    BrainDecision,
    MainBrain,
    ShadowBrain,
)


def test_fingerprint_dedup():
    fp1 = FingerprintStore.content_fingerprint("hello world")
    fp2 = FingerprintStore.content_fingerprint("hello world")
    fp3 = FingerprintStore.content_fingerprint("different")
    assert fp1 == fp2
    assert fp1 != fp3


def test_atomic_store_dedup(tmp_path):
    db = tmp_path / "test.db"
    store = PersistenceManager(base_path=tmp_path)
    m1 = AtomicMemory(id="a", content="x", content_type="t", source="s", fingerprint="f1")
    m2 = AtomicMemory(id="b", content="x", content_type="t", source="s", fingerprint="f1")
    id1 = store.atomic.add(m1)
    id2 = store.atomic.add(m2)
    assert id1 == id2  # dedup by fingerprint


def test_turboquant_search():
    vi = TurboQuantVectorIndex(dim=4, compress=False)
    vi.add("v1", [1.0, 0.0, 0.0, 0.0])
    vi.add("v2", [0.0, 1.0, 0.0, 0.0])
    res = vi.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert res[0][0] == "v1"


def test_skill_catalog_count():
    assert len(SKILL_CATALOG) == 250


def test_skill_by_category():
    ds = list_skills_by_category(SkillCategory.DATA_SENSING)
    assert len(ds) == 50


def test_search_skills():
    res = search_skills("web scraper")
    assert len(res) >= 1


def test_get_skill_not_found():
    with pytest.raises(KeyError):
        get_skill("NONEXISTENT_SKILL")


def test_main_brain_decision():
    sys_persistence = PersistenceManager(base_path=os.path.join(os.path.dirname(__file__), "..", "data"))
    brain = MainBrain(sys_persistence)
    job = Job(title="t", description="web scraping task", status=JobStatus.PENDING)
    decision = brain.handle_job(job)
    assert isinstance(decision, BrainDecision)
    assert len(decision.skills) > 0


def test_audit_and_shadow():
    persistence = PersistenceManager(base_path=os.path.join(os.path.dirname(__file__), "..", "data"))
    audit = AuditBrain(persistence)
    shadow = ShadowBrain(persistence)
    job = Job(title="t", description="d", status=JobStatus.COMPLETED)
    res = audit.approve(job, {"output": "ok"})
    assert res.approved is True
    challenge = shadow.challenge(BrainDecision(job_id="1", action="x", skills=[]), {"output": "ok"})
    assert len(challenge) > 0


def test_evaporator(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "temp.txt").write_text("temp")
    evap = WorkspaceEvaporator(work_dirs=[work], preserve=[])
    result = evap.evaporate(save_checkpoint=False, checkpoint_store=None, job_id="j1", step="s")
    assert result["evaporated"] is True
    assert not (work / "temp.txt").exists()


def test_autonomous_system_health():
    system = AutonomousSystem()
    health = system.health()
    assert health.status == "healthy"


def test_job_status_enum():
    assert JobStatus.PENDING == "pending"
    assert JobStatus.COMPLETED == "completed"


def test_event_type_values():
    assert EventType.JOB_CREATED == "job.created"
    assert EventType.CHECKPOINT_SAVED == "checkpoint.saved"


if __name__ == "__main__":
    pytest.main([__file__])
