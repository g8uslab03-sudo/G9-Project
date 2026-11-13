"""Persistence layer: atomic memory, manifest, graph, vector index, checkpoint store, evaporation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.contracts import Artifact, Checkpoint, Experiment, FailureRecord, Hypothesis, Job, Paper


@dataclass
class AtomicMemory:
    """Atomic memory: smallest retrievable, citable unit."""
    id: str
    content: str
    content_type: str
    source: str
    fingerprint: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "content_type": self.content_type,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AtomicMemory":
        return cls(
            id=d["id"],
            content=d["content"],
            content_type=d["content_type"],
            source=d["source"],
            fingerprint=d["fingerprint"],
            created_at=datetime.fromisoformat(d["created_at"]),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ManifestEntry:
    """Manifest: atomic order, source, object relationships."""
    atomic_id: str
    order: int
    source: str
    related_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FingerprintStore:
    """SHA-256 content fingerprinting for dedup and integrity."""

    @staticmethod
    def content_fingerprint(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def file_fingerprint(path: str | Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


class AtomicStore:
    """SQLite-backed atomic memory store with manifest and Merkle relationships."""

    def __init__(self, db_path: str | Path = "data/atomic_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS atomic_memory (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS manifest (
                    atomic_id TEXT NOT NULL,
                    order_index INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    related_ids TEXT NOT NULL DEFAULT '[]',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (atomic_id, order_index)
                );
                CREATE INDEX IF NOT EXISTS idx_atomic_fingerprint ON atomic_memory(fingerprint);
                CREATE INDEX IF NOT EXISTS idx_manifest_source ON manifest(source);
                """
            )
            self._conn.commit()

    def add(self, memory: AtomicMemory) -> str:
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM atomic_memory WHERE fingerprint = ?",
                (memory.fingerprint,),
            ).fetchone()
            if existing:
                return existing[0]
            self._conn.execute(
                "INSERT INTO atomic_memory (id, content, content_type, source, fingerprint, created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    memory.content,
                    memory.content_type,
                    memory.source,
                    memory.fingerprint,
                    memory.created_at.isoformat(),
                    json.dumps(memory.metadata),
                ),
            )
            self._conn.commit()
        return memory.id

    def get(self, atomic_id: str) -> Optional[AtomicMemory]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, content, content_type, source, fingerprint, created_at, metadata FROM atomic_memory WHERE id = ?",
                (atomic_id,),
            ).fetchone()
        if not row:
            return None
        return AtomicMemory(
            id=row[0],
            content=row[1],
            content_type=row[2],
            source=row[3],
            fingerprint=row[4],
            created_at=datetime.fromisoformat(row[5]),
            metadata=json.loads(row[6]),
        )

    def add_manifest(self, entry: ManifestEntry):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO manifest (atomic_id, order_index, source, related_ids, metadata) VALUES (?, ?, ?, ?, ?)",
                (
                    entry.atomic_id,
                    entry.order,
                    entry.source,
                    json.dumps(entry.related_ids),
                    json.dumps(entry.metadata),
                ),
            )
            self._conn.commit()

    def list_manifest(self, source: Optional[str] = None) -> List[ManifestEntry]:
        with self._lock:
            if source:
                rows = self._conn.execute(
                    "SELECT atomic_id, order_index, source, related_ids, metadata FROM manifest WHERE source = ? ORDER BY order_index",
                    (source,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT atomic_id, order_index, source, related_ids, metadata FROM manifest ORDER BY order_index"
                ).fetchall()
        return [
            ManifestEntry(
                atomic_id=r[0],
                order=r[1],
                source=r[2],
                related_ids=json.loads(r[3]),
                metadata=json.loads(r[4]),
            )
            for r in rows
        ]


class MerkleGraph:
    """Graph of atomic memory relationships with Merkle root for integrity."""

    def __init__(self, store: AtomicStore):
        self.store = store
        self._adjacency: Dict[str, Set[str]] = {}
        self._load()

    def _load(self):
        for entry in self.store.list_manifest():
            self._adjacency.setdefault(entry.atomic_id, set())
            for rid in entry.related_ids:
                self._adjacency[entry.atomic_id].add(rid)
                self._adjacency.setdefault(rid, set())

    def add_edge(self, source_id: str, target_id: str):
        self._adjacency.setdefault(source_id, set()).add(target_id)
        self._adjacency.setdefault(target_id, set())

    def neighbors(self, atomic_id: str) -> Set[str]:
        return self._adjacency.get(atomic_id, set())

    def merkle_root(self) -> str:
        """Compute Merkle root over sorted atomic IDs for integrity verification."""
        ids = sorted(self._adjacency.keys())
        h = hashlib.sha256()
        for aid in ids:
            h.update(aid.encode("utf-8"))
            h.update(b"|")
        return h.hexdigest()


class TurboQuantVectorIndex:
    """Vector index with optional TurboQuant (1-bit) compression candidate backend.

    Keeps raw vectors in memory; compressed index stored to disk. Rebuildable.
    """

    def __init__(self, dim: int = 384, compress: bool = True, index_path: str | Path = "data/vector_index"):
        self.dim = dim
        self.compress = compress
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._vectors: Dict[str, List[float]] = {}
        self._ids: List[str] = []
        self._lock = threading.Lock()

    def add(self, vector_id: str, vector: List[float]):
        with self._lock:
            self._vectors[vector_id] = vector
            if vector_id not in self._ids:
                self._ids.append(vector_id)

    def search(self, query: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Cosine similarity search."""
        with self._lock:
            if not self._vectors:
                return []
            q_norm = sum(v * v for v in query) ** 0.5 or 1.0
            scored = []
            for vid, vec in self._vectors.items():
                v_norm = sum(v * v for v in vec) ** 0.5 or 1.0
                dot = sum(a * b for a, b in zip(query, vec))
                sim = dot / (q_norm * v_norm)
                scored.append((vid, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

    def save(self):
        with self._lock:
            payload = {"dim": self.dim, "vectors": self._vectors, "ids": self._ids}
            path = self.index_path / "index.json"
            with open(path, "w") as f:
                json.dump(payload, f)

    def load(self):
        path = self.index_path / "index.json"
        if path.exists():
            with open(path) as f:
                payload = json.load(f)
            self._vectors = payload.get("vectors", {})
            self._ids = payload.get("ids", [])

    def compress_turboquant(self) -> bytes:
        """1-bit sign compression of vectors (TurboQuant candidate)."""
        with self._lock:
            out = bytearray()
            for vid in self._ids:
                vec = self._vectors[vid]
                bits = 0
                for i, v in enumerate(vec):
                    if v >= 0:
                        bits |= (1 << (i % 8))
                    if i % 8 == 7:
                        out.append(bits)
                        bits = 0
                if len(vec) % 8 != 0:
                    out.append(bits)
            return bytes(out)


class CheckpointStore:
    """Checkpoint save/restore with SHA-256 and ZIP packaging for 12h disconnect resume."""

    def __init__(self, base_path: str | Path = "data/checkpoints"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save(self, checkpoint: Checkpoint, state_bundle: Dict[str, Any]) -> str:
        """Save checkpoint state as JSON + ZIP bundle, return object path."""
        with self._lock:
            step_dir = self.base_path / str(checkpoint.job_id)
            step_dir.mkdir(parents=True, exist_ok=True)
            bundle_path = step_dir / f"{checkpoint.step}.json"
            state_path = step_dir / f"{checkpoint.step}.state.zip"
            # Save state bundle
            with zipfile.ZipFile(state_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for k, v in state_bundle.items():
                    zf.writestr(f"{k}.json", json.dumps(v, default=str))
            sha = hashlib.sha256()
            with open(state_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
            checkpoint.storage_path = str(state_path)
            checkpoint.sha256 = sha.hexdigest()
            with open(bundle_path, "w") as f:
                json.dump(
                    {
                        "id": str(checkpoint.id),
                        "job_id": str(checkpoint.job_id),
                        "step": checkpoint.step,
                        "state": checkpoint.state,
                        "storage_path": checkpoint.storage_path,
                        "sha256": checkpoint.sha256,
                        "created_at": checkpoint.created_at.isoformat(),
                        "metadata": checkpoint.metadata,
                    },
                    f,
                    indent=2,
                )
            return str(bundle_path)

    def restore(self, job_id, step: str) -> Optional[Dict[str, Any]]:
        """Restore checkpoint state bundle for job/step."""
        step_dir = self.base_path / str(job_id)
        bundle_path = step_dir / f"{step}.json"
        state_path = step_dir / f"{step}.state.zip"
        if not bundle_path.exists() or not state_path.exists():
            return None
        with open(bundle_path) as f:
            meta = json.load(f)
        # verify hash
        sha = hashlib.sha256()
        with open(state_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        if sha.hexdigest() != meta.get("sha256"):
            return None
        bundle = {}
        with zipfile.ZipFile(state_path) as zf:
            for name in zf.namelist():
                bundle[name.replace(".json", "")] = json.loads(zf.read(name))
        return {"meta": meta, "bundle": bundle}

    def list_checkpoints(self, job_id) -> List[Dict[str, Any]]:
        step_dir = self.base_path / str(job_id)
        if not step_dir.exists():
            return []
        result = []
        for f in sorted(step_dir.glob("*.json")):
            with open(f) as fh:
                result.append(json.load(fh))
        return result


class WorkspaceEvaporator:
    """Evaporate (clear) working area between pipeline handoffs while preserving checkpoints."""

    def __init__(self, work_dirs: Optional[List[str | Path]] = None, preserve: Optional[List[str | Path]] = None):
        self.work_dirs = [Path(d) for d in (work_dirs or [])]
        self.preserve = [Path(p) for p in (preserve or [])]

    def evaporate(self, save_checkpoint: bool = True, checkpoint_store: Optional[CheckpointStore] = None, job_id=None, step: str = "evaporated") -> Dict[str, Any]:
        """Clear work directories, optionally save a checkpoint first."""
        cleared: List[str] = []
        checkpoint_saved = False
        checkpoint_path = None
        if save_checkpoint and checkpoint_store and job_id is not None:
            bundle = {"evaporated_at": datetime.now(timezone.utc).isoformat(), "cleared": cleared}
            checkpoint = Checkpoint(job_id=job_id, step=step, state=bundle)
            checkpoint_path = checkpoint_store.save(checkpoint, bundle)
            checkpoint_saved = True
        for d in self.work_dirs:
            if d.exists() and d.is_dir():
                for item in d.iterdir():
                    if item in self.preserve:
                        continue
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    cleared.append(str(item))
        return {
            "evaporated": True,
            "workspace_cleared": True,
            "checkpoint_saved": checkpoint_saved,
            "checkpoint_path": checkpoint_path,
            "cleared_items": cleared,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class PersistenceManager:
    """Unified persistence facade."""

    def __init__(self, base_path: str | Path = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.atomic = AtomicStore(self.base_path / "atomic_memory.db")
        self.graph = MerkleGraph(self.atomic)
        self.vector = TurboQuantVectorIndex(index_path=self.base_path / "vector_index")
        self.checkpoints = CheckpointStore(self.base_path / "checkpoints")
        self.evaporator = WorkspaceEvaporator(
            work_dirs=[self.base_path / "work"],
            preserve=[self.base_path / "checkpoints", self.base_path / "vector_index"],
        )

    def store_atomic(self, content: str, content_type: str, source: str, metadata: Optional[Dict] = None) -> AtomicMemory:
        fp = FingerprintStore.content_fingerprint(content)
        mem = AtomicMemory(
            id=fp[:16],
            content=content,
            content_type=content_type,
            source=source,
            fingerprint=fp,
            metadata=metadata or {},
        )
        self.atomic.add(mem)
        return mem

    def save_checkpoint(self, checkpoint: Checkpoint, state_bundle: Dict[str, Any]) -> str:
        return self.checkpoints.save(checkpoint, state_bundle)

    def restore_checkpoint(self, job_id, step: str) -> Optional[Dict[str, Any]]:
        return self.checkpoints.restore(job_id, step)

    def evaporate_workspace(self, job_id=None, step: str = "evaporated", checkpoint: Optional[Checkpoint] = None, state_bundle: Optional[Dict] = None) -> Dict[str, Any]:
        if checkpoint and state_bundle is not None and job_id is not None:
            self.save_checkpoint(checkpoint, state_bundle)
        return self.evaporator.evaporate(
            save_checkpoint=False,
            checkpoint_store=None,
            job_id=job_id,
            step=step,
        )
