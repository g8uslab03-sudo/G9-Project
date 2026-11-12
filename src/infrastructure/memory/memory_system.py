"""
Infrastructure - Memory System with Compression.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from datetime import datetime
import asyncio
import json
import pickle
import gzip
import lz4.frame
import zstandard as zstd
import numpy as np
from collections import defaultdict

from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import MemoryError, CompressionError, ResourceExhaustedError
from ..shared.observability import get_logger, get_metrics, traced, metered
from ..domain.entities import (
    MemoryId, MemoryAtom, MemoryManifest, MemoryGraphEdge,
    ProblemId, AgentId, SkillId, ResearchId
)
from ..application.ports import VectorSearchPort, CompressionPort, StoragePort


logger = get_logger("infrastructure.memory")
metrics = get_metrics()


@dataclass
class MemoryConfig:
    """Memory system configuration."""
    # Storage tiers
    hot_tier_max_gb: float = 2.0      # GPU VRAM / fast RAM
    warm_tier_max_gb: float = 8.0     # CPU RAM
    cold_tier_max_gb: float = 50.0    # Disk
    
    # Compression
    vector_compression: str = "RaBitQ"
    vector_compression_ratio: float = 0.25
    object_compression: str = "zstd"
    object_compression_level: int = 3
    
    # Vector index
    index_type: str = "HNSW"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100
    
    # Checkpointing
    checkpoint_interval_seconds: int = 300
    max_checkpoints: int = 10
    
    # Prefetching
    prefetch_batches: int = 2


class MemoryTier(str, Enum):
    HOT = "hot"      # GPU VRAM / fastest access
    WARM = "warm"    # CPU RAM
    COLD = "cold"    # Disk storage


@dataclass
class MemoryObject:
    """Object stored in memory system."""
    object_id: str
    object_type: str  # "atom", "manifest", "graph_edge", "vector_index"
    data: bytes
    metadata: Dict[str, Any]
    tier: MemoryTier = MemoryTier.WARM
    size_bytes: int = 0
    compressed: bool = False
    compression_algorithm: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    
    def __post_init__(self):
        self.size_bytes = len(self.data)


class VectorIndex:
    """In-memory vector index with compression support."""
    
    def __init__(self, config: MemoryConfig, compression_port: CompressionPort):
        self._config = config
        self._compression = compression_port
        self._vectors: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._compressed_vectors: Dict[str, bytes] = {}
        self._dimension: Optional[int] = None
    
    async def add(self, ids: List[str], vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> Result[None]:
        try:
            if not vectors:
                return success(None)
            
            # Validate dimensions
            if self._dimension is None:
                self._dimension = len(vectors[0])
            elif any(len(v) != self._dimension for v in vectors):
                return failure(MemoryError("Vector dimension mismatch"))
            
            # Store vectors
            for i, (id_, vector, meta) in enumerate(zip(ids, vectors, metadata)):
                arr = np.array(vector, dtype=np.float32)
                self._vectors[id_] = arr
                self._metadata[id_] = meta
                
                # Compress if enabled
                if self._config.vector_compression != "none":
                    compressed = await self._compression.compress_vectors([vector], self._config.vector_compression)
                    if compressed.is_success():
                        self._compressed_vectors[id_] = compressed.unwrap()[0]
            
            metrics.gauge("vector_index_size", len(self._vectors))
            return success(None)
            
        except Exception as e:
            return failure(MemoryError(f"Failed to add vectors: {e}"))
    
    async def search(self, vector: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None) -> Result[List[Dict[str, Any]]]:
        try:
            if not self._vectors:
                return success([])
            
            query_vector = np.array(vector, dtype=np.float32)
            
            # Use compressed vectors if available
            if self._compressed_vectors:
                # Decompress for search (in practice, would use compressed search)
                vectors_to_search = {}
                for id_, compressed in self._compressed_vectors.items():
                    decomp = await self._compression.decompress_vectors([compressed], self._config.vector_compression, (self._dimension,))
                    if decomp.is_success():
                        vectors_to_search[id_] = decomp.unwrap()[0]
            else:
                vectors_to_search = self._vectors
            
            # Compute similarities (cosine)
            similarities = []
            for id_, vec in vectors_to_search.items():
                # Apply filters
                if filters:
                    meta = self._metadata.get(id_, {})
                    match = all(meta.get(k) == v for k, v in filters.items())
                    if not match:
                        continue
                
                # Cosine similarity
                dot = np.dot(query_vector, vec)
                norm_q = np.linalg.norm(query_vector)
                norm_v = np.linalg.norm(vec)
                if norm_q > 0 and norm_v > 0:
                    sim = dot / (norm_q * norm_v)
                    similarities.append((id_, sim, self._metadata.get(id_, {})))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = [
                {"id": id_, "score": float(score), "metadata": meta}
                for id_, score, meta in similarities[:top_k]
            ]
            
            return success(results)
            
        except Exception as e:
            return failure(MemoryError(f"Vector search failed: {e}"))
    
    async def delete(self, ids: List[str]) -> Result[None]:
        for id_ in ids:
            self._vectors.pop(id_, None)
            self._metadata.pop(id_, None)
            self._compressed_vectors.pop(id_, None)
        return success(None)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_vectors": len(self._vectors),
            "dimension": self._dimension,
            "compressed_count": len(self._compressed_vectors),
            "memory_estimate_mb": len(self._vectors) * (self._dimension or 0) * 4 / (1024 * 1024)
        }


class TieredStorage:
    """Tiered storage manager (Hot/Warm/Cold)."""
    
    def __init__(self, config: MemoryConfig, storage_port: StoragePort):
        self._config = config
        self._storage = storage_port
        
        # In-memory tiers
        self._hot: Dict[str, MemoryObject] = {}      # GPU VRAM simulation
        self._warm: Dict[str, MemoryObject] = {}     # CPU RAM
        self._cold_index: Dict[str, Dict[str, Any]] = {}  # Disk index
        
        # Current usage
        self._hot_usage = 0
        self._warm_usage = 0
        self._cold_usage = 0
        
        self._lock = asyncio.Lock()
    
    async def store(self, obj: MemoryObject) -> Result[str]:
        async with self._lock:
            # Determine initial tier based on size and type
            tier = self._determine_tier(obj)
            
            # Compress if needed
            if obj.size_bytes > 1024 * 1024:  # > 1MB
                obj = await self._compress_object(obj)
            
            # Store in appropriate tier
            if tier == MemoryTier.HOT:
                return await self._store_hot(obj)
            elif tier == MemoryTier.WARM:
                return await self._store_warm(obj)
            else:
                return await self._store_cold(obj)
    
    def _determine_tier(self, obj: MemoryObject) -> MemoryTier:
        """Determine storage tier for object."""
        # Vector indexes and hot atoms go to HOT
        if obj.object_type in ["vector_index", "atom"] and obj.size_bytes < 10 * 1024 * 1024:
            return MemoryTier.HOT
        # Medium objects to WARM
        elif obj.size_bytes < 100 * 1024 * 1024:
            return MemoryTier.WARM
        # Large objects to COLD
        else:
            return MemoryTier.COLD
    
    async def _compress_object(self, obj: MemoryObject) -> MemoryObject:
        """Compress object data."""
        try:
            if self._config.object_compression == "zstd":
                cctx = zstd.ZstdCompressor(level=self._config.object_compression_level)
                compressed = cctx.compress(obj.data)
            elif self._config.object_compression == "lz4":
                compressed = lz4.frame.compress(obj.data)
            elif self._config.object_compression == "gzip":
                compressed = gzip.compress(obj.data)
            else:
                return obj
            
            obj.data = compressed
            obj.compressed = True
            obj.compression_algorithm = self._config.object_compression
            obj.size_bytes = len(compressed)
            
            return obj
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return obj
    
    async def _decompress_object(self, obj: MemoryObject) -> MemoryObject:
        """Decompress object data."""
        if not obj.compressed:
            return obj
        
        try:
            if obj.compression_algorithm == "zstd":
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.decompress(obj.data)
            elif obj.compression_algorithm == "lz4":
                decompressed = lz4.frame.decompress(obj.data)
            elif obj.compression_algorithm == "gzip":
                decompressed = gzip.decompress(obj.data)
            else:
                return obj
            
            obj.data = decompressed
            obj.compressed = False
            obj.size_bytes = len(decompressed)
            
            return obj
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return obj
    
    async def _store_hot(self, obj: MemoryObject) -> Result[str]:
        if self._hot_usage + obj.size_bytes > self._config.hot_tier_max_gb * 1024 * 1024 * 1024:
            # Evict least recently used
            await self._evict_from_hot(obj.size_bytes)
        
        self._hot[obj.object_id] = obj
        self._hot_usage += obj.size_bytes
        return success(obj.object_id)
    
    async def _store_warm(self, obj: MemoryObject) -> Result[str]:
        if self._warm_usage + obj.size_bytes > self._config.warm_tier_max_gb * 1024 * 1024 * 1024:
            await self._evict_from_warm(obj.size_bytes)
        
        self._warm[obj.object_id] = obj
        self._warm_usage += obj.size_bytes
        return success(obj.object_id)
    
    async def _store_cold(self, obj: MemoryObject) -> Result[str]:
        # Store to persistent storage
        key = f"cold/{obj.object_type}/{obj.object_id}"
        result = await self._storage.save(key, obj.data, obj.metadata)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        # Update index
        self._cold_index[obj.object_id] = {
            "key": key,
            "size": obj.size_bytes,
            "metadata": obj.metadata,
            "tier": MemoryTier.COLD
        }
        self._cold_usage += obj.size_bytes
        
        return success(obj.object_id)
    
    async def _evict_from_hot(self, needed_bytes: int) -> None:
        """Evict LRU objects from hot tier to warm."""
        sorted_objects = sorted(
            self._hot.values(),
            key=lambda o: o.last_accessed
        )
        
        freed = 0
        for obj in sorted_objects:
            if freed >= needed_bytes:
                break
            
            # Move to warm
            await self._store_warm(obj)
            del self._hot[obj.object_id]
            self._hot_usage -= obj.size_bytes
            freed += obj.size_bytes
    
    async def _evict_from_warm(self, needed_bytes: int) -> None:
        """Evict LRU objects from warm tier to cold."""
        sorted_objects = sorted(
            self._warm.values(),
            key=lambda o: o.last_accessed
        )
        
        freed = 0
        for obj in sorted_objects:
            if freed >= needed_bytes:
                break
            
            # Move to cold
            await self._store_cold(obj)
            del self._warm[obj.object_id]
            self._warm_usage -= obj.size_bytes
            freed += obj.size_bytes
    
    async def retrieve(self, object_id: str) -> Result[Optional[MemoryObject]]:
        async with self._lock:
            # Check hot
            if object_id in self._hot:
                obj = self._hot[object_id]
                obj.last_accessed = datetime.utcnow()
                obj.access_count += 1
                return success(obj)
            
            # Check warm
            if object_id in self._warm:
                obj = self._warm[object_id]
                obj.last_accessed = datetime.utcnow()
                obj.access_count += 1
                
                # Promote to hot if frequently accessed
                if obj.access_count > 10 and obj.size_bytes < 10 * 1024 * 1024:
                    await self._promote_to_hot(object_id)
                
                return success(obj)
            
            # Check cold
            if object_id in self._cold_index:
                index_entry = self._cold_index[object_id]
                load_result = await self._storage.load(index_entry["key"])
                if load_result.is_failure():
                    return failure(load_result.unwrap_err())
                
                obj = MemoryObject(
                    object_id=object_id,
                    object_type=index_entry["metadata"].get("object_type", "unknown"),
                    data=load_result.unwrap(),
                    metadata=index_entry["metadata"],
                    tier=MemoryTier.COLD,
                    size_bytes=index_entry["size"]
                )
                
                # Decompress if needed
                obj = await self._decompress_object(obj)
                
                # Promote to warm
                await self._store_warm(obj)
                del self._cold_index[object_id]
                self._cold_usage -= obj.size_bytes
                
                return success(obj)
            
            return success(None)
    
    async def _promote_to_hot(self, object_id: str) -> None:
        """Promote object from warm to hot."""
        if object_id in self._warm:
            obj = self._warm[object_id]
            if self._hot_usage + obj.size_bytes <= self._config.hot_tier_max_gb * 1024 * 1024 * 1024:
                del self._warm[object_id]
                self._warm_usage -= obj.size_bytes
                self._hot[object_id] = obj
                self._hot_usage += obj.size_bytes
                obj.tier = MemoryTier.HOT
    
    def get_usage_stats(self) -> Dict[str, Any]:
        return {
            "hot": {"count": len(self._hot), "usage_mb": self._hot_usage / (1024 * 1024), "max_mb": self._config.hot_tier_max_gb * 1024},
            "warm": {"count": len(self._warm), "usage_mb": self._warm_usage / (1024 * 1024), "max_mb": self._config.warm_tier_max_gb * 1024},
            "cold": {"count": len(self._cold_index), "usage_mb": self._cold_usage / (1024 * 1024), "max_mb": self._config.cold_tier_max_gb * 1024}
        }


class MemoryStoreImpl:
    """Main memory store implementation."""
    
    def __init__(
        self,
        config: MemoryConfig,
        vector_search: VectorSearchPort,
        compression: CompressionPort,
        storage: StoragePort
    ):
        self._config = config
        self._vector_search = vector_search
        self._compression = compression
        self._storage = storage
        
        # Tiered storage
        self._tiered_storage = TieredStorage(config, storage)
        
        # Vector index
        self._vector_index = VectorIndex(config, compression)
        
        # Graph edges
        self._graph_edges: Dict[Tuple[str, str, str], MemoryGraphEdge] = {}
        
        # Manifests
        self._manifests: Dict[str, MemoryManifest] = {}
        
        # Checkpointing
        self._checkpoint_timer: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start background tasks."""
        self._checkpoint_timer = asyncio.create_task(self._checkpoint_loop())
    
    async def stop(self) -> None:
        """Stop background tasks."""
        if self._checkpoint_timer:
            self._checkpoint_timer.cancel()
            try:
                await self._checkpoint_timer
            except asyncio.CancelledError:
                pass
    
    async def _checkpoint_loop(self) -> None:
        """Periodic checkpoint to persistent storage."""
        while True:
            try:
                await asyncio.sleep(self._config.checkpoint_interval_seconds)
                await self._checkpoint()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint failed: {e}")
    
    async def _checkpoint(self) -> None:
        """Save current state to persistent storage."""
        # Save manifests
        for manifest_id, manifest in self._manifests.items():
            data = pickle.dumps(manifest)
            await self._storage.save(f"manifest/{manifest_id}", data, {"type": "manifest"})
        
        # Save graph edges
        edges_data = {f"{s}|{t}|{r}": e.__dict__ for (s, t, r), e in self._graph_edges.items()}
        await self._storage.save("graph_edges", pickle.dumps(edges_data), {"type": "graph_edges"})
        
        logger.info("Memory checkpoint completed")
    
    # Atom operations
    async def store_atom(self, atom: MemoryAtom) -> Result[MemoryId]:
        try:
            # Create memory object
            obj = MemoryObject(
                object_id=str(atom.atom_id),
                object_type="atom",
                data=pickle.dumps(atom),
                metadata={
                    "content_type": atom.content_type,
                    "source": atom.source,
                    "tags": atom.tags,
                    "importance": atom.importance
                }
            )
            
            # Store in tiered storage
            result = await self._tiered_storage.store(obj)
            if result.is_failure():
                return failure(result.unwrap_err())
            
            # Add to vector index if embedding exists
            if atom.embedding:
                await self._vector_index.add(
                    [str(atom.atom_id)],
                    [atom.embedding],
                    [{"content_type": atom.content_type, "tags": atom.tags}]
                )
            
            metrics.counter("memory_atoms_stored", 1)
            return success(atom.atom_id)
            
        except Exception as e:
            return failure(MemoryError(f"Failed to store atom: {e}"))
    
    async def get_atom(self, atom_id: MemoryId) -> Result[Optional[MemoryAtom]]:
        try:
            result = await self._tiered_storage.retrieve(str(atom_id))
            if result.is_failure():
                return failure(result.unwrap_err())
            
            obj = result.unwrap()
            if obj is None:
                return success(None)
            
            atom = pickle.loads(obj.data)
            return success(atom)
            
        except Exception as e:
            return failure(MemoryError(f"Failed to get atom: {e}"))
    
    # Manifest operations
    async def store_manifest(self, manifest: MemoryManifest) -> Result[UUID]:
        try:
            self._manifests[str(manifest.manifest_id)] = manifest
            
            # Store in tiered storage
            obj = MemoryObject(
                object_id=str(manifest.manifest_id),
                object_type="manifest",
                data=pickle.dumps(manifest),
                metadata={
                    "object_type": manifest.object_type,
                    "object_id": manifest.object_id,
                    "atom_count": len(manifest.atom_ids)
                }
            )
            
            result = await self._tiered_storage.store(obj)
            if result.is_failure():
                return failure(result.unwrap_err())
            
            return success(manifest.manifest_id)
            
        except Exception as e:
            return failure(MemoryError(f"Failed to store manifest: {e}"))
    
    async def get_manifest(self, manifest_id: UUID) -> Result[Optional[MemoryManifest]]:
        manifest = self._manifests.get(str(manifest_id))
        if manifest:
            return success(manifest)
        
        # Try loading from storage
        result = await self._storage.load(f"manifest/{manifest_id}")
        if result.is_success():
            manifest = pickle.loads(result.unwrap())
            self._manifests[str(manifest_id)] = manifest
            return success(manifest)
        
        return success(None)
    
    async def get_by_manifest(self, object_type: str, object_id: str) -> Result[List[MemoryAtom]]:
        # Find manifest
        manifest = None
        for m in self._manifests.values():
            if m.object_type == object_type and m.object_id == object_id:
                manifest = m
                break
        
        if not manifest:
            return success([])
        
        # Retrieve atoms
        atoms = []
        for atom_id in manifest.atom_ids:
            atom_result = await self.get_atom(atom_id)
            if atom_result.is_success() and atom_result.unwrap():
                atoms.append(atom_result.unwrap())
        
        return success(atoms)
    
    # Graph operations
    async def add_edge(self, edge: MemoryGraphEdge) -> Result[None]:
        key = (str(edge.source_id), str(edge.target_id), edge.relationship_type)
        self._graph_edges[key] = edge
        return success(None)
    
    async def get_edges(self, source_id: Optional[MemoryId] = None, target_id: Optional[MemoryId] = None) -> Result[List[MemoryGraphEdge]]:
        results = []
        for (s, t, r), edge in self._graph_edges.items():
            if source_id and s != str(source_id):
                continue
            if target_id and t != str(target_id):
                continue
            results.append(edge)
        return success(results)
    
    # Search operations
    async def search(self, query: str, limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> Result[List[MemoryAtom]]:
        # For text search, would use embedding
        # For now, search metadata in atoms
        # This is simplified - real implementation would use vector search
        
        results = []
        # Would search through atoms
        # For now, return empty
        return success(results)
    
    async def vector_search(self, vector: List[float], top_k: int, filters: Optional[Dict[str, Any]] = None) -> Result[List[Dict[str, Any]]]:
        return await self._vector_index.search(vector, top_k, filters)
    
    # Stats
    def get_stats(self) -> Dict[str, Any]:
        return {
            "tiered_storage": self._tiered_storage.get_usage_stats(),
            "vector_index": self._vector_index.get_stats(),
            "manifests": len(self._manifests),
            "graph_edges": len(self._graph_edges)
        }


# RaBitQ Compression Implementation
class RaBitQCompressor:
    """RaBitQ vector quantization for memory-efficient similarity search."""
    
    def __init__(self, bits_per_dim: int = 1):
        self._bits_per_dim = bits_per_dim
        self._codebook: Optional[np.ndarray] = None
        self._dimension: Optional[int] = None
    
    def train(self, vectors: np.ndarray, n_codebooks: int = 256) -> None:
        """Train codebook using k-means."""
        from sklearn.cluster import KMeans
        
        self._dimension = vectors.shape[1]
        n_samples = vectors.shape[0]
        
        # Reshape for k-means: each dimension separately
        # For 1-bit, we just need centroid per dimension
        if self._bits_per_dim == 1:
            # Simple threshold at median per dimension
            self._codebook = np.median(vectors, axis=0).reshape(1, -1)
        else:
            # Multi-bit: use k-means per dimension group
            kmeans = KMeans(n_clusters=2**self._bits_per_dim, random_state=42)
            # This is simplified - real RaBitQ uses product quantization
            self._codebook = kmeans.fit_predict(vectors).reshape(-1, 1)
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors to compact representation."""
        if self._codebook is None:
            raise ValueError("Codebook not trained")
        
        if self._bits_per_dim == 1:
            # 1-bit quantization: sign of (vector - median)
            return (vectors > self._codebook).astype(np.uint8)
        else:
            # Multi-bit: would use product quantization
            return vectors.astype(np.uint8)
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes back to approximate vectors."""
        if self._codebook is None:
            raise ValueError("Codebook not trained")
        
        if self._bits_per_dim == 1:
            # Reconstruct using codebook
            return codes.astype(np.float32) * self._codebook
        else:
            return codes.astype(np.float32)
    
    def compressed_size(self, n_vectors: int) -> int:
        """Calculate compressed size in bytes."""
        return n_vectors * self._dimension * self._bits_per_dim // 8


class TurboQuantCompressor:
    """TurboQuant for model weight quantization."""
    
    def __init__(self):
        self._scales: Dict[str, np.ndarray] = {}
        self._zero_points: Dict[str, np.ndarray] = {}
    
    def quantize_weights(self, weights: Dict[str, np.ndarray], bits: int = 4) -> Dict[str, np.ndarray]:
        """Quantize model weights to lower precision."""
        quantized = {}
        
        for name, weight in weights.items():
            if weight.dtype != np.float32 and weight.dtype != np.float16:
                quantized[name] = weight
                continue
            
            # Symmetric quantization
            max_val = np.max(np.abs(weight))
            scale = max_val / (2**(bits-1) - 1)
            
            quantized_weight = np.round(weight / scale).astype(np.int8)
            quantized_weight = np.clip(quantized_weight, -2**(bits-1), 2**(bits-1) - 1)
            
            self._scales[name] = np.array([scale], dtype=np.float32)
            quantized[name] = quantized_weight
        
        return quantized
    
    def dequantize_weights(self, quantized: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Dequantize weights back to float."""
        dequantized = {}
        
        for name, qweight in quantized.items():
            if name in self._scales:
                dequantized[name] = qweight.astype(np.float32) * self._scales[name]
            else:
                dequantized[name] = qweight.astype(np.float32)
        
        return dequantized


# 1-bit Adam Gradient Compression
class OneBitAdamCompressor:
    """1-bit Adam gradient compression for distributed training."""
    
    def __init__(self):
        self._momentum: Dict[str, np.ndarray] = {}
        self._velocity: Dict[str, np.ndarray] = {}
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps = 1e-8
    
    def compress_gradient(self, name: str, gradient: np.ndarray) -> np.ndarray:
        """Compress gradient to 1-bit (sign)."""
        # Update momentum
        if name not in self._momentum:
            self._momentum[name] = np.zeros_like(gradient)
            self._velocity[name] = np.zeros_like(gradient)
        
        self._momentum[name] = self._beta1 * self._momentum[name] + (1 - self._beta1) * gradient
        self._velocity[name] = self._beta2 * self._velocity[name] + (1 - self._beta2) * (gradient ** 2)
        
        # Bias correction
        m_hat = self._momentum[name] / (1 - self._beta1)
        v_hat = self._velocity[name] / (1 - self._beta2)
        
        # Compute update
        update = m_hat / (np.sqrt(v_hat) + self._eps)
        
        # 1-bit compression: sign of update
        return np.sign(update).astype(np.int8)
    
    def decompress_gradient(self, name: str, compressed: np.ndarray, learning_rate: float = 1e-3) -> np.ndarray:
        """Decompress gradient (approximate)."""
        # Reconstruct update direction
        update = compressed.astype(np.float32)
        
        # Scale by learning rate and momentum magnitude
        if name in self._momentum:
            momentum_mag = np.mean(np.abs(self._momentum[name]))
            update = update * learning_rate * momentum_mag
        
        return update