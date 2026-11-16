'English text removed.'
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np

from .base import Document, FilterableVectorStore, SearchResult

logger = logging.getLogger(__name__)


class FaissVectorStore(FilterableVectorStore):
    'English text removed.'

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "FLAT",
        metric_type: str = "COSINE",
        **kwargs: Any,
    ) -> None:
        super().__init__(embedding_dim, index_type, metric_type, **kwargs)

                                                                     
        self._index = faiss.IndexFlatIP(embedding_dim)

                                            
                                                          
        self._id_list: List[str] = []
                                          
        self._metadata: Dict[str, Dict[str, Any]] = {}
                                  
        self._texts: Dict[str, str] = {}

                                   

    def add(
        self,
        documents: List[Document],
        batch_size: int = 1000,
    ) -> List[str]:
        'English text removed.'
        if not documents:
            return []

        ids = []
        vectors = []

        for doc in documents:
            emb = np.array(doc.embedding, dtype=np.float32).flatten()
            if emb.shape[0] != self.embedding_dim:
                raise ValueError(
                    f"Embedding dim mismatch: expected {self.embedding_dim}, "
                    f"got {emb.shape[0]} for doc {doc.id}"
                )
            vectors.append(emb)
            ids.append(doc.id)
            self._id_list.append(doc.id)
            if doc.metadata:
                self._metadata[doc.id] = doc.metadata
            if doc.text:
                self._texts[doc.id] = doc.text

                    
        vectors_np = np.stack(vectors).astype(np.float32)
        faiss.normalize_L2(vectors_np)
        self._index.add(vectors_np)

        logger.debug(f"Added {len(ids)} documents, total={self._index.ntotal}")
        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[SearchResult]:
        'English text removed.'
        if self._index.ntotal == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

                           
        search_k = min(k * 10, self._index.ntotal) if filter_expr else min(k, self._index.ntotal)

        distances, indices = self._index.search(query, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:                     
                continue
            doc_id = self._id_list[idx]
            metadata = self._metadata.get(doc_id, {})

                  
            if filter_expr and not self._match_filter(metadata, filter_expr):
                continue

            results.append(SearchResult(
                id=doc_id,
                score=float(dist),                              
                metadata=metadata,
                text=self._texts.get(doc_id),
            ))

            if len(results) >= k:
                break

        return results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[List[SearchResult]]:
        'English text removed.'
        queries = np.array(query_embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        faiss.normalize_L2(queries)

        search_k = min(k * 10, self._index.ntotal) if filter_expr else min(k, self._index.ntotal)
        distances, indices = self._index.search(queries, search_k)

        all_results = []
        for q_dists, q_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(q_dists, q_indices):
                if idx < 0:
                    continue
                doc_id = self._id_list[idx]
                metadata = self._metadata.get(doc_id, {})
                if filter_expr and not self._match_filter(metadata, filter_expr):
                    continue
                results.append(SearchResult(
                    id=doc_id,
                    score=float(dist),
                    metadata=metadata,
                    text=self._texts.get(doc_id),
                ))
                if len(results) >= k:
                    break
            all_results.append(results)

        return all_results

    def delete(self, ids: List[str]) -> int:
        'English text removed.'
        ids_set = set(ids)
        keep_indices = [i for i, doc_id in enumerate(self._id_list) if doc_id not in ids_set]

        if len(keep_indices) == len(self._id_list):
            return 0            

        deleted = len(self._id_list) - len(keep_indices)

              
        if keep_indices:
            vectors = self._index.reconstruct_batch(keep_indices)
            new_index = faiss.IndexFlatIP(self.embedding_dim)
            new_index.add(vectors)
            self._index = new_index
            self._id_list = [self._id_list[i] for i in keep_indices]
        else:
            self._index = faiss.IndexFlatIP(self.embedding_dim)
            self._id_list = []

               
        for doc_id in ids:
            self._metadata.pop(doc_id, None)
            self._texts.pop(doc_id, None)

        logger.debug(f"Deleted {deleted} documents, remaining={self._index.ntotal}")
        return deleted

    def get(self, ids: List[str]) -> List[Optional[Document]]:
        'English text removed.'
        results = []
        id_to_idx = {doc_id: i for i, doc_id in enumerate(self._id_list)}

        for doc_id in ids:
            idx = id_to_idx.get(doc_id)
            if idx is None:
                results.append(None)
            else:
                emb = self._index.reconstruct(idx)
                results.append(Document(
                    id=doc_id,
                    embedding=emb,
                    text=self._texts.get(doc_id),
                    metadata=self._metadata.get(doc_id, {}),
                ))
        return results

    def count(self) -> int:
        return self._index.ntotal

    def clear(self) -> None:
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._id_list.clear()
        self._metadata.clear()
        self._texts.clear()

                               

    def persist(self, path: str) -> None:
        'English text removed.'
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

                        
        faiss.write_index(self._index, str(save_dir / "index.faiss"))

               
        meta = {
            "embedding_dim": self.embedding_dim,
            "id_list": self._id_list,
            "metadata": self._metadata,
            "texts": self._texts,
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(f"Persisted {self._index.ntotal} vectors to {save_dir}")

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "FaissVectorStore":
        'English text removed.'
        load_dir = Path(path)

               
        with open(load_dir / "metadata.json", "r") as f:
            meta = json.load(f)

        embedding_dim = meta["embedding_dim"]
        store = cls(embedding_dim=embedding_dim, **kwargs)

                        
        store._index = faiss.read_index(str(load_dir / "index.faiss"))
        store._id_list = meta["id_list"]
        store._metadata = meta["metadata"]
        store._texts = meta.get("texts", {})

        logger.info(f"Loaded {store._index.ntotal} vectors from {load_dir}")
        return store

                                                    

    def search_by_layer(
        self,
        query_embedding: np.ndarray,
        layer: str,
        k: int = 10,
    ) -> List[SearchResult]:
        'English text removed.'
        return self.search(query_embedding, k=k, filter_expr=f"layer == '{layer}'")

    def search_by_time_range(
        self,
        query_embedding: np.ndarray,
        start_time: float,
        end_time: float,
        k: int = 10,
    ) -> List[SearchResult]:
        'English text removed.'
        return self.search(
            query_embedding, k=k,
            filter_expr=f"timestamp_start >= {start_time} AND timestamp_start <= {end_time}",
        )

    def search_by_service(
        self,
        query_embedding: np.ndarray,
        services: List[str],
        k: int = 10,
    ) -> List[SearchResult]:
        'English text removed.'
        services_str = ",".join(services)
        return self.search(
            query_embedding, k=k,
            filter_expr=f"affected_services IN [{services_str}]",
        )

                                

    def _match_filter(self, metadata: Dict[str, Any], filter_expr: str) -> bool:
        'English text removed.'
                   
        if " AND " in filter_expr:
            parts = filter_expr.split(" AND ")
            return all(self._match_filter(metadata, p.strip()) for p in parts)

        # "key == 'value'"
        if "==" in filter_expr:
            key, value = filter_expr.split("==", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            return str(metadata.get(key, "")) == value

        # "key >= number"
        if ">=" in filter_expr:
            key, value = filter_expr.split(">=", 1)
            key = key.strip()
            value = float(value.strip())
            return float(metadata.get(key, 0)) >= value

        # "key <= number"
        if "<=" in filter_expr:
            key, value = filter_expr.split("<=", 1)
            key = key.strip()
            value = float(value.strip())
            return float(metadata.get(key, 0)) <= value

        # "key IN [val1,val2]"
        if " IN " in filter_expr:
            key, values_str = filter_expr.split(" IN ", 1)
            key = key.strip()
            values = [v.strip() for v in values_str.strip("[]").split(",")]
            meta_val = metadata.get(key, [])
            if isinstance(meta_val, list):
                return bool(set(meta_val) & set(values))
            return str(meta_val) in values

        logger.warning(f"Unsupported filter expression: {filter_expr}")
        return True                   
