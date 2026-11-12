"""
Infrastructure - RAG System with Open Source Integration.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
import asyncio
import aiohttp
import feedparser
import json
import hashlib
from urllib.parse import urlencode, quote

from ..shared.kernel.result import Result, success, failure
from ..shared.exceptions import ExternalServiceError, ResourceExhaustedError
from ..shared.observability import get_logger, get_metrics, traced, metered
from ..domain.entities import MemoryAtom, MemoryId
from ..application.ports import LLMProviderPort, VectorSearchPort, OpenSourceSearchPort, EmbeddingPort


logger = get_logger("infrastructure.rag")
metrics = get_metrics()


class DataSource(str, Enum):
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    CROSSREF = "crossref"
    OPENALEX = "openalex"
    HUGGINGFACE_PAPERS = "huggingface_papers"
    PAPERS_WITH_CODE = "papers_with_code"
    GITHUB_TRENDING = "github_trending"
    HACKER_NEWS = "hacker_news"
    LOBSTEERS = "lobsteers"
    RSS_FEEDS = "rss_feeds"


@dataclass
class SearchResult:
    """Unified search result."""
    id: str
    source: DataSource
    title: str
    abstract: str
    authors: List[str]
    url: str
    published_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeSearchResult:
    """Code search result."""
    id: str
    source: str
    repository: str
    file_path: str
    code_snippet: str
    language: str
    stars: int
    url: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSearchResult:
    """Model search result."""
    id: str
    source: str
    name: str
    description: str
    task: str
    framework: str
    size: str
    url: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArxivClient:
    """arXiv API client."""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
    
    async def search(self, query: str, max_results: int = 10) -> Result[List[SearchResult]]:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            async with self._session.get(self.BASE_URL, params=params) as resp:
                if resp.status != 200:
                    return failure(ExternalServiceError(f"arXiv API error: {resp.status}"))
                
                content = await resp.text()
                return self._parse_response(content)
        except Exception as e:
            return failure(ExternalServiceError(f"arXiv search failed: {e}"))
    
    async def get_trending(self, category: str = "cs.AI", max_results: int = 10) -> Result[List[SearchResult]]:
        query = f"cat:{category}"
        return await self.search(query, max_results)
    
    def _parse_response(self, xml_content: str) -> Result[List[SearchResult]]:
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(xml_content)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            
            results = []
            for entry in root.findall("atom:entry", ns):
                arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
                title = entry.find("atom:title", ns).text.strip()
                summary = entry.find("atom:summary", ns).text.strip()
                
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns).text
                    authors.append(name)
                
                published = entry.find("atom:published", ns).text
                published_at = datetime.fromisoformat(published.replace("Z", "+00:00"))
                
                results.append(SearchResult(
                    id=arxiv_id,
                    source=DataSource.ARXIV,
                    title=title,
                    abstract=summary,
                    authors=authors,
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    published_at=published_at,
                    metadata={"categories": [c.get("term") for c in entry.findall("arxiv:primary_category", ns)]}
                ))
            
            return success(results)
        except Exception as e:
            return failure(ExternalServiceError(f"Failed to parse arXiv response: {e}"))


class SemanticScholarClient:
    """Semantic Scholar API client."""
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, session: aiohttp.ClientSession, api_key: Optional[str] = None):
        self._session = session
        self._api_key = api_key
    
    async def search(self, query: str, max_results: int = 10) -> Result[List[SearchResult]]:
        headers = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        
        params = {
            "query": query,
            "limit": max_results,
            "fields": "paperId,title,abstract,authors,year,venue,url,citationCount,fieldsOfStudy"
        }
        
        try:
            async with self._session.get(f"{self.BASE_URL}/paper/search", params=params, headers=headers) as resp:
                if resp.status != 200:
                    return failure(ExternalServiceError(f"Semantic Scholar API error: {resp.status}"))
                
                data = await resp.json()
                return self._parse_response(data)
        except Exception as e:
            return failure(ExternalServiceError(f"Semantic Scholar search failed: {e}"))
    
    async def get_trending(self, field: str = "computer-science", max_results: int = 10) -> Result[List[SearchResult]]:
        # Get recent highly-cited papers
        params = {
            "query": f"fieldsOfStudy:{field}",
            "limit": max_results,
            "fields": "paperId,title,abstract,authors,year,venue,url,citationCount",
            "sort": "citationCount"
        }
        
        headers = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        
        try:
            async with self._session.get(f"{self.BASE_URL}/paper/search", params=params, headers=headers) as resp:
                if resp.status != 200:
                    return failure(ExternalServiceError(f"Semantic Scholar API error: {resp.status}"))
                
                data = await resp.json()
                return self._parse_response(data)
        except Exception as e:
            return failure(ExternalServiceError(f"Semantic Scholar trending failed: {e}"))
    
    def _parse_response(self, data: Dict) -> Result[List[SearchResult]]:
        try:
            results = []
            for paper in data.get("data", []):
                authors = [a.get("name", "") for a in paper.get("authors", [])]
                
                results.append(SearchResult(
                    id=paper.get("paperId", ""),
                    source=DataSource.SEMANTIC_SCHOLAR,
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract", "") or "",
                    authors=authors,
                    url=paper.get("url", ""),
                    published_at=datetime(paper.get("year", 2024), 1, 1),
                    metadata={
                        "venue": paper.get("venue", ""),
                        "citation_count": paper.get("citationCount", 0),
                        "fields": paper.get("fieldsOfStudy", [])
                    }
                ))
            
            return success(results)
        except Exception as e:
            return failure(ExternalServiceError(f"Failed to parse Semantic Scholar response: {e}"))


class PubMedClient:
    """PubMed/NCBI API client."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, session: aiohttp.ClientSession, api_key: Optional[str] = None):
        self._session = session
        self._api_key = api_key
    
    async def search(self, query: str, max_results: int = 10) -> Result[List[SearchResult]]:
        # First, search for IDs
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        if self._api_key:
            search_params["api_key"] = self._api_key
        
        try:
            async with self._session.get(f"{self.BASE_URL}/esearch.fcgi", params=search_params) as resp:
                if resp.status != 200:
                    return failure(ExternalServiceError(f"PubMed search error: {resp.status}"))
                
                search_data = await resp.json()
                ids = search_data.get("esearchresult", {}).get("idlist", [])
                
                if not ids:
                    return success([])
                
                # Fetch details
                return await self._fetch_details(ids)
        except Exception as e:
            return failure(ExternalServiceError(f"PubMed search failed: {e}"))
    
    async def _fetch_details(self, ids: List[str]) -> Result[List[SearchResult]]:
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml"
        }
        if self._api_key:
            fetch_params["api_key"] = self._api_key
        
        try:
            async with self._session.get(f"{self.BASE_URL}/efetch.fcgi", params=fetch_params) as resp:
                if resp.status != 200:
                    return failure(ExternalServiceError(f"PubMed fetch error: {resp.status}"))
                
                xml_content = await resp.text()
                return self._parse_response(xml_content)
        except Exception as e:
            return failure(ExternalServiceError(f"PubMed fetch failed: {e}"))
    
    def _parse_response(self, xml_content: str) -> Result[List[SearchResult]]:
        import xml.etree.ElementTree as ET
        
        try:
            root = ET.fromstring(xml_content)
            results = []
            
            for article in root.findall(".//PubmedArticle"):
                pmid = article.find(".//PMID").text
                title = article.find(".//ArticleTitle").text or ""
                
                abstract_elem = article.find(".//Abstract/AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None else ""
                
                authors = []
                for author in article.findall(".//Author"):
                    lastname = author.find("LastName")
                    firstname = author.find("ForeName")
                    if lastname is not None and firstname is not None:
                        authors.append(f"{lastname.text} {firstname.text}")
                
                pub_date = article.find(".//PubDate")
                year = int(pub_date.find("Year").text) if pub_date is not None and pub_date.find("Year") is not None else 2024
                
                results.append(SearchResult(
                    id=pmid,
                    source=DataSource.PUBMED,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    published_at=datetime(year, 1, 1),
                    metadata={"journal": article.find(".//Journal/Title").text if article.find(".//Journal/Title") is not None else ""}
                ))
            
            return success(results)
        except Exception as e:
            return failure(ExternalServiceError(f"Failed to parse PubMed response: {e}"))


class GitHubTrendingClient:
    """GitHub Trending client."""
    
    BASE_URL = "https://github.com/trending"
    
    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
    
    async def get_trending(self, language: str = "", max_results: int = 10) -> Result[List[SearchResult]]:
        params = {}
        if language:
            params["language"] = language
        
        try:
            async with self._session.get(self.BASE_URL, params=params) as resp:
                if resp.status != 200:
                    return failure(ExternalServiceError(f"GitHub Trending error: {resp.status}"))
                
                html = await resp.text()
                return self._parse_trending(html, max_results)
        except Exception as e:
            return failure(ExternalServiceError(f"GitHub Trending failed: {e}"))
    
    def _parse_trending(self, html: str, max_results: int) -> Result[List[SearchResult]]:
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html, "html.parser")
            results = []
            
            for repo in soup.find_all("article", class_="Box-row")[:max_results]:
                link = repo.find("h2", class_="h3").find("a")
                repo_name = link.get("href").strip("/")
                repo_url = f"https://github.com/{repo_name}"
                
                description_elem = repo.find("p", class_="col-9")
                description = description_elem.text.strip() if description_elem else ""
                
                language_elem = repo.find("span", itemprop="programmingLanguage")
                language = language_elem.text.strip() if language_elem else ""
                
                stars_elem = repo.find("a", href=lambda x: x and "stargazers" in x)
                stars = int(stars_elem.text.strip().replace(",", "")) if stars_elem else 0
                
                results.append(SearchResult(
                    id=repo_name,
                    source=DataSource.GITHUB_TRENDING,
                    title=repo_name,
                    abstract=description,
                    authors=[],
                    url=repo_url,
                    published_at=datetime.utcnow(),
                    metadata={"language": language, "stars": stars}
                ))
            
            return success(results)
        except Exception as e:
            return failure(ExternalServiceError(f"Failed to parse GitHub Trending: {e}"))


class HackerNewsClient:
    """Hacker News API client."""
    
    BASE_URL = "https://hacker-news.firebaseio.com/v0"
    
    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
    
    async def get_trending(self, max_results: int = 10) -> Result[List[SearchResult]]:
        # Get top stories
        try:
            async with self._session.get(f"{self.BASE_URL}/topstories.json") as resp:
                if resp.status != 200:
                    return failure(ExternalServiceError(f"HN top stories error: {resp.status}"))
                
                story_ids = await resp.json()
                
                # Fetch details for top stories
                stories = []
                for story_id in story_ids[:max_results]:
                    story = await self._fetch_story(story_id)
                    if story:
                        stories.append(story)
                
                return success(stories)
        except Exception as e:
            return failure(ExternalServiceError(f"Hacker News failed: {e}"))
    
    async def _fetch_story(self, story_id: int) -> Optional[SearchResult]:
        try:
            async with self._session.get(f"{self.BASE_URL}/item/{story_id}.json") as resp:
                if resp.status != 200:
                    return None
                
                data = await resp.json()
                if data.get("type") != "story":
                    return None
                
                return SearchResult(
                    id=str(story_id),
                    source=DataSource.HACKER_NEWS,
                    title=data.get("title", ""),
                    abstract=data.get("text", "")[:500] if data.get("text") else "",
                    authors=[data.get("by", "")] if data.get("by") else [],
                    url=data.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
                    published_at=datetime.fromtimestamp(data.get("time", 0)),
                    metadata={"score": data.get("score", 0), "descendants": data.get("descendants", 0)}
                )
        except Exception:
            return None


class OpenSourceSearchImpl(OpenSourceSearchPort):
    """Open source search implementation."""
    
    def __init__(
        self,
        session: aiohttp.ClientSession,
        embedding_port: EmbeddingPort,
        vector_search: VectorSearchPort,
        config: Dict[str, Any]
    ):
        self._session = session
        self._embedding = embedding_port
        self._vector_search = vector_search
        self._config = config
        
        # Initialize clients
        self._arxiv = ArxivClient(session)
        self._semantic_scholar = SemanticScholarClient(session, config.get("semantic_scholar_api_key"))
        self._pubmed = PubMedClient(session, config.get("pubmed_api_key"))
        self._github = GitHubTrendingClient(session)
        self._hacker_news = HackerNewsClient(session)
    
    @traced("open_source.search_papers")
    async def search_papers(
        self,
        query: str,
        sources: List[str],
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        all_results = []
        
        source_map = {
            "arxiv": self._arxiv,
            "semantic_scholar": self._semantic_scholar,
            "pubmed": self._pubmed
        }
        
        for source_name in sources:
            client = source_map.get(source_name)
            if not client:
                continue
            
            result = await client.search(query, limit)
            if result.is_success():
                for paper in result.unwrap():
                    all_results.append({
                        "id": paper.id,
                        "source": paper.source.value,
                        "title": paper.title,
                        "abstract": paper.abstract,
                        "authors": paper.authors,
                        "url": paper.url,
                        "published_at": paper.published_at.isoformat(),
                        "metadata": paper.metadata
                    })
        
        # Deduplicate by title similarity
        unique_results = self._deduplicate(all_results)
        
        return success(unique_results[:limit])
    
    @traced("open_source.search_code")
    async def search_code(
        self,
        query: str,
        languages: List[str],
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        # Would use GitHub Code Search API or similar
        # For now, return trending repos matching query
        results = []
        
        for lang in languages[:3]:
            trending = await self._github.get_trending(language=lang, max_results=limit//len(languages))
            if trending.is_success():
                for repo in trending.unwrap():
                    if query.lower() in repo.title.lower() or query.lower() in repo.abstract.lower():
                        results.append({
                            "id": repo.id,
                            "source": "github",
                            "repository": repo.id,
                            "description": repo.abstract,
                            "language": repo.metadata.get("language", lang),
                            "stars": repo.metadata.get("stars", 0),
                            "url": repo.url
                        })
        
        return success(results[:limit])
    
    @traced("open_source.search_models")
    async def search_models(
        self,
        query: str,
        tasks: List[str],
        limit: int = 10
    ) -> Result[List[Dict[str, Any]]]:
        # Would use Hugging Face Hub API
        # For now, return mock results
        return success([
            {
                "id": f"model_{i}",
                "source": "huggingface",
                "name": f"model-for-{query}",
                "description": f"Model for {query}",
                "task": tasks[0] if tasks else "text-generation",
                "framework": "pytorch",
                "size": "1.5B",
                "url": f"https://huggingface.co/model_{i}"
            }
            for i in range(min(limit, 5))
        ])
    
    @traced("open_source.download_resource")
    async def download_resource(self, resource_id: str, resource_type: str) -> Result[bytes]:
        # Would download actual resource
        return success(b"Mock resource content")
    
    @traced("open_source.get_trending")
    async def get_trending(self, source: str, limit: int = 10) -> Result[List[Dict[str, Any]]]:
        client_map = {
            "arxiv": self._arxiv,
            "semantic_scholar": self._semantic_scholar,
            "pubmed": self._pubmed,
            "github_trending": self._github,
            "hacker_news": self._hacker_news
        }
        
        client = client_map.get(source)
        if not client:
            return failure(ExternalServiceError(f"Unknown source: {source}"))
        
        result = await client.get_trending(max_results=limit)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        papers = result.unwrap()
        return success([
            {
                "id": p.id,
                "source": p.source.value,
                "title": p.title,
                "abstract": p.abstract,
                "authors": p.authors,
                "url": p.url,
                "published_at": p.published_at.isoformat(),
                "metadata": p.metadata
            }
            for p in papers
        ])
    
    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate results by title similarity."""
        seen = set()
        unique = []
        
        for result in results:
            title_key = result["title"].lower().strip()
            # Simple deduplication
            if title_key not in seen:
                seen.add(title_key)
                unique.append(result)
        
        return unique


class RAGPipeline:
    """RAG Pipeline for retrieval-augmented generation."""
    
    def __init__(
        self,
        open_source_search: OpenSourceSearchPort,
        embedding_port: EmbeddingPort,
        vector_search: VectorSearchPort,
        llm_provider: LLMProviderPort,
        memory_store: 'MemoryStoreImpl',
        config: Dict[str, Any]
    ):
        self._open_source = open_source_search
        self._embedding = embedding_port
        self._vector_search = vector_search
        self._llm = llm_provider
        self._memory = memory_store
        self._config = config
    
    @traced("rag.retrieve_and_generate")
    async def retrieve_and_generate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> Result[str]:
        # Step 1: Embed query
        embedding_result = await self._embedding.embed([query])
        if embedding_result.is_failure():
            return failure(embedding_result.unwrap_err())
        
        query_embedding = embedding_result.unwrap()[0]
        
        # Step 2: Search vector index
        search_result = await self._vector_search.search(query_embedding, top_k)
        if search_result.is_failure():
            return failure(search_result.unwrap_err())
        
        # Step 3: Retrieve relevant atoms
        retrieved = []
        for hit in search_result.unwrap():
            atom_id = hit.get("id")
            if atom_id:
                atom_result = await self._memory.get_atom(MemoryId(UUID(atom_id)))
                if atom_result.is_success() and atom_result.unwrap():
                    retrieved.append(atom_result.unwrap())
        
        # Step 4: If not enough results, search open sources
        if len(retrieved) < top_k:
            open_results = await self._open_source.search_papers(
                query,
                sources=["arxiv", "semantic_scholar", "pubmed"],
                limit=top_k - len(retrieved)
            )
            
            if open_results.is_success():
                for paper in open_results.unwrap():
                    # Create atom from paper
                    atom = MemoryAtom(
                        atom_id=MemoryId(),
                        content=f"{paper['title']}\n{paper['abstract']}",
                        content_type="paper",
                        source=paper["source"],
                        source_id=paper["id"],
                        tags=["retrieved", "open_source"],
                        importance=0.8
                    )
                    retrieved.append(atom)
        
        # Step 5: Generate answer
        context_text = "\n\n".join([
            f"Source: {a.source} ({a.source_id})\n{a.content[:500]}"
            for a in retrieved
        ])
        
        prompt = f"""
        Answer the following question using the provided context:
        
        Question: {query}
        
        Context:
        {context_text}
        
        Provide a comprehensive answer with citations to sources.
        """
        
        system_prompt = "You are a helpful assistant that answers questions using retrieved sources. Always cite your sources."
        
        result = await self._llm.complete(prompt, system_prompt=system_prompt)
        if result.is_failure():
            return failure(result.unwrap_err())
        
        return success(result.unwrap())
    
    @traced("rag.index_documents")
    async def index_documents(self, documents: List[Dict[str, Any]]) -> Result[int]:
        """Index documents into vector store."""
        texts = [doc.get("content", "") for doc in documents]
        
        if not texts:
            return success(0)
        
        # Embed
        embedding_result = await self._embedding.embed(texts)
        if embedding_result.is_failure():
            return failure(embedding_result.unwrap_err())
        
        embeddings = embedding_result.unwrap()
        
        # Prepare for vector search
        ids = [str(uuid4()) for _ in documents]
        metadata = [
            {
                "source": doc.get("source", "unknown"),
                "source_id": doc.get("id", ""),
                "title": doc.get("title", ""),
                "tags": doc.get("tags", [])
            }
            for doc in documents
        ]
        
        # Add to vector index
        add_result = await self._vector_search.add(ids, embeddings, metadata)
        if add_result.is_failure():
            return failure(add_result.unwrap_err())
        
        # Store in memory
        for i, doc in enumerate(documents):
            atom = MemoryAtom(
                atom_id=MemoryId(UUID(ids[i])),
                content=doc.get("content", ""),
                content_type=doc.get("type", "document"),
                source=doc.get("source", "unknown"),
                source_id=doc.get("id", ""),
                tags=doc.get("tags", []),
                embedding=embeddings[i],
                importance=doc.get("importance", 1.0)
            )
            await self._memory.store_atom(atom)
        
        return success(len(documents))


class EmbeddingPortImpl:
    """Embedding port implementation using sentence transformers or similar."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._dimension = 384  # Default for MiniLM
    
    async def initialize(self) -> None:
        """Initialize embedding model."""
        # Would load model here
        # from sentence_transformers import SentenceTransformer
        # self._model = SentenceTransformer(self._model_name)
        # self._dimension = self._model.get_sentence_embedding_dimension()
        pass
    
    async def embed(self, texts: List[str]) -> Result[List[List[float]]]:
        """Generate embeddings for texts."""
        # Mock implementation
        import random
        embeddings = []
        for text in texts:
            # Deterministic pseudo-embedding based on text hash
            seed = hash(text) % 2**32
            random.seed(seed)
            embedding = [random.uniform(-1, 1) for _ in range(self._dimension)]
            # Normalize
            norm = sum(x*x for x in embedding) ** 0.5
            embedding = [x/norm for x in embedding]
            embeddings.append(embedding)
        
        return success(embeddings)