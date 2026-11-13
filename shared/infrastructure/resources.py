"""Infrastructure: open-source resource integration (HF, ModelScope, ArXiv, DuckDuckGo, GitHub, WorldMonitor)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ResourceResult:
    source: str
    query: str
    items: List[Dict[str, Any]] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    success: bool = True
    error: Optional[str] = None


class OpenSourceMarket:
    """Client for Hugging Face, ModelScope, GitHub open source model/skill downloads."""

    HF_API = "https://huggingface.co/api/models"
    MODELSCOPE_API = "https://modelscope.cn/api/v1/models"
    GITHUB_API = "https://api.github.com/search/repositories"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def _get(self, url: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "G9-AutonomousSystem/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def search_huggingface(self, query: str, limit: int = 10) -> ResourceResult:
        try:
            data = self._get(self.HF_API, {"search": query, "limit": str(limit)})
            items = []
            for m in data:
                items.append({
                    "id": m.get("id"),
                    "author": m.get("author"),
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "tags": m.get("tags", []),
                    "url": f"https://huggingface.co/{m.get('id')}",
                })
            return ResourceResult(source="huggingface", query=query, items=items)
        except Exception as e:
            return ResourceResult(source="huggingface", query=query, items=[], success=False, error=str(e))

    def search_modelscope(self, query: str, limit: int = 10) -> ResourceResult:
        try:
            data = self._get(self.MODELSCOPE_API, {"q": query, "limit": str(limit)})
            items = []
            for m in data.get("data", {}).get("list", [])[:limit]:
                items.append({
                    "id": m.get("modelId"),
                    "author": m.get("author"),
                    "downloads": m.get("downloadCount", 0),
                    "likes": m.get("likeCount", 0),
                    "url": f"https://modelscope.cn/models/{m.get('modelId')}",
                })
            return ResourceResult(source="modelscope", query=query, items=items)
        except Exception as e:
            return ResourceResult(source="modelscope", query=query, items=[], success=False, error=str(e))

    def search_github(self, query: str, limit: int = 10) -> ResourceResult:
        try:
            data = self._get(self.GITHUB_API, {"q": query, "per_page": str(limit)})
            items = []
            for m in data.get("items", []):
                items.append({
                    "id": m.get("full_name"),
                    "author": m.get("owner", {}).get("login"),
                    "stars": m.get("stargazers_count", 0),
                    "forks": m.get("forks_count", 0),
                    "url": m.get("html_url"),
                    "clone_url": m.get("clone_url"),
                })
            return ResourceResult(source="github", query=query, items=items)
        except Exception as e:
            return ResourceResult(source="github", query=query, items=[], success=False, error=str(e))


class LiteratureLibrary:
    """ArXiv and WorldMonitor literature fetcher."""

    ARXIV_API = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def search_arxiv(self, query: str, max_results: int = 10) -> ResourceResult:
        try:
            params = {
                "search_query": f"all:{query}",
                "start": "0",
                "max_results": str(max_results),
            }
            data = self._get(self.ARXIV_API, params)
            items = self._parse_arxiv_feed(data)
            return ResourceResult(source="arxiv", query=query, items=items)
        except Exception as e:
            return ResourceResult(source="arxiv", query=query, items=[], success=False, error=str(e))

    def _get(self, url: str, params: Optional[Dict[str, str]] = None) -> str:
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "G9-AutonomousSystem/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read().decode("utf-8")

    def _parse_arxiv_feed(self, xml_text: str) -> List[Dict[str, Any]]:
        import re
        items = []
        entries = re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL)
        for entry in entries:
            title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            link = re.search(r'<id>(.*?)</id>', entry)
            items.append({
                "title": title.group(1).strip() if title else "",
                "summary": summary.group(1).strip() if summary else "",
                "url": link.group(1) if link else "",
            })
        return items

    def worldmonitor_search(self, query: str) -> ResourceResult:
        """WorldMonitor: open real-time news/monitor search via DuckDuckGo HTML."""
        try:
            items = self._duckduckgo_html(query)
            return ResourceResult(source="worldmonitor", query=query, items=items)
        except Exception as e:
            return ResourceResult(source="worldmonitor", query=query, items=[], success=False, error=str(e))

    def _duckduckgo_html(self, query: str) -> List[Dict[str, Any]]:
        url = "https://duckduckgo.com/html/"
        data = urllib.parse.urlencode({"q": query}).encode()
        req = urllib.request.Request(url, data=data, headers={
            "User-Agent": "Mozilla/5.0",
            "Content-Type": "application/x-www-form-urlencoded",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        items = []
        for m in re.finditer(r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>', html, re.DOTALL):
            href = m.group(1)
            title = re.sub(r"<.*?>", "", m.group(2)).strip()
            items.append({"title": title, "url": href})
        return items[:10]


class DownloadManager:
    """Download open-source resources (models, skills, papers) with SHA-256 verification."""

    def __init__(self, download_dir: str = "downloads"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def download_file(self, url: str, dest: Optional[str] = None, expected_sha256: Optional[str] = None) -> str:
        dest_path = Path(dest) if dest else Path(self.download_dir) / url.split("/")[-1]
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "G9-AutonomousSystem/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                f.write(chunk)
        if expected_sha256:
            actual = self._sha256(dest_path)
            if actual != expected_sha256:
                raise ValueError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual}")
        return str(dest_path)

    @staticmethod
    def _sha256(path) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


__all__ = [
    "ResourceResult",
    "OpenSourceMarket",
    "LiteratureLibrary",
    "DownloadManager",
]
