"""
GITHUB SERVİS MODÜLÜ (core/github_fetcher.py)

GitHub REST API ile iletişim kurar, gürültü yaratan dosyaları filtreler ve
analize uygun kaynak kod/dokümantasyon dosyalarını bellekte döndürür.
Hiçbir disk I/O veya CLI (input) bağımlılığı içermez.
"""

import base64
import os
import re
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from logger import logger

load_dotenv()

API_BASE_URL = "https://api.github.com"

ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".cs", ".cpp", ".c", ".h",
    ".java", ".kt", ".go", ".rs", ".rb", ".php", ".swift", ".vue",
    ".html", ".json", ".yaml", ".yml", ".xml", ".sql", ".md"
}

ALLOWED_FILENAMES = {
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Makefile", "requirements.txt", "Gemfile", "Cargo.toml"
}

IGNORED_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Cargo.lock", "composer.lock", "Gemfile.lock", "tsconfig.tsbuildinfo"
}

IGNORED_DIRS = {
    "node_modules", "dist", "build", ".git", ".github", ".vscode",
    ".idea", "venv", ".venv", "__pycache__", "coverage", "public", "assets"
}

MAX_FILE_SIZE_BYTES = 100 * 1024  # 100 KB


def parse_repo_url(url: str) -> Tuple[str, str]:
    """GitHub URL'sini (owner, repo_name) ikilisine ayrıştırır."""
    parsed = urlparse(url.strip())
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Geçersiz GitHub Repo URL'si. Örnek: https://github.com/kullanici/repo")
    return parts[0], re.sub(r"\.git$", "", parts[1])


def is_allowed_file(path: str, size: int = 0) -> bool:
    """Dosyanın yoluna, adına, boyutuna ve uzantısına göre indirilebilir olup olmadığını denetler."""
    parts = path.split("/")
    filename = parts[-1]

    if any(part in IGNORED_DIRS for part in parts[:-1]):
        return False

    if filename in IGNORED_FILENAMES or filename.endswith(".min.js") or filename.endswith(".min.css"):
        return False

    if size > MAX_FILE_SIZE_BYTES:
        logger.debug(f"Boyut sınırı nedeniyle atlandı ({size / 1024:.1f} KB): {path}")
        return False

    if filename in ALLOWED_FILENAMES:
        return True

    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


class GitHubFetcher:
    def __init__(self, token: Optional[str] = None):
        raw_token = token or os.getenv("GITHUB_TOKEN")
        self.token = raw_token.strip().strip('"').strip("'") if raw_token else None

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get(self, path: str, **params) -> dict:
        """GitHub API'ye GET isteği gönderir ve hata durumlarını yönetir."""
        response = requests.get(
            f"{API_BASE_URL}{path}",
            headers=self._headers(),
            params=params,
            timeout=20,
        )
        if response.status_code == 401:
            raise RuntimeError("GitHub Token geçersiz. .env içindeki GITHUB_TOKEN değerini kontrol edin.")
        if response.status_code == 404:
            raise RuntimeError("Repo bulunamadı (özel repo olabilir veya token yetkisi yetersiz).")
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            raise RuntimeError("GitHub API rate limit doldu. .env dosyasına geçerli bir GITHUB_TOKEN ekleyin.")
        response.raise_for_status()
        return response.json()

    def fetch_repository(
        self,
        owner: str,
        repo: str,
        progress_callback: Optional[Callable[[int, int, str, int], None]] = None,
    ) -> dict:
        """
        Repodaki uygun dosyaları indirir ve standart veri sözlüğünü bellekte döndürür.
        İsteğe bağlı `progress_callback(current_idx, total_count, file_path, file_size)` ile ilerlemeyi bildirir.
        """
        repo_info = self.get(f"/repos/{owner}/{repo}")
        branch = repo_info.get("default_branch", "main")
        languages = self.get(f"/repos/{owner}/{repo}/languages")

        tree_data = self.get(f"/repos/{owner}/{repo}/git/trees/{branch}", recursive=1)
        if tree_data.get("truncated"):
            logger.warning("Repo çok büyük, GitHub dosya ağacını kırptı. Bazı dosyalar eksik olabilir.")

        blobs: List[dict] = [
            e for e in tree_data.get("tree", [])
            if e.get("type") == "blob" and is_allowed_file(e["path"], e.get("size", 0))
        ]
        if not blobs:
            raise RuntimeError("Filtreleme sonrası indirilecek uygun kaynak dosyası kalmadı.")

        files: List[dict] = []
        total_size_bytes = 0

        for idx, blob in enumerate(blobs, 1):
            path = blob["path"]
            size = blob.get("size", 0)
            if progress_callback:
                progress_callback(idx, len(blobs), path, size)

            try:
                blob_data = self.get(f"/repos/{owner}/{repo}/git/blobs/{blob['sha']}")
                content = base64.b64decode(blob_data["content"]).decode("utf-8", errors="replace")
                total_size_bytes += size
            except Exception as e:
                content = f"[İndirme/Çözümleme Hatası: {e}]"

            files.append({
                "path": path,
                "size": size,
                "sha": blob["sha"],
                "content": content,
            })

        return {
            "repository": repo_info.get("full_name") or f"{owner}/{repo}",
            "description": repo_info.get("description"),
            "stargazers_count": repo_info.get("stargazers_count", 0),
            "languages": list(languages.keys()),
            "total_files": len(files),
            "total_size_kb": round(total_size_bytes / 1024, 2),
            "directory_structure": [f["path"] for f in files],
            "files": files,
        }