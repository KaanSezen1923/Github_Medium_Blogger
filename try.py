"""
Bu script, verilen GitHub reposunu tarar, gürültü yaratan dosyaları (lock dosyaları, 
CSS'ler, derleme klasörleri vb.) filtreler ve blog yazısı üretimi için sadece 
mimari açıdan önemli kod/dokümantasyon dosyalarını JSON dosyasına kaydeder.

GitHub token'ı proje klasöründeki .env dosyasından okunur:
    GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

Kurulum: pip install requests python-dotenv
Kullanım: python try.py
"""

import base64
import json
import os
import re
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

API = "https://api.github.com"

# İndirilmesine izin verilen dosya uzantıları (.css ve .scss çıkarıldı, .md eklendi)
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".cs", ".cpp", ".c", ".h",
    ".java", ".kt", ".go", ".rs", ".rb", ".php", ".swift", ".vue",
    ".html", ".json", ".yaml", ".yml", ".xml", ".sql", ".md"
}

# Uzantısı olmasa bile indirilmesi istenen tam dosya adları
ALLOWED_FILENAMES = {
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Makefile", "requirements.txt", "Gemfile", "Cargo.toml"
}

# Kesinlikle hariç tutulacak gürültü/kilit dosyaları
IGNORED_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", " poetry.lock",
    "Cargo.lock", "composer.lock", "Gemfile.lock", "tsconfig.tsbuildinfo"
}

# Kesinlikle hariç tutulacak klasörler
IGNORED_DIRS = {
    "node_modules", "dist", "build", ".git", ".github", ".vscode", 
    ".idea", "venv", ".venv", "__pycache__", "coverage", "public", "assets"
}

# Tek bir dosyanın alabileceği maksimum boyut (Byte cinsinden - Örn: 100 KB)
# Çok büyük veri/mock JSON dosyalarının token şişirmesini engeller.
MAX_FILE_SIZE_BYTES = 100 * 1024


def get_token() -> str | None:
    """.env dosyasından (veya ortam değişkenlerinden) GITHUB_TOKEN değerini döndürür."""
    token = os.getenv("GITHUB_TOKEN")
    if token:
        token = token.strip().strip('"').strip("'")
    return token or None


GITHUB_TOKEN = get_token()


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def gh_get(path: str, **params) -> dict:
    r = requests.get(f"{API}{path}", headers=_headers(), params=params, timeout=20)
    if r.status_code == 401:
        raise RuntimeError("Token geçersiz. .env içindeki GITHUB_TOKEN değerini kontrol et.")
    if r.status_code == 404:
        raise RuntimeError("Repo bulunamadı (özel olabilir veya token yetkisi yetersiz).")
    if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
        raise RuntimeError("GitHub rate limit doldu. .env dosyasına geçerli bir GITHUB_TOKEN ekle.")
    r.raise_for_status()
    return r.json()


def is_allowed_file(path: str, size: int = 0) -> bool:
    """Dosyanın yoluna, adına, boyutuna ve uzantısına göre filtrelenip filtrelenmeyeceğini kontrol eder."""
    parts = path.split("/")
    filename = parts[-1]

    # 1. Yasaklı klasör kontrolü (yolun herhangi bir parçasında varsa elenir)
    if any(part in IGNORED_DIRS for part in parts[:-1]):
        return False

    # 2. Yasaklı dosya adı kontrolü (örn. package-lock.json)
    if filename in IGNORED_FILENAMES or filename.endswith(".min.js") or filename.endswith(".min.css"):
        return False

    # 3. Dosya boyutu kontrolü (100 KB'dan büyük dosyaları atla)
    if size > MAX_FILE_SIZE_BYTES:
        print(f"  [ATLANDI - Boyut Sınırı ({size / 1024:.1f} KB)]: {path}")
        return False

    # 4. Tam eşleşen özel dosya adları (örn. Dockerfile)
    if filename in ALLOWED_FILENAMES:
        return True

    # 5. Uzantı kontrolü
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def main():
    if GITHUB_TOKEN:
        print("GITHUB_TOKEN .env dosyasından yüklendi.")
    else:
        print("UYARI: .env içinde GITHUB_TOKEN bulunamadı, isteklerde rate limit düşük olacak.")

    REPO_URL = input("GitHub Repo URL'sini girin: ")
    parsed = urlparse(REPO_URL.strip())
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Geçersiz GitHub Repo URL'si.")
    owner, repo_name = parts[0], re.sub(r"\.git$", "", parts[1])

    print(f"Repo bilgileri alınıyor: {owner}/{repo_name}...")
    repo_info = gh_get(f"/repos/{owner}/{repo_name}")
    branch = repo_info.get("default_branch", "main")

    languages = gh_get(f"/repos/{owner}/{repo_name}/languages")

    print(f"Dosya ağacı alınıyor (Branch: {branch})...")
    tree_data = gh_get(f"/repos/{owner}/{repo_name}/git/trees/{branch}", recursive=1)

    all_tree = tree_data.get("tree", [])

    # Sadece blob tipinde olan ve filtreleme kurallarına uyan dosyaları seç
    blobs = [
        e for e in all_tree
        if e["type"] == "blob" and is_allowed_file(e["path"], e.get("size", 0))
    ]

    print(f"Filtreleme sonrası indirilecek toplam dosya sayısı: {len(blobs)}")

    file_data_list = []
    total_size_bytes = 0

    for idx, blob in enumerate(blobs, 1):
        path = blob["path"]
        sha = blob["sha"]
        size = blob.get("size", 0)
        print(f"[{idx}/{len(blobs)}] İndiriliyor: {path} ({size / 1024:.1f} KB)")

        try:
            blob_data = gh_get(f"/repos/{owner}/{repo_name}/git/blobs/{sha}")
            content = base64.b64decode(blob_data["content"]).decode("utf-8", errors="replace")
            total_size_bytes += size
        except Exception as e:
            content = f"[İndirme/Çözümleme Hatası: {e}]"

        file_data_list.append({
            "path": path,
            "size": size,
            "sha": sha,
            "content": content
        })

    # LLM'in proje mimarisini kolayca anlaması için sadece dosya yollarının listesi
    directory_structure = [f["path"] for f in file_data_list]

    output_json = {
        "repository": repo_info.get("full_name"),
        "description": repo_info.get("description"),
        "stargazers_count": repo_info.get("stargazers_count"),
        "languages": list(languages.keys()),
        "total_files": len(file_data_list),
        "total_size_kb": round(total_size_bytes / 1024, 2),
        "directory_structure": directory_structure,
        "files": file_data_list
    }

    output_filename = f"{repo_name}_code_contents.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"İşlem tamamlandı! '{output_filename}' kaydedildi.")
    print(f"İndirilen Dosya Sayısı : {len(file_data_list)}")
    print(f"Toplam Kod Boyutu      : {output_json['total_size_kb']} KB")
    print("=" * 50)


if __name__ == "__main__":
    main()