import glob
import json
import os
import re
import sys
import time
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from logger import logger

# .env dosyasındaki ortam değişkenlerini yükle
load_dotenv()

# Çok büyük dosyalarda hız ve doğruluk için içerik kırpma sınırı
MAX_CONTENT_CHARS = 12000


# =====================================================================
# 1. VERİ ŞEMALARI
# =====================================================================

class CodeSnippet(BaseModel):
    code: str = Field(
        description="Dosyadan seçilen, Medium blog yazısında paylaşmaya değer en çarpıcı 10-25 satırlık gerçek kod kesiti. Sadece iş mantığını (business logic) içermeli, gereksiz import veya uzun HTML/JSX yığınlarını içermemelidir."
    )
    explanation: str = Field(
        description="Bu kod kesitinin neden önemli olduğu, hangi problemi çözdüğü ve ne yaptığının 1-2 cümlelik teknik açıklaması."
    )


class FileAnalysisSchema(BaseModel):
    responsibility: str = Field(
        description="Bu dosyanın projedeki temel sorumluluğunu ve mimarideki yerini anlatan 3-4 cümlelik kapsamlı teknik özet."
    )
    libraries_and_hooks: List[str] = Field(
        description="Dosyada kullanılan önemli kütüphaneler, React/framework hook'ları, API fonksiyonları veya yardımcı modüller."
    )
    engineering_highlights: List[str] = Field(
        description="Bu dosyadaki dikkat çekici mühendislik çözümleri ve tasarım desenleri (Örn: Optimistic UI Update, Debouncing, Client-side Fallback Mekanizması, Custom Hook, Caching, State Senkronizasyonu vb.)."
    )
    blog_worthiness_score: int = Field(
        description="Bu dosyanın teknik bir Medium blog yazısında anlatılmaya ne kadar değer olduğu (1: Sıradan/Boilerplate/Config, 10: Çok özgün, öğretici ve derinlikli mimari).",
        ge=1,
        le=10
    )
    best_snippet: Optional[CodeSnippet] = Field(
        default=None,
        description="Eğer dosyada blogda gösterilmeye değer özgün bir kod yapısı varsa en iyi kod kesiti, dosya çok sıradansa null."
    )


# =====================================================================
# 2. SİSTEM PROMPTU
# =====================================================================

ANALYZER_SYSTEM_PROMPT = """Sen kıdemli bir Yazılım Mimarı (Senior Software Architect) ve Teknik İçerik Editörüsün.
Görevin, bir GitHub reposundan alınan kaynak kod dosyasını incelemek ve bu projeyi anlatacak bir Medium blog yazısı için derinlikli teknik malzeme çıkarmaktır.

Analiz ederken şu kurallara sıkı sıkıya uy:
1. YÜZEYSEL OLMA: "Bu dosya profil sayfasıdır" veya "Bu dosya sohbet bileşenidir" gibi sığ cümleler kurma. Bunun yerine; "Kullanıcının sinema DNA'sını çıkaran, backend'de AI persona verisi henüz oluşmamışsa kullanıcının favori listesinden ağırlıklı tür/yönetmen hesaplayarak istemci tarafında fallback profil üreten bileşendir" gibi kodun içindeki gerçek algoritmayı anlatan analizler yap.
2. MÜHENDİSLİK PRATİKLERİNİ YAKALA: Kod içindeki Optimistic UI Update (iyimser güncelleme), Debouncing, Caching (staleTime), Custom Hooks, Context API, Fallback Mekanizmaları, Polling, Token Interceptor gibi mühendislik detaylarını tespit et ve `engineering_highlights` listesine ekle.
3. KOD KESİTİ (SNIPPET) SEÇİMİ: Asla tüm dosyayı kopyalama! Sadece en öğretici, en temiz veya en akıllıca yazılmış 10 ila 25 satırlık gerçek kod bloğunu seç. Sıradan import satırlarını veya uzun standart JSX/CSS kalabalığını alma; doğrudan algoritmayı veya mimari çözümü gösteren kısmı al.
4. ADİL PUANLAMA (`blog_worthiness_score`):
   - 1-4 Puan: Basit 404 sayfaları, standart login/signup formları, basit listeleme kartları veya sarmalayıcı (wrapper) bileşenler.
   - 5-6 Puan: Standart API çağrıları, temel routing yapıları, orta düzey bileşenler.
   - 7-10 Puan: Karmaşık state yönetimi, AI/sohbet entegrasyonu, akıllı arama/filtreleme, özel algoritmalar, custom hook'lar ve performans optimizasyonları içeren dosyalar.
5. DİL: Kod kesitleri ve evrensel teknik terimler hariç tüm açıklamaları akıcı ve profesyonel bir Türkçe ile yaz."""

# Şemayı prompt'a da gömüyoruz (format= parametresini yok sayan modellere karşı ikinci savunma hattı)
SCHEMA_JSON = json.dumps(FileAnalysisSchema.model_json_schema(), ensure_ascii=False, indent=2)

ANALYZER_SYSTEM_PROMPT_FULL = ANALYZER_SYSTEM_PROMPT + f"""

ÇIKTI FORMATI (ZORUNLU):
Yalnızca aşağıdaki JSON şemasına uyan TEK BİR JSON nesnesi döndür.
- Kök seviyede doğrudan şu alanlar olmalı: responsibility, libraries_and_hooks, engineering_highlights, blog_worthiness_score, best_snippet.
- Başka bir anahtarın (örn. "file_analysis") içine SARMA.
- Şemada olmayan alan ekleme (deep_dive, code_snippet, blog_angle vb. KULLANMA).
- best_snippet bir nesnedir: {{"code": "...", "explanation": "..."}} veya null.

Şema:
{SCHEMA_JSON}
"""


# =====================================================================
# 3. ÇIKTI NORMALİZASYONU
# =====================================================================

ALLOWED_FIELDS = set(FileAnalysisSchema.model_fields.keys())

FIELD_ALIASES = {
    "responsibility": ["deep_dive", "summary", "overview", "description", "analysis"],
    "libraries_and_hooks": ["libraries", "hooks", "dependencies", "libraries_used"],
    "engineering_highlights": ["highlights", "engineering_practices", "patterns"],
    "blog_worthiness_score": ["score", "blog_score", "worthiness_score"],
    "best_snippet": ["code_snippet", "educational_snippet", "snippet"],
}


def _to_str_list(value) -> List[str]:
    """Liste elemanlarını (dict/str karışık gelse bile) düz string listesine çevirir."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    result = []
    for item in value:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            # {"name": "...", "description": "..."} gibi gelirse birleştir
            parts = [str(v) for v in item.values() if v]
            result.append(": ".join(parts))
        else:
            result.append(str(item))
    return result


def _extract_json_text(raw: str) -> str:
    """Model çıktısındaki kod bloğu çitlerini ve fazladan metni temizleyip JSON'u ayıklar."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def normalize_llm_output(data: dict) -> dict:
    """Modelin bozuk/sarılmış çıktısını şemaya yaklaştırır."""
    # 1) Tek anahtarlı sarmalayıcıyı aç ({"file_analysis": {...}})
    while (
        isinstance(data, dict)
        and len(data) == 1
        and isinstance(next(iter(data.values())), dict)
        and next(iter(data.keys())) not in ALLOWED_FIELDS
    ):
        data = next(iter(data.values()))

    if not isinstance(data, dict):
        raise ValueError("Model çıktısı bir JSON nesnesi değil.")

    # 2) Bilinen alternatif alan adlarını eşle
    for target, alts in FIELD_ALIASES.items():
        if target not in data:
            for a in alts:
                if a in data:
                    data[target] = data[a]
                    break

    # 3) Liste alanlarını güvenceye al
    data["libraries_and_hooks"] = _to_str_list(data.get("libraries_and_hooks"))
    data["engineering_highlights"] = _to_str_list(data.get("engineering_highlights"))

    # 4) Snippet string geldiyse nesneye çevir
    snip = data.get("best_snippet")
    if isinstance(snip, str):
        code = re.sub(r"^```[a-zA-Z]*\n|```$", "", snip.strip()).strip()
        explanation = data.get("blog_angle") or data.get("score_justification") or "Öne çıkan kod kesiti."
        data["best_snippet"] = {"code": code, "explanation": str(explanation)}
    elif isinstance(snip, dict):
        code = str(snip.get("code", "")).strip()
        if not code:
            data["best_snippet"] = None
        else:
            data["best_snippet"] = {
                "code": code,
                "explanation": str(snip.get("explanation") or "Öne çıkan kod kesiti."),
            }

    # 5) Puanı güvenceye al
    try:
        data["blog_worthiness_score"] = max(1, min(10, int(data.get("blog_worthiness_score", 5))))
    except (TypeError, ValueError):
        data["blog_worthiness_score"] = 5

    # 6) Şemada olmayan fazlalıkları at
    return {k: v for k, v in data.items() if k in ALLOWED_FIELDS}


# =====================================================================
# 4. LLM İSTEMCİ SINIFI (Ollama + Retry + Normalizasyon)
# =====================================================================

class LLMAnalyzer:
    def __init__(self):
        model_name = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
        ollama_host = (os.getenv("OLLAMA_HOST") or "https://ollama.com").rstrip("/")
        ollama_key = os.getenv("OLLAMA_API_KEY")

        client_kwargs = {"trust_env": False}
        if ollama_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {ollama_key}"}

        logger.info(f"Ollama -> {ollama_host} | model={model_name}")

        # with_structured_output yerine şemayı doğrudan Ollama'ya format olarak veriyoruz
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=ollama_host,
            client_kwargs=client_kwargs,
            format=FileAnalysisSchema.model_json_schema(),
        )

    def analyze_file(self, repo_name: str, file_path: str, file_content: str, max_retries: int = 3) -> dict:
        """Verilen dosyayı LLM ile analiz eder; hatalı/bozuk çıktıda tekrar dener."""
        if len(file_content) > MAX_CONTENT_CHARS:
            file_content = file_content[:MAX_CONTENT_CHARS] + "\n... [dosya kırpıldı]"

        user_prompt = (
            f"Proje Adı: {repo_name}\n"
            f"Dosya Yolu: {file_path}\n\n"
            f"Dosya İçeriği:\n```\n{file_content}\n```\n\n"
            "Bu dosyayı analiz et ve SADECE şemaya uyan JSON döndür."
        )
        messages = [("system", ANALYZER_SYSTEM_PROMPT_FULL), ("user", user_prompt)]

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = self.llm.invoke(messages).content
                data = normalize_llm_output(json.loads(_extract_json_text(raw)))
                return FileAnalysisSchema(**data).model_dump()
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait = attempt * 3
                    print(
                        f"   [UYARI] {type(e).__name__}: {str(e)[:200]} "
                        f"-> {wait} sn sonra tekrar ({attempt}/{max_retries})"
                    )
                    time.sleep(wait)
        raise RuntimeError(f"{max_retries} deneme sonunda başarısız: {str(last_err)[:300]}")


# =====================================================================
# 5. DOSYA FİLTRELEME VE YARDIMCI FONKSİYONLAR
# =====================================================================

SKIP_ANALYSIS_FILENAMES = {
    "eslint.config.js", ".eslintrc.js", ".eslintrc.json", ".prettierrc",
    "vite.config.js", "vite.config.ts", "index.html", "postcss.config.js",
    "tailwind.config.js", "tsconfig.json", "jsconfig.json"
}

META_FILENAMES = {
    "package.json", "requirements.txt", "pyproject.toml", "cargo.toml", "go.mod"
}


def should_analyze_with_llm(file_info: dict) -> tuple[bool, str]:
    path = file_info.get("path", "")
    filename = path.split("/")[-1]
    filename_lower = filename.lower()
    content = file_info.get("content", "").strip()

    if not content or content.startswith("[İndirme/Çözümleme Hatası"):
        return False, "Boş veya hatalı içerik"

    if filename_lower == "readme.md" or filename_lower.endswith(".md"):
        return False, "Dokümantasyon dosyası (Genel bağlama eklendi)"

    if filename_lower in META_FILENAMES:
        return False, "Bağımlılık/Paket dosyası (Genel bağlama eklendi)"

    if filename in SKIP_ANALYSIS_FILENAMES or filename_lower in SKIP_ANALYSIS_FILENAMES:
        return False, "Standart konfigürasyon/boilerplate dosyası"

    if len(content) < 40:
        return False, "Çok kısa dosya (<40 karakter)"

    return True, "Analize uygun"


def select_input_json_file() -> str:
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]

    candidates = sorted(glob.glob("*_code_contents.json"))
    if len(candidates) == 1:
        print(f"Otomatik bulundu: '{candidates[0]}'")
        user_choice = input(f"'{candidates[0]}' analiz edilsin mi? (Enter = Evet / farklı dosya adı yazın): ").strip()
        return user_choice if user_choice else candidates[0]
    elif len(candidates) > 1:
        print("\nDizinde bulunan repo JSON dosyaları:")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. {c}")
        choice = input("Seçiminiz (numara veya dosya adı): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        return choice

    return input("Analiz edilecek JSON dosyasının adını girin (örn: Cinewave_React_Website_code_contents.json): ").strip()


# =====================================================================
# 6. ANA ÇALIŞTIRMA AKIŞI
# =====================================================================

def main():
    print("=" * 65)
    print(" AŞAMA 1: GİTHUB REPO MİMARİ ANALİZÖRÜ (CODE ANALYZER AGENT)")
    print("=" * 65)

    input_file = select_input_json_file()
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"'{input_file}' bulunamadı! Lütfen önce try.py scriptini çalıştırarak repo JSON dosyasını üretin."
        )

    with open(input_file, "r", encoding="utf-8") as f:
        repo_data = json.load(f)

    repo_name = repo_data.get("repository", "Bilinmeyen_Repo")
    files = repo_data.get("files", [])
    directory_structure = repo_data.get("directory_structure") or [f["path"] for f in files]

    dependency_manifests = {}
    markdown_docs = {}
    analyzable_files = []
    skipped_log = []

    for f_info in files:
        path = f_info["path"]
        fname_lower = path.split("/")[-1].lower()
        content = f_info.get("content", "")

        if fname_lower in META_FILENAMES:
            try:
                dependency_manifests[path] = json.loads(content) if fname_lower.endswith(".json") else content
            except Exception:
                dependency_manifests[path] = content
            skipped_log.append(f"  [META] {path}")
            continue

        if fname_lower.endswith(".md"):
            markdown_docs[path] = content
            skipped_log.append(f"  [DOKÜMAN] {path}")
            continue

        ok, reason = should_analyze_with_llm(f_info)
        if ok:
            analyzable_files.append(f_info)
        else:
            skipped_log.append(f"  [ATLANDI] {path} -> {reason}")

    print(f"\nProje             : {repo_name}")
    print(f"Toplam Dosya       : {len(files)}")
    print(f"Analiz Edilecek    : {len(analyzable_files)} kod dosyası")
    print(f"Meta / Atlanan     : {len(skipped_log)} dosya")

    if skipped_log:
        print("\nDoğrudan Bağlama Eklenen / Atlanan Dosyalar:")
        for log_line in skipped_log:
            print(log_line)

    print("\n" + "-" * 65)
    analyzer = LLMAnalyzer()
    print("-" * 65 + "\n")

    analyzed_results = []
    failed_files = []

    start_time = time.time()

    for idx, f_info in enumerate(analyzable_files, 1):
        path = f_info["path"]
        size_kb = f_info.get("size", len(f_info.get("content", ""))) / 1024
        print(f"[{idx}/{len(analyzable_files)}] İnceleniyor: {path} ({size_kb:.1f} KB)...")

        try:
            analysis = analyzer.analyze_file(
                repo_name=repo_name,
                file_path=path,
                file_content=f_info["content"]
            )
            analyzed_results.append({
                "path": path,
                "size": f_info.get("size", 0),
                **analysis
            })
            score = analysis.get("blog_worthiness_score", 0)
            highlights_preview = ", ".join(analysis.get("engineering_highlights", [])[:2])
            print(f"   ✓ Puan: {score}/10 | Öne Çıkan: {highlights_preview[:150] or 'Standart yapı'}")

        except Exception as e:
            print(f"   ✗ [HATA] {path} analiz edilemedi: {str(e)[:300]}")
            failed_files.append({"path": path, "error": str(e)[:500]})

        time.sleep(0.4)

    analyzed_results.sort(key=lambda x: x.get("blog_worthiness_score", 0), reverse=True)

    top_architectural_files = [
        {
            "path": item["path"],
            "score": item["blog_worthiness_score"],
            "responsibility": item["responsibility"],
            "engineering_highlights": item["engineering_highlights"]
        }
        for item in analyzed_results
        if item.get("blog_worthiness_score", 0) >= 7
    ]

    all_engineering_highlights = []
    for item in analyzed_results:
        for h in item.get("engineering_highlights", []):
            if h not in all_engineering_highlights:
                all_engineering_highlights.append(h)

    output_data = {
        "repository": repo_name,
        "description": repo_data.get("description"),
        "stargazers_count": repo_data.get("stargazers_count", 0),
        "languages": repo_data.get("languages", []),
        "directory_structure": directory_structure,
        "dependency_manifests": dependency_manifests,
        "markdown_docs": markdown_docs,
        "project_wide_highlights": all_engineering_highlights,
        "top_architectural_files": top_architectural_files,
        "analyzed_files_count": len(analyzed_results),
        "failed_files": failed_files,
        "file_analyses": analyzed_results
    }

    base_name = input_file.replace("_code_contents.json", "").replace(".json", "")
    output_filename = f"{base_name}_analysis.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print(f" AŞAMA 1 TAMAMLANDI! ({elapsed:.1f} saniye)")
    print(f" Çıktı Dosyası : {output_filename}")
    print(f" Başarılı       : {len(analyzed_results)}/{len(analyzable_files)} dosya")
    if failed_files:
        print(f" Başarısız      : {len(failed_files)} dosya (detaylar çıktı JSON'unda 'failed_files')")
    print("-" * 65)
    print(" Blog Yazısında Öne Çıkarılacak En Kritik Dosyalar (Top 5):")
    for top_item in analyzed_results[:5]:
        print(f"  ★ [{top_item['blog_worthiness_score']}/10] {top_item['path']}")
        if top_item.get("engineering_highlights"):
            print(f"       ↳ {', '.join(top_item['engineering_highlights'][:3])[:200]}")
    print("=" * 65)


if __name__ == "__main__":
    main()