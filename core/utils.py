"""
ÇEKİRDEK YARDIMCI FONKSİYONLAR (core/utils.py)

Kod kesiti doğrulama, dil algılama, metin temizleme, Markdown birleştirme ve
LLM bağlam hazırlama gibi saf (pure) veri işleme fonksiyonlarını barındırır.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from schemas.blog_schemas import BlogOutline

# =====================================================================
# SABİTLER
# =====================================================================

SNIPPET_TOKEN = "[[KOD_KESITI]]"
SNIPPET_MATCH_THRESHOLD = 0.8
README_MAX_CHARS = 3000
MAX_FILES_IN_CONTEXT = 20
MIN_SCORE_IN_CONTEXT = 5

LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "jsx", ".ts": "typescript", ".tsx": "tsx",
    ".java": "java", ".kt": "kotlin", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".cs": "csharp", ".cpp": "cpp", ".c": "c", ".h": "c", ".swift": "swift", ".vue": "vue",
    ".html": "html", ".css": "css", ".json": "json", ".yml": "yaml", ".yaml": "yaml",
    ".sql": "sql", ".xml": "xml",
}

SKIP_ANALYSIS_FILENAMES = {
    "eslint.config.js", ".eslintrc.js", ".eslintrc.json", ".prettierrc",
    "vite.config.js", "vite.config.ts", "index.html", "postcss.config.js",
    "tailwind.config.js", "tsconfig.json", "jsconfig.json"
}

META_FILENAMES = {
    "package.json", "requirements.txt", "pyproject.toml", "cargo.toml", "go.mod"
}


# =====================================================================
# 1. DOSYA FİLTRELEME VE VERİ HAZIRLIK YARDIMCILARI
# =====================================================================

def should_analyze_with_llm(file_info: dict) -> Tuple[bool, str]:
    """Dosyanın LLM ile analiz edilmeye uygun olup olmadığını kontrol eder."""
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


def partition_repo_files(files: List[dict]) -> Tuple[Dict[str, Any], Dict[str, str], List[dict], List[str]]:
    """
    Repodaki dosyaları bağımlılık manifestoları, markdown dokümanları ve
    LLM ile analiz edilecek kod dosyaları olarak ayrıştırır.
    """
    dependency_manifests: Dict[str, Any] = {}
    markdown_docs: Dict[str, str] = {}
    analyzable: List[dict] = []
    skipped_log: List[str] = []

    for f_info in files:
        path = f_info["path"]
        fname = path.split("/")[-1].lower()
        content = f_info.get("content", "")

        if fname in META_FILENAMES:
            try:
                dependency_manifests[path] = json.loads(content) if fname.endswith(".json") else content
            except Exception:
                dependency_manifests[path] = content
            skipped_log.append(f"[META] {path}")
            continue

        if fname.endswith(".md"):
            markdown_docs[path] = content
            skipped_log.append(f"[DOKÜMAN] {path}")
            continue

        ok, reason = should_analyze_with_llm(f_info)
        if ok:
            analyzable.append(f_info)
        else:
            skipped_log.append(f"[ATLANDI] {path} -> {reason}")

    return dependency_manifests, markdown_docs, analyzable, skipped_log


def summarize_manifests(manifests: dict) -> str:
    """package.json / requirements.txt gibi dosyalardan bağımlılık adlarını özetler."""
    lines = []
    for path, content in manifests.items():
        if isinstance(content, dict):
            deps = list((content.get("dependencies") or {}).keys())
            dev = list((content.get("devDependencies") or {}).keys())
            if deps:
                lines.append(f"{path} -> dependencies: {', '.join(deps)}")
            if dev:
                lines.append(f"{path} -> devDependencies: {', '.join(dev)}")
        elif isinstance(content, str):
            pkgs = [
                l.split("==")[0].split(">=")[0].strip()
                for l in content.splitlines()
                if l.strip() and not l.startswith("#")
            ]
            lines.append(f"{path} -> {', '.join(pkgs[:30])}")
    return "\n".join(lines) or "(yok)"


def select_context_files(
    analysis: dict,
    min_score: int = MIN_SCORE_IN_CONTEXT,
    focus_area: Optional[str] = None,
) -> List[dict]:
    """Analiz sonuçlarından puanı ve (varsa) odak alanı eşleşmesine göre en değerli dosyaları seçer."""
    files = [
        f for f in analysis.get("file_analyses", [])
        if f.get("blog_worthiness_score", 0) >= min_score
    ]
    if not files:
        files = list(analysis.get("file_analyses", []))

    if focus_area:
        keywords = [w.lower() for w in re.findall(r"\w+", focus_area) if len(w) > 2]

        def _sort_key(item: dict):
            blob = (
                f"{item.get('architectural_domain', '')} "
                f"{item.get('responsibility', '')} "
                f"{' '.join(item.get('engineering_highlights', []))}"
            ).lower()
            matches = sum(1 for kw in keywords if kw in blob)
            return (matches > 0, item.get("blog_worthiness_score", 0), matches)

        files.sort(key=_sort_key, reverse=True)
    else:
        files.sort(key=lambda x: x.get("blog_worthiness_score", 0), reverse=True)

    return files[:MAX_FILES_IN_CONTEXT]


def build_planner_context(analysis: dict, context_files: Optional[List[dict]] = None) -> str:
    """Analiz verilerinden PlannerAgent için metin bağlamını oluşturur."""
    if context_files is None:
        context_files = select_context_files(analysis)

    readme = ""
    for path, content in (analysis.get("markdown_docs") or {}).items():
        if path.lower().endswith("readme.md"):
            readme = content[:README_MAX_CHARS]
            break

    domain_map: Dict[str, List[str]] = {}
    for f in context_files:
        dom = f.get("architectural_domain", "Genel Mimari")
        domain_map.setdefault(dom, []).append(f["path"])
    domain_summary = "\n".join(
        f"  * {dom} ({len(paths)} dosya): {', '.join(paths[:4])}"
        for dom, paths in domain_map.items()
    )

    file_blocks = []
    for f in context_files:
        snippet_flag = "var" if f.get("best_snippet") else "yok"
        file_blocks.append(
            f"- yol: {f['path']}\n"
            f"  alan: {f.get('architectural_domain', 'Genel Mimari')} | puan: {f.get('blog_worthiness_score')}/10 | snippet: {snippet_flag}\n"
            f"  sorumluluk: {f.get('responsibility', '')}\n"
            f"  kütüphaneler: {', '.join(f.get('libraries_and_hooks', [])[:8])}\n"
            f"  öne çıkanlar: {' | '.join(f.get('engineering_highlights', [])[:4])}"
        )

    structure = "\n".join(analysis.get("directory_structure", [])[:80])

    return (
        f"PROJE: {analysis.get('repository')}\n"
        f"AÇIKLAMA: {analysis.get('description') or '(yok)'}\n"
        f"DİLLER: {', '.join(analysis.get('languages', []))}\n"
        f"YILDIZ: {analysis.get('stargazers_count', 0)}\n\n"
        f"BAĞIMLILIKLAR:\n{summarize_manifests(analysis.get('dependency_manifests', {}))}\n\n"
        f"DİZİN YAPISI:\n{structure}\n\n"
        f"README (kısaltılmış):\n{readme or '(yok)'}\n\n"
        f"TESPİT EDİLEN MİMARİ ALANLAR (DOMAINS):\n{domain_summary or '(yok)'}\n\n"
        f"EN DEĞERLİ DOSYALAR (puana göre sıralı):\n" + "\n".join(file_blocks)
    )


# =====================================================================
# 2. METİN İŞLEME, KOD KESİTİ DOĞRULAMA VE DİL ALGILAMA
# =====================================================================

def word_count(text: str) -> int:
    """Metindeki toplam kelime sayısını hesaplar."""
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def detect_language(path: str) -> str:
    """Dosya uzantısına göre Markdown kod bloğu dil etiketini döndürür."""
    return LANG_BY_EXT.get(os.path.splitext(path)[1].lower(), "")


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def verify_snippet(code: str, source: Optional[str]) -> Tuple[bool, str]:
    """
    Seçilen kod kesitinin kaynak dosyada gerçekten bulunup bulunmadığını doğrular.
    Dönüş: (gecerli_mi, aciklama)
    """
    if source is None:
        return True, "kaynak dosya içeriği yok, doğrulanamadı"

    code_lines = [_norm_line(l) for l in code.splitlines() if _norm_line(l)]
    if not code_lines:
        return False, "kesit boş"

    src_norm = _norm_line(source)
    if _norm_line(code) in src_norm:
        return True, "birebir eşleşti"

    src_lines = {_norm_line(l) for l in source.splitlines() if _norm_line(l)}
    matched = sum(1 for l in code_lines if l in src_lines)
    ratio = matched / len(code_lines)
    if ratio >= SNIPPET_MATCH_THRESHOLD:
        return True, f"satırların %{ratio * 100:.0f}'i kaynakta bulundu"
    return False, f"satırların sadece %{ratio * 100:.0f}'i kaynakta bulundu (model kodu değiştirmiş olabilir)"


def clean_model_text(text: str, heading: Optional[str] = None) -> Tuple[str, List[str]]:
    """Model çıktısındaki <think> bloklarını, fazladan başlıkları ve izinsiz kod bloklarını temizler."""
    notes = []
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    lines = text.splitlines()
    while lines and (not lines[0].strip() or re.match(r"^#{1,2}\s+", lines[0])):
        if lines[0].strip():
            notes.append("modelin eklediği başlık silindi")
        lines.pop(0)
    text = "\n".join(lines).strip()

    fence_count = len(re.findall(r"```.*?```", text, flags=re.DOTALL))
    if fence_count:
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        notes.append(f"modelin yazdığı {fence_count} kod bloğu silindi")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, notes


def inject_snippet(text: str, snippet_md: Optional[str]) -> Tuple[str, List[str]]:
    """Metin içindeki [[KOD_KESITI]] yer tutucusunu gerçek kod bloğu ile değiştirir."""
    notes = []
    count = text.count(SNIPPET_TOKEN)

    if snippet_md is None:
        if count:
            text = text.replace(SNIPPET_TOKEN, "")
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            notes.append("kesit yokken yer tutucu kullanılmış, kaldırıldı")
        return text, notes

    if count == 0:
        paragraphs = text.split("\n\n")
        paragraphs.insert(min(1, len(paragraphs)), snippet_md)
        notes.append("yer tutucu yoktu, kesit ilk paragraftan sonra eklendi")
        return "\n\n".join(paragraphs), notes

    if count > 1:
        first = text.index(SNIPPET_TOKEN) + len(SNIPPET_TOKEN)
        text = text[:first] + text[first:].replace(SNIPPET_TOKEN, "")
        notes.append("birden fazla yer tutucu vardı, ilki kullanıldı")

    text = re.sub(r"[ \t]*" + re.escape(SNIPPET_TOKEN) + r"[ \t]*", f"\n\n{SNIPPET_TOKEN}\n\n", text, count=1)
    text = text.replace(SNIPPET_TOKEN, snippet_md, 1)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, notes


def assemble_markdown(outline: Union[BlogOutline, dict], repository: str, parts: Dict[str, str]) -> str:
    """Giriş, bölümler ve sonuç parçalarını nihai Markdown belgesine dönüştürür."""
    o = outline.model_dump() if isinstance(outline, BlogOutline) else outline
    out = [f"# {o['chosen_title']}", "", f"*{o['subtitle']}*", "", parts["intro"], ""]
    for i, sec in enumerate(o["sections"]):
        out += [f"## {sec['heading']}", "", parts[f"sec_{i}"], ""]
    out += ["## Sonuç", "", parts["conclusion"], ""]
    if repository:
        out += ["---", "", f"Projenin kaynak kodu: [github.com/{repository}](https://github.com/{repository})", ""]
    return "\n".join(out)


def sanitize_for_json(data: Any) -> Any:
    """
    Veri yapısı (dict, list, str vb.) içerisindeki PostgreSQL JSONB alanlarını 
    patlatabilecek \u0000 (null byte) karakterlerini recursive olarak temizler.
    """
    if isinstance(data, str):
        return data.replace("\x00", "").replace("\u0000", "")
    elif isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(sanitize_for_json(item) for item in data)
    return data


def final_checks(markdown: str) -> List[str]:
    """Üretilen Markdown taslağında kalan yer tutucu veya dengesiz kod çitlerini kontrol eder."""
    issues = []
    if SNIPPET_TOKEN in markdown:
        issues.append("Metinde kullanılmamış yer tutucu kaldı!")
    if markdown.count("```") % 2 != 0:
        issues.append("Kod bloğu çitleri dengesiz (``` sayısı tek).")
    if re.search(r"\[(?:Bölüm|Section|\.\.\.)\]", markdown):
        issues.append("Şablon artığı gibi görünen köşeli parantezli metin var.")
    return issues