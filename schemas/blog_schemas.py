"""
MERKEZİ VERİ ŞEMALARI VE DOĞRULAMA MODÜLÜ (schemas/blog_schemas.py)

Tüm ajanlar arasındaki veri transferini standartlaştıran Pydantic şemalarını,
JSON ayıklama ve çıktı normalizasyonu fonksiyonlarını içerir.
"""

import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field


# =====================================================================
# 1. ORTAK YARDIMCI FONKSİYONLAR
# =====================================================================

def extract_json_text(raw: str) -> str:
    """Model çıktısındaki kod bloğu çitlerini ve fazladan metni temizleyip JSON'u ayıklar."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


def to_str_list(value) -> List[str]:
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
            parts = [str(v) for v in item.values() if v]
            result.append(": ".join(parts))
        else:
            result.append(str(item))
    return result


def apply_aliases(data: dict, aliases: dict) -> dict:
    """Verilen sözlükteki alternatif anahtar isimlerini standart şema anahtarlarına eşler."""
    for target, alts in aliases.items():
        if target not in data:
            for a in alts:
                if a in data:
                    data[target] = data[a]
                    break
    return data


# =====================================================================
# 2. AŞAMA 1 ŞEMALARI (DOSYA ANALİZİ)
# =====================================================================

class CodeSnippet(BaseModel):
    code: str = Field(
        description="Dosyadan seçilen, Medium blog yazısında paylaşmaya değer en çarpıcı 10-25 satırlık gerçek kod kesiti. Sadece iş mantığını (business logic) içermeli, gereksiz import veya uzun HTML/JSX yığınlarını içermemelidir."
    )
    explanation: str = Field(
        description="Bu kod kesitinin neden önemli olduğu, hangi problemi çözdüğü ve ne yaptığının 1-2 cümlelik teknik açıklaması."
    )


class FileAnalysisSchema(BaseModel):
    architectural_domain: str = Field(
        default="Genel Mimari",
        description="Dosyanın ait olduğu ana teknik/mimari alan (Örn: 'State Yönetimi ve Önbellekleme', 'AI/LLM Ajan Orkestrasyonu', 'Veri Doğrulama ve Şema Yönetimi', 'API ve Ağ Katmanı', 'UI/UX Performansı')."
    )
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

    @classmethod
    def from_raw_response(cls, raw_text: str) -> "FileAnalysisSchema":
        """Ham LLM yanıtını temizler, normalize eder ve doğrulanmış şema nesnesi döndürür."""
        json_str = extract_json_text(raw_text)
        data = normalize_llm_output(json.loads(json_str))
        return cls(**data)


ALLOWED_ANALYSIS_FIELDS = set(FileAnalysisSchema.model_fields.keys())

ANALYSIS_FIELD_ALIASES = {
    "architectural_domain": ["domain", "area", "category", "module_type", "focus_area"],
    "responsibility": ["deep_dive", "summary", "overview", "description", "analysis"],
    "libraries_and_hooks": ["libraries", "hooks", "dependencies", "libraries_used"],
    "engineering_highlights": ["highlights", "engineering_practices", "patterns"],
    "blog_worthiness_score": ["score", "blog_score", "worthiness_score"],
    "best_snippet": ["code_snippet", "educational_snippet", "snippet"],
}


def normalize_llm_output(data: dict) -> dict:
    """Modelin bozuk/sarılmış dosya analizi çıktısını FileAnalysisSchema yapısına yaklaştırır."""
    while (
        isinstance(data, dict)
        and len(data) == 1
        and isinstance(next(iter(data.values())), dict)
        and next(iter(data.keys())) not in ALLOWED_ANALYSIS_FIELDS
    ):
        data = next(iter(data.values()))

    if not isinstance(data, dict):
        raise ValueError("Model çıktısı bir JSON nesnesi değil.")

    data = apply_aliases(data, ANALYSIS_FIELD_ALIASES)

    if not data.get("architectural_domain") or not isinstance(data.get("architectural_domain"), str):
        data["architectural_domain"] = "Genel Mimari"

    data["libraries_and_hooks"] = to_str_list(data.get("libraries_and_hooks"))
    data["engineering_highlights"] = to_str_list(data.get("engineering_highlights"))

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

    try:
        data["blog_worthiness_score"] = max(1, min(10, int(data.get("blog_worthiness_score", 5))))
    except (TypeError, ValueError):
        data["blog_worthiness_score"] = 5

    return {k: v for k, v in data.items() if k in ALLOWED_ANALYSIS_FIELDS}


# =====================================================================
# 3. AŞAMA 2 ŞEMALARI (BLOG İSKELETİ / OUTLINE)
# =====================================================================

class OutlineSection(BaseModel):
    heading: str = Field(description="Bölümün Medium'daki H2 başlığı. Merak uyandırıcı ve somut olmalı.")
    goal: str = Field(description="Bu bölümün okuyucuya ne kazandıracağını anlatan 1 cümlelik amaç.")
    key_points: List[str] = Field(
        description="Bu bölümde mutlaka anlatılması gereken 3-5 somut teknik nokta. Analizdeki gerçek bulgulara dayanmalı."
    )
    source_files: List[str] = Field(
        description="Bu bölümün dayandığı dosya yolları. SADECE verilen dosya listesindeki yolları kullan."
    )
    use_snippet_from: Optional[str] = Field(
        default=None,
        description="Bu bölümde kod kesiti gösterilecekse, kesitin alınacağı dosyanın yolu. Kod kesiti yoksa null."
    )
    estimated_words: int = Field(default=250, description="Bölüm için tahmini kelime sayısı (150-500).")


class BlogOutline(BaseModel):
    recommended_focus_areas: List[str] = Field(
        default_factory=list,
        description="Bu projenin kaynak kodlarından çıkabilecek en güçlü 3 farklı teknik blog odak alanı/konusu."
    )
    selected_focus_area: str = Field(
        default="Genel Mimari ve Mühendislik Pratikleri",
        description="recommended_focus_areas içinden bu yazı için seçilen (veya kullanıcının belirttiği) spesifik odak alanı."
    )
    title_options: List[str] = Field(
        description="3-5 alternatif başlık. Tıklanabilir ama clickbait olmayan, somut teknik vaat içeren başlıklar."
    )
    chosen_title: str = Field(description="title_options içinden en güçlü bulunan başlık.")
    subtitle: str = Field(description="Medium alt başlığı (tek cümle).")
    core_angle: str = Field(
        description="Yazının ana tezi: bu yazı neden var, okuyucu sonunda neyi öğrenmiş olacak? 2-3 cümle."
    )
    hook: str = Field(description="Giriş paragrafının nasıl açılacağını anlatan 2-3 cümlelik fikir (gerçek yazı değil, yön).")
    sections: List[OutlineSection] = Field(description="Sıralı bölüm listesi (4-7 bölüm). Giriş ve sonuç dahil değil.")
    conclusion_idea: str = Field(description="Sonuç bölümünün vurgusu: çıkarılan dersler, sonraki adımlar, çağrı.")
    tags: List[str] = Field(description="En fazla 5 Medium etiketi (örn: React, Web Development).")
    estimated_read_minutes: int = Field(default=8, description="Tahmini okuma süresi (dakika).")

    @classmethod
    def from_raw_response(cls, raw_text: str) -> "BlogOutline":
        """Ham LLM yanıtını temizler, normalize eder ve doğrulanmış BlogOutline nesnesi döndürür."""
        json_str = extract_json_text(raw_text)
        data = normalize_outline(json.loads(json_str))
        return cls(**data)


OUTLINE_FIELDS = set(BlogOutline.model_fields.keys())
SECTION_FIELDS = set(OutlineSection.model_fields.keys())

OUTLINE_ALIASES = {
    "recommended_focus_areas": ["focus_options", "focus_areas", "suggested_focuses"],
    "selected_focus_area": ["focus_area", "chosen_focus", "focus"],
    "title_options": ["titles", "title_suggestions", "alternative_titles"],
    "chosen_title": ["title", "selected_title", "best_title"],
    "core_angle": ["angle", "thesis", "main_angle"],
    "sections": ["outline", "structure", "body_sections"],
    "conclusion_idea": ["conclusion", "closing"],
    "tags": ["labels", "medium_tags"],
}

SECTION_ALIASES = {
    "heading": ["title", "section_title", "name"],
    "goal": ["purpose", "objective"],
    "key_points": ["points", "bullets", "topics"],
    "source_files": ["files", "sources", "related_files"],
    "use_snippet_from": ["snippet_file", "snippet_from", "snippet"],
    "estimated_words": ["words", "word_count"],
}


def normalize_outline(data: dict) -> dict:
    """Modelin ham outline çıktısını BlogOutline şemasına uygun hale getirir."""
    while (
        isinstance(data, dict)
        and len(data) == 1
        and isinstance(next(iter(data.values())), dict)
        and next(iter(data.keys())) not in OUTLINE_FIELDS
    ):
        data = next(iter(data.values()))

    if not isinstance(data, dict):
        raise ValueError("Model çıktısı bir JSON nesnesi değil.")

    data = apply_aliases(data, OUTLINE_ALIASES)

    data["recommended_focus_areas"] = to_str_list(data.get("recommended_focus_areas"))[:5]
    if not data.get("selected_focus_area"):
        data["selected_focus_area"] = (
            data["recommended_focus_areas"][0]
            if data["recommended_focus_areas"]
            else "Genel Mimari ve Mühendislik Pratikleri"
        )
    elif data["selected_focus_area"] not in data["recommended_focus_areas"]:
        data["recommended_focus_areas"].insert(0, data["selected_focus_area"])

    data["title_options"] = to_str_list(data.get("title_options"))
    data["tags"] = to_str_list(data.get("tags"))[:5]

    if not data.get("chosen_title") and data["title_options"]:
        data["chosen_title"] = data["title_options"][0]
    if data.get("chosen_title") and data["chosen_title"] not in data["title_options"]:
        data["title_options"].insert(0, data["chosen_title"])

    sections = data.get("sections") or []
    clean_sections = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        sec = apply_aliases(sec, SECTION_ALIASES)
        sec["key_points"] = to_str_list(sec.get("key_points"))
        sec["source_files"] = to_str_list(sec.get("source_files"))
        snip = sec.get("use_snippet_from")
        if not isinstance(snip, str) or not snip.strip() or snip.strip().lower() in ("null", "none", "yok"):
            sec["use_snippet_from"] = None
        try:
            sec["estimated_words"] = max(100, min(800, int(sec.get("estimated_words", 250))))
        except (TypeError, ValueError):
            sec["estimated_words"] = 250
        sec.setdefault("goal", "")
        clean_sections.append({k: v for k, v in sec.items() if k in SECTION_FIELDS})
    data["sections"] = clean_sections

    try:
        data["estimated_read_minutes"] = max(1, int(data.get("estimated_read_minutes", 8)))
    except (TypeError, ValueError):
        data["estimated_read_minutes"] = 8

    return {k: v for k, v in data.items() if k in OUTLINE_FIELDS}


def validate_outline(outline: dict, analysis: dict) -> List[str]:
    """
    Dosya yollarını gerçek analizle karşılaştırır, uydurma yolları temizler,
    kod kesiti kuralını uygular. Yapılan düzeltmeleri uyarı listesi olarak döndürür.
    """
    warnings = []
    by_path = {f["path"]: f for f in analysis.get("file_analyses", [])}
    by_name = {}
    for p in by_path:
        by_name.setdefault(p.split("/")[-1], []).append(p)

    def resolve(path: str) -> Optional[str]:
        if path in by_path:
            return path
        candidates = by_name.get(path.split("/")[-1], [])
        return candidates[0] if len(candidates) == 1 else None

    used_snippets = set()

    for sec in outline["sections"]:
        fixed = []
        for p in sec["source_files"]:
            real = resolve(p)
            if real:
                if real not in fixed:
                    fixed.append(real)
            else:
                warnings.append(f"'{sec['heading']}': var olmayan dosya çıkarıldı -> {p}")
        sec["source_files"] = fixed

        snip = sec.get("use_snippet_from")
        if snip:
            real = resolve(snip)
            if not real:
                warnings.append(f"'{sec['heading']}': var olmayan snippet dosyası -> {snip}")
                sec["use_snippet_from"] = None
            elif not by_path[real].get("best_snippet"):
                warnings.append(f"'{sec['heading']}': {real} için kod kesiti yok, snippet kaldırıldı")
                sec["use_snippet_from"] = None
            elif real in used_snippets:
                warnings.append(f"'{sec['heading']}': {real} kesiti başka bölümde kullanılıyor, kaldırıldı")
                sec["use_snippet_from"] = None
            else:
                sec["use_snippet_from"] = real
                used_snippets.add(real)
                if real not in sec["source_files"]:
                    sec["source_files"].append(real)

        if not sec["source_files"]:
            warnings.append(f"'{sec['heading']}': hiçbir geçerli kaynak dosya kalmadı")

    return warnings