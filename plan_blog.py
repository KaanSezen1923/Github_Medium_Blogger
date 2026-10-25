"""
AŞAMA 2: BLOG PLANLAYICI (OUTLINE AGENT)

Girdi : analyze_repo.py çıktısı olan *_analysis.json
Çıktı : *_outline.json  (Aşama 3 - Yazar Ajanı bu dosyayı okuyacak)

Kullanım:
    python plan_blog.py                      # dizindeki *_analysis.json dosyalarından seçtirir
    python plan_blog.py Repo_analysis.json   # doğrudan dosya verilir

.env ile varsayılanlar (isteğe bağlı):
    OUTLINE_MODEL=gpt-oss:120b-cloud   # yoksa OLLAMA_MODEL, o da yoksa gemma4:31b-cloud
    BLOG_AUDIENCE=orta seviye frontend geliştiriciler
    BLOG_TONE=öğretici ama samimi
"""

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

load_dotenv()

DEFAULT_AUDIENCE = os.getenv("BLOG_AUDIENCE", "orta seviye yazılım geliştiriciler")
DEFAULT_TONE = os.getenv("BLOG_TONE", "öğretici ama samimi, birinci tekil şahıs (projeyi yapan kişi anlatıyor)")

README_MAX_CHARS = 3000
MAX_FILES_IN_CONTEXT = 20     # LLM'e gösterilecek maksimum dosya analizi
MIN_SCORE_IN_CONTEXT = 5      # Bu puanın altındaki dosyalar bağlama alınmaz


# =====================================================================
# 1. VERİ ŞEMALARI
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


# =====================================================================
# 2. SİSTEM PROMPTU
# =====================================================================

OUTLINE_SYSTEM_PROMPT = """Sen deneyimli bir Teknik İçerik Editörü ve Medium yazarısın.
Görevin, bir GitHub reposunun mimari analizinden yola çıkarak ilgi çekici bir Medium blog yazısının İSKELETİNİ (outline) tasarlamaktır. Yazının kendisini yazmayacaksın, sadece planını çıkaracaksın.

Kurallara sıkı sıkıya uy:
1. HİKÂYE KUR, DOSYA GEZMESİ YAPMA: Yazı "dosya dosya anlatım" olmamalı. Ortak bir tez etrafında, okuyucuyu bir problemden çözüme götüren akış kur (örn. "Bir sinema uygulamasında AI önerisi nasıl istemci tarafında desteklenir?"). Birbiriyle ilgili dosyaları aynı bölümde birleştir.
2. SADECE GERÇEĞE DAYAN: Sadece sana verilen dosya analizlerindeki bulguları kullan. Analizde olmayan bir özelliği, kütüphaneyi veya mimariyi uydurma.
3. DOSYA YOLLARI: `source_files` ve `use_snippet_from` alanlarında SADECE sana verilen dosya listesindeki yolları, birebir aynı yazımla kullan.
4. KOD KESİTİ: `use_snippet_from` yalnızca listede "snippet: var" olarak işaretli dosyalar için doldurulabilir. Her kod kesiti en fazla bir bölümde kullanılsın.
5. AKIŞ: Giriş (hook) -> projenin bağlamı/problemi -> en değerli 3-5 teknik konu (en çok puanlı dosyalardan, kolaydan zora) -> öğrenilen dersler. Sıradan dosyaları (düşük puanlı) ayrı bölüm yapma, gerekirse tek cümleyle geç.
6. BAŞLIK: Somut, teknik bir vaat içersin. "X Projemi Nasıl Yaptım" gibi jenerik kalıplardan kaçın.
7. DİL: Tüm içeriği akıcı ve profesyonel Türkçe ile yaz. Evrensel teknik terimler İngilizce kalabilir.
"""

SCHEMA_JSON = json.dumps(BlogOutline.model_json_schema(), ensure_ascii=False, indent=2)

OUTLINE_SYSTEM_PROMPT_FULL = OUTLINE_SYSTEM_PROMPT + f"""

ÇIKTI FORMATI (ZORUNLU):
Yalnızca aşağıdaki JSON şemasına uyan TEK BİR JSON nesnesi döndür.
- Kök seviyede doğrudan şemadaki alanlar olmalı. Başka bir anahtarın (örn. "outline") içine SARMA.
- Şemada olmayan alan ekleme.

Şema:
{SCHEMA_JSON}
"""


# =====================================================================
# 3. BAĞLAM HAZIRLAMA
# =====================================================================

def summarize_manifests(manifests: dict) -> str:
    """package.json / requirements.txt gibi dosyalardan sadece bağımlılık adlarını çıkarır."""
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
            pkgs = [l.split("==")[0].split(">=")[0].strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
            lines.append(f"{path} -> {', '.join(pkgs[:30])}")
    return "\n".join(lines) or "(yok)"


def select_context_files(analysis: dict) -> List[dict]:
    files = [
        f for f in analysis.get("file_analyses", [])
        if f.get("blog_worthiness_score", 0) >= MIN_SCORE_IN_CONTEXT
    ]
    files.sort(key=lambda x: x.get("blog_worthiness_score", 0), reverse=True)
    return files[:MAX_FILES_IN_CONTEXT]


def build_context(analysis: dict, context_files: List[dict]) -> str:
    readme = ""
    for path, content in (analysis.get("markdown_docs") or {}).items():
        if path.lower().endswith("readme.md"):
            readme = content[:README_MAX_CHARS]
            break

    file_blocks = []
    for f in context_files:
        snippet_flag = "var" if f.get("best_snippet") else "yok"
        file_blocks.append(
            f"- yol: {f['path']}\n"
            f"  puan: {f.get('blog_worthiness_score')}/10 | snippet: {snippet_flag}\n"
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
        f"EN DEĞERLİ DOSYALAR (puana göre sıralı):\n" + "\n".join(file_blocks)
    )


# =====================================================================
# 4. ÇIKTI NORMALİZASYONU VE DOĞRULAMA
# =====================================================================

OUTLINE_FIELDS = set(BlogOutline.model_fields.keys())
SECTION_FIELDS = set(OutlineSection.model_fields.keys())

OUTLINE_ALIASES = {
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


def _apply_aliases(data: dict, aliases: dict) -> dict:
    for target, alts in aliases.items():
        if target not in data:
            for a in alts:
                if a in data:
                    data[target] = data[a]
                    break
    return data


def _to_str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    out = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            out.append(": ".join(str(v) for v in item.values() if v))
        else:
            out.append(str(item))
    return out


def _extract_json_text(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return raw[start:end + 1]
    return raw


def normalize_outline(data: dict) -> dict:
    # Tek anahtarlı sarmalayıcıyı aç
    while (
        isinstance(data, dict)
        and len(data) == 1
        and isinstance(next(iter(data.values())), dict)
        and next(iter(data.keys())) not in OUTLINE_FIELDS
    ):
        data = next(iter(data.values()))

    if not isinstance(data, dict):
        raise ValueError("Model çıktısı bir JSON nesnesi değil.")

    data = _apply_aliases(data, OUTLINE_ALIASES)

    data["title_options"] = _to_str_list(data.get("title_options"))
    data["tags"] = _to_str_list(data.get("tags"))[:5]

    if not data.get("chosen_title") and data["title_options"]:
        data["chosen_title"] = data["title_options"][0]
    if data.get("chosen_title") and data["chosen_title"] not in data["title_options"]:
        data["title_options"].insert(0, data["chosen_title"])

    sections = data.get("sections") or []
    clean_sections = []
    for sec in sections:
        if not isinstance(sec, dict):
            continue
        sec = _apply_aliases(sec, SECTION_ALIASES)
        sec["key_points"] = _to_str_list(sec.get("key_points"))
        sec["source_files"] = _to_str_list(sec.get("source_files"))
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


# =====================================================================
# 5. LLM İSTEMCİSİ
# =====================================================================

class OutlinePlanner:
    def __init__(self):
        model_name = os.getenv("OUTLINE_MODEL") or os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
        ollama_host = (os.getenv("OLLAMA_HOST") or "https://ollama.com").rstrip("/")
        ollama_key = os.getenv("OLLAMA_API_KEY")

        client_kwargs = {"trust_env": False}
        if ollama_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {ollama_key}"}

        logger.info(f"Ollama -> {ollama_host} | outline model={model_name}")
        self.model_name = model_name

        self.llm = ChatOllama(
            model=model_name,
            temperature=0.4,   # Analizden farklı: planlamada biraz yaratıcılık gerekir
            base_url=ollama_host,
            client_kwargs=client_kwargs,
            format=BlogOutline.model_json_schema(),
        )

    def plan(self, context: str, audience: str, tone: str,
             previous_outline: Optional[dict] = None, feedback: Optional[str] = None,
             max_retries: int = 3) -> dict:

        user_prompt = (
            f"HEDEF KİTLE: {audience}\n"
            f"TON: {tone}\n\n"
            f"{context}\n\n"
            "Bu bilgilere dayanarak blog yazısının iskeletini çıkar ve SADECE şemaya uyan JSON döndür."
        )
        if previous_outline and feedback:
            user_prompt += (
                "\n\n--- GERİ BİLDİRİM ---\n"
                f"Önceki taslak:\n{json.dumps(previous_outline, ensure_ascii=False)}\n\n"
                f"Editörün isteği: {feedback}\n"
                "Önceki taslağı bu isteğe göre revize et."
            )

        messages = [("system", OUTLINE_SYSTEM_PROMPT_FULL), ("user", user_prompt)]

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = self.llm.invoke(messages).content
                data = normalize_outline(json.loads(_extract_json_text(raw)))
                outline = BlogOutline(**data).model_dump()
                if len(outline["sections"]) < 2:
                    raise ValueError("Outline en az 2 bölüm içermeli.")
                return outline
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait = attempt * 3
                    print(f"   [UYARI] {type(e).__name__}: {str(e)[:200]} -> {wait} sn sonra tekrar ({attempt}/{max_retries})")
                    time.sleep(wait)
        raise RuntimeError(f"{max_retries} deneme sonunda başarısız: {str(last_err)[:300]}")


# =====================================================================
# 6. YARDIMCI FONKSİYONLAR
# =====================================================================

def select_input_json_file() -> str:
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]

    candidates = sorted(glob.glob("*_analysis.json"))
    if len(candidates) == 1:
        print(f"Otomatik bulundu: '{candidates[0]}'")
        choice = input("Bu dosya kullanılsın mı? (Enter = Evet / farklı dosya adı yazın): ").strip()
        return choice or candidates[0]
    elif len(candidates) > 1:
        print("\nDizinde bulunan analiz dosyaları:")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. {c}")
        choice = input("Seçiminiz (numara veya dosya adı): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        return choice

    return input("Analiz JSON dosyasının adını girin: ").strip()


def print_outline(outline: dict):
    print("\n" + "=" * 65)
    print(f" BAŞLIK  : {outline['chosen_title']}")
    print(f" ALT BAŞLIK: {outline['subtitle']}")
    print(f" OKUMA   : ~{outline['estimated_read_minutes']} dk | ETİKETLER: {', '.join(outline['tags'])}")
    print("-" * 65)
    print(" Alternatif başlıklar:")
    for t in outline["title_options"]:
        print(f"   • {t}")
    print("-" * 65)
    print(f" ANA AÇI : {outline['core_angle']}")
    print(f" GİRİŞ   : {outline['hook']}")
    print("-" * 65)
    for i, sec in enumerate(outline["sections"], 1):
        print(f" {i}. {sec['heading']}  (~{sec['estimated_words']} kelime)")
        print(f"    Amaç    : {sec['goal']}")
        for kp in sec["key_points"]:
            print(f"      - {kp}")
        print(f"    Dosyalar: {', '.join(sec['source_files']) or '-'}")
        if sec.get("use_snippet_from"):
            print(f"    Kod     : {sec['use_snippet_from']}")
    print("-" * 65)
    print(f" SONUÇ   : {outline['conclusion_idea']}")
    print("=" * 65)


def ask_with_default(prompt: str, default: str) -> str:
    value = input(f"{prompt}\n  (Enter = '{default}'): ").strip()
    return value or default


# =====================================================================
# 7. ANA AKIŞ
# =====================================================================

def main():
    print("=" * 65)
    print(" AŞAMA 2: BLOG PLANLAYICI (OUTLINE AGENT)")
    print("=" * 65)

    input_file = select_input_json_file()
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"'{input_file}' bulunamadı! Önce analyze_repo.py çalıştırın.")

    with open(input_file, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    if not analysis.get("file_analyses"):
        raise ValueError("Analiz dosyasında 'file_analyses' boş. analyze_repo.py çıktısını kontrol edin.")

    print(f"\nProje: {analysis.get('repository')}")
    print(f"Analiz edilmiş dosya: {analysis.get('analyzed_files_count')} | Başarısız: {len(analysis.get('failed_files', []))}")

    audience = ask_with_default("\nHedef kitle kim?", DEFAULT_AUDIENCE)
    tone = ask_with_default("Yazının tonu nasıl olsun?", DEFAULT_TONE)

    context_files = select_context_files(analysis)
    if not context_files:
        raise ValueError(
            f"Puanı {MIN_SCORE_IN_CONTEXT} ve üzeri dosya yok. MIN_SCORE_IN_CONTEXT değerini düşürün."
        )
    context = build_context(analysis, context_files)
    print(f"\nPlanlama bağlamına {len(context_files)} dosya alındı (~{len(context)} karakter).")

    planner = OutlinePlanner()
    print(f"Model: {planner.model_name}\n")

    outline = None
    feedback = None
    need_generate = True

    while True:
        if need_generate:
            print("Outline hazırlanıyor...")
            outline = planner.plan(
                context=context, audience=audience, tone=tone,
                previous_outline=outline if feedback else None,
                feedback=feedback,
            )
            warnings = validate_outline(outline, analysis)
            print_outline(outline)
            if warnings:
                print("\n[DOĞRULAMA] Otomatik düzeltmeler:")
                for w in warnings:
                    print(f"  ! {w}")
            need_generate = False

        print("\nSeçenekler:")
        print("  Enter  -> Onayla ve kaydet")
        print("  r      -> Sıfırdan yeniden üret")
        print("  t <n>  -> Başlık seç (örn: t 2)")
        print("  Diğer  -> Yazdığınız metni geri bildirim olarak uygula (örn: 'daha az bölüm, auth kısmını öne al')")
        choice = input("> ").strip()

        if choice == "":
            break
        elif choice.lower() == "r":
            outline, feedback, need_generate = None, None, True
        elif re.fullmatch(r"t\s+\d+", choice.lower()):
            idx = int(choice.split()[1]) - 1
            if 0 <= idx < len(outline["title_options"]):
                outline["chosen_title"] = outline["title_options"][idx]
                print(f"Başlık seçildi: {outline['chosen_title']}")
            else:
                print("Geçersiz numara.")
        else:
            feedback, need_generate = choice, True

    output_data = {
        "repository": analysis.get("repository"),
        "source_analysis_file": input_file,
        "audience": audience,
        "tone": tone,
        "outline": outline,
    }

    base_name = input_file.replace("_analysis.json", "").replace(".json", "")
    output_filename = f"{base_name}_outline.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(" AŞAMA 2 TAMAMLANDI!")
    print(f" Çıktı Dosyası : {output_filename}")
    print(f" Başlık        : {outline['chosen_title']}")
    print(f" Bölüm Sayısı  : {len(outline['sections'])}")
    print("=" * 65)


if __name__ == "__main__":
    main()