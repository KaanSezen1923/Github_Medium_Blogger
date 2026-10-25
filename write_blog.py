"""
AŞAMA 3: BLOG YAZARI (WRITER AGENT)

Girdi : plan_blog.py çıktısı olan *_outline.json
        (+ analyze_repo.py çıktısı *_analysis.json
         + isteğe bağlı: try.py çıktısı *_code_contents.json -> kod kesitlerini doğrulamak için)
Çıktı : *_draft.md          (Medium'a yapıştırılacak/import edilecek taslak)
        *_draft_meta.json   (başlık, etiketler, uyarılar, istatistikler)

Önemli tasarım kararı:
    Kod kesitlerini LLM YAZMAZ. Model sadece metni yazar ve kesitin yerine bir yer tutucu
    ([[KOD_KESITI]]) bırakır. Gerçek kod, analiz dosyasından alınır, kaynak dosyada
    doğrulanır ve Python tarafından metne gömülür. Böylece uydurma/bozulmuş kod çıkmaz.

Kullanım:
    python write_blog.py                       # dizindeki *_outline.json dosyalarından seçtirir
    python write_blog.py Repo_outline.json     # doğrudan dosya verilir

.env (isteğe bağlı):
    WRITER_MODEL=gpt-oss:120b-cloud   # yoksa OLLAMA_MODEL, o da yoksa gemma4:31b-cloud
"""

import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from logger import logger

load_dotenv()

SNIPPET_TOKEN = "[[KOD_KESITI]]"
INTRO_WORDS = 180
CONCLUSION_WORDS = 160
SNIPPET_MATCH_THRESHOLD = 0.8   # kesitteki satırların kaynak dosyada bulunma oranı

LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "jsx", ".ts": "typescript", ".tsx": "tsx",
    ".java": "java", ".kt": "kotlin", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".cs": "csharp", ".cpp": "cpp", ".c": "c", ".h": "c", ".swift": "swift", ".vue": "vue",
    ".html": "html", ".css": "css", ".json": "json", ".yml": "yaml", ".yaml": "yaml",
    ".sql": "sql", ".xml": "xml",
}


# =====================================================================
# 1. SİSTEM PROMPTU
# =====================================================================

WRITER_SYSTEM_PROMPT = """Sen deneyimli bir Medium yazarı ve kıdemli yazılım mühendisisin.
Sana verilen blog planının TEK BİR bölümünü (veya giriş/sonuç kısmını) yazacaksın.

Kurallara sıkı sıkıya uy:
1. SADECE GERÇEĞE DAYAN: Yalnızca sana verilen dosya analizlerindeki bulguları kullan. Analizde olmayan bir özelliği, kütüphaneyi, performans rakamını veya mimari kararı uydurma. Emin olmadığın bir şeyi yazma.
2. KOD YAZMA: Cevabında KESİNLİKLE kod bloğu (``` ile) kullanma. Gerekirse satır içi kod (`useEffect`, `apiFetch` gibi) kullanabilirsin. Kod kesiti verildiyse onu kendin yazma, kopyalama; sadece yer tutucuyu koy (aşağıya bak).
3. KOD KESİTİ YER TUTUCUSU: Bölüm için bir kod kesiti verilmişse, kesitin görünmesini istediğin yere TEK BİR SATIRDA, tam olarak şunu yaz: [[KOD_KESITI]]
   Yer tutucudan ÖNCE kesitin ne yapacağını anlatan bir giriş cümlesi, SONRA ise kodun neden böyle yazıldığını açıklayan bir paragraf yaz. Kod kesiti verilmediyse yer tutucuyu HİÇ kullanma.
4. BAŞLIK YAZMA: Bölüm başlığını sen yazma, sistem ekleyecek. Doğrudan paragraflarla başla. Alt başlık gerekiyorsa en fazla bir tane ### kullan.
5. ÜSLUP: Verilen hedef kitleye ve tona uy. Dosya adlarını sıralayan bir "rapor" gibi değil, bir mühendisin deneyimini anlattığı akıcı bir yazı gibi yaz. "Bu dosya şunu yapar" kalıbını tekrar tekrar kullanma. Doğal geçişler kur; önceki bölümü tekrar etme.
6. UZUNLUK: İstenen hedef kelime sayısına yakın kal (±%25).
7. DİL: Akıcı ve profesyonel Türkçe. Evrensel teknik terimler İngilizce kalabilir.
8. ÇIKTI: Sadece yazının metnini (Markdown) döndür. "İşte bölüm:" gibi giriş cümleleri, açıklamalar veya notlar ekleme."""


# =====================================================================
# 2. YARDIMCI FONKSİYONLAR (dosya yükleme, temizlik, doğrulama)
# =====================================================================

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def select_outline_file() -> str:
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]

    candidates = sorted(glob.glob("*_outline.json"))
    if len(candidates) == 1:
        print(f"Otomatik bulundu: '{candidates[0]}'")
        choice = input("Bu dosya kullanılsın mı? (Enter = Evet / farklı dosya adı yazın): ").strip()
        return choice or candidates[0]
    elif len(candidates) > 1:
        print("\nDizinde bulunan outline dosyaları:")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. {c}")
        choice = input("Seçiminiz (numara veya dosya adı): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            return candidates[int(choice) - 1]
        return choice

    return input("Outline JSON dosyasının adını girin: ").strip()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_code_contents(outline_file: str, analysis_file: str) -> Dict[str, str]:
    """Kod kesitlerini doğrulamak için ham dosya içeriklerini yükler (yoksa boş döner)."""
    candidates = []
    for name in (analysis_file, outline_file):
        if name:
            base = re.sub(r"_(analysis|outline)\.json$", "", name)
            candidates.append(f"{base}_code_contents.json")
    for c in candidates:
        if os.path.exists(c):
            try:
                data = load_json(c)
                return {f["path"]: f.get("content", "") for f in data.get("files", [])}
            except Exception:
                pass
    return {}


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def verify_snippet(code: str, source: Optional[str]) -> Tuple[bool, str]:
    """
    Kesitin kaynak dosyada gerçekten bulunup bulunmadığını kontrol eder.
    Dönüş: (kabul_edildi_mi, açıklama)
    """
    if source is None:
        return True, "kaynak dosya içeriği yok, doğrulanamadı"

    code_lines = [_norm(l) for l in code.splitlines() if _norm(l)]
    if not code_lines:
        return False, "kesit boş"

    src_norm = _norm(source)
    if _norm(code) in src_norm:
        return True, "birebir eşleşti"

    src_lines = {_norm(l) for l in source.splitlines() if _norm(l)}
    matched = sum(1 for l in code_lines if l in src_lines)
    ratio = matched / len(code_lines)
    if ratio >= SNIPPET_MATCH_THRESHOLD:
        return True, f"satırların %{ratio * 100:.0f}'i kaynakta bulundu"
    return False, f"satırların sadece %{ratio * 100:.0f}'i kaynakta bulundu (model kodu değiştirmiş olabilir)"


def detect_language(path: str) -> str:
    return LANG_BY_EXT.get(os.path.splitext(path)[1].lower(), "")


def clean_model_text(text: str, heading: Optional[str] = None) -> Tuple[str, List[str]]:
    """Model çıktısını temizler: think etiketleri, kendi koyduğu başlık, kod blokları."""
    notes = []
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Baştaki "İşte bölüm:" gibi giriş satırı veya başlık
    lines = text.splitlines()
    while lines and (not lines[0].strip() or re.match(r"^#{1,2}\s+", lines[0])):
        if lines[0].strip():
            notes.append("modelin eklediği başlık silindi")
        lines.pop(0)
    text = "\n".join(lines).strip()

    # Model kendi kod bloğu yazdıysa sil (kodlar yalnızca analizden gelmeli)
    fence_count = len(re.findall(r"```.*?```", text, flags=re.DOTALL))
    if fence_count:
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        notes.append(f"modelin yazdığı {fence_count} kod bloğu silindi")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, notes


def inject_snippet(text: str, snippet_md: Optional[str]) -> Tuple[str, List[str]]:
    """Yer tutucuyu gerçek kod kesitiyle değiştirir; yer tutucu yoksa/fazlaysa düzeltir."""
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

    # Yer tutucunun kendi satırında durmasını garanti et
    text = re.sub(r"[ \t]*" + re.escape(SNIPPET_TOKEN) + r"[ \t]*", f"\n\n{SNIPPET_TOKEN}\n\n", text, count=1)
    text = text.replace(SNIPPET_TOKEN, snippet_md, 1)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, notes


# =====================================================================
# 3. YAZAR SINIFI
# =====================================================================

class BlogWriter:
    def __init__(self, outline_data: dict, analysis: dict, code_contents: Dict[str, str]):
        self.outline_data = outline_data
        self.outline = outline_data["outline"]
        self.analysis = analysis
        self.code_contents = code_contents
        self.audience = outline_data.get("audience", "yazılım geliştiriciler")
        self.tone = outline_data.get("tone", "öğretici ama samimi")
        self.by_path = {f["path"]: f for f in analysis.get("file_analyses", [])}
        self.warnings: List[str] = []

        model_name = os.getenv("WRITER_MODEL") or os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
        ollama_host = (os.getenv("OLLAMA_HOST") or "https://ollama.com").rstrip("/")
        ollama_key = os.getenv("OLLAMA_API_KEY")

        client_kwargs = {"trust_env": False}
        if ollama_key:
            client_kwargs["headers"] = {"Authorization": f"Bearer {ollama_key}"}

        logger.info(f"Ollama -> {ollama_host} | writer model={model_name}")
        self.model_name = model_name

        # Düz metin üretiyoruz: format= verilmiyor
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.6,
            base_url=ollama_host,
            client_kwargs=client_kwargs,
        )

    # ---------- Bağlam üreticiler ----------

    def _blog_header(self) -> str:
        o = self.outline
        return (
            f"BLOG BAŞLIĞI: {o['chosen_title']}\n"
            f"ANA AÇI: {o['core_angle']}\n"
            f"HEDEF KİTLE: {self.audience}\n"
            f"TON: {self.tone}\n"
            f"PROJE: {self.analysis.get('repository')} ({', '.join(self.analysis.get('languages', []))})"
        )

    def _roadmap(self) -> str:
        return "\n".join(f"  {i}. {s['heading']}" for i, s in enumerate(self.outline["sections"], 1))

    def _file_context(self, paths: List[str]) -> str:
        blocks = []
        for p in paths:
            f = self.by_path.get(p)
            if not f:
                continue
            blocks.append(
                f"- Dosya: {p}\n"
                f"  Sorumluluk: {f.get('responsibility', '')}\n"
                f"  Kütüphane/Hook: {', '.join(f.get('libraries_and_hooks', [])[:10])}\n"
                f"  Mühendislik detayları:\n    - " + "\n    - ".join(f.get("engineering_highlights", [])[:6])
            )
        return "\n".join(blocks) or "(kaynak dosya bilgisi yok)"

    def prepare_snippet(self, section: dict) -> Tuple[Optional[dict], Optional[str]]:
        """
        Bölüm için kod kesitini hazırlar.
        Dönüş: (snippet_bilgisi | None, gömülecek_markdown | None)
        """
        path = section.get("use_snippet_from")
        if not path:
            return None, None

        analysis_entry = self.by_path.get(path)
        snippet = (analysis_entry or {}).get("best_snippet")
        if not snippet or not snippet.get("code", "").strip():
            self.warnings.append(f"'{section['heading']}': {path} için kod kesiti bulunamadı, kesitsiz yazılacak")
            return None, None

        code = snippet["code"].strip("\n")
        code = re.sub(r"^```[a-zA-Z]*\n|\n?```$", "", code).strip("\n")

        ok, detail = verify_snippet(code, self.code_contents.get(path))
        if not ok:
            self.warnings.append(f"'{section['heading']}': {path} kesiti doğrulanamadı -> {detail}. Kesit kullanılmadı.")
            return None, None
        if "doğrulanamadı" in detail:
            self.warnings.append(f"'{section['heading']}': {path} -> {detail}")

        lang = detect_language(path)
        md = f"```{lang}\n{code}\n```\n*Kaynak: `{path}`*"
        return {"path": path, "code": code, "explanation": snippet.get("explanation", "")}, md

    # ---------- LLM çağrısı ----------

    def _generate(self, user_prompt: str, heading: Optional[str], max_retries: int = 3, min_words: int = 40) -> str:
        messages = [("system", WRITER_SYSTEM_PROMPT), ("user", user_prompt)]
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                raw = self.llm.invoke(messages).content
                text, notes = clean_model_text(raw, heading)
                for n in notes:
                    self.warnings.append(f"'{heading or 'metin'}': {n}")
                if word_count(text) < min_words:
                    raise ValueError(f"Çıktı çok kısa ({word_count(text)} kelime)")
                return text
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait = attempt * 3
                    print(f"   [UYARI] {type(e).__name__}: {str(e)[:200]} -> {wait} sn sonra tekrar ({attempt}/{max_retries})")
                    time.sleep(wait)
        raise RuntimeError(f"{max_retries} deneme sonunda başarısız: {str(last_err)[:300]}")

    # ---------- Bölüm yazıcılar ----------

    def write_intro(self, feedback: Optional[str] = None) -> str:
        o = self.outline
        prompt = (
            f"{self._blog_header()}\n\n"
            f"YAZILACAK KISIM: GİRİŞ (başlıksız, doğrudan paragraflarla başla)\n"
            f"Hedef kelime sayısı: ~{INTRO_WORDS}\n"
            f"Girişin açılış fikri: {o['hook']}\n\n"
            f"Yazının ilerleyişi (okuyucuya merak uyandıracak şekilde, liste yapmadan kısaca ima et):\n{self._roadmap()}\n\n"
            "Giriş okuyucuyu bir problemle/merakla yakalamalı, projeyi tanıtmalı ve yazıdan ne kazanacağını söylemeli. "
            "Kod kesiti yok, yer tutucu kullanma."
        )
        if feedback:
            prompt += f"\n\nEDİTÖR İSTEĞİ: {feedback}"
        text = self._generate(prompt, "Giriş")
        text, notes = inject_snippet(text, None)
        return text

    def write_section(self, idx: int, previous_parts: List[Tuple[str, str]], feedback: Optional[str] = None) -> Tuple[str, Optional[dict]]:
        sec = self.outline["sections"][idx]
        snippet_info, snippet_md = self.prepare_snippet(sec)

        prev_summary = "\n".join(f"  - {h}" for h, _ in previous_parts) or "  (bu ilk bölüm)"
        last_tail = ""
        if previous_parts:
            last_text = re.sub(r"```.*?```", "", previous_parts[-1][1], flags=re.DOTALL).strip()
            last_tail = last_text.split("\n\n")[-1][:500]

        paths = list(sec.get("source_files", []))
        if snippet_info and snippet_info["path"] not in paths:
            paths.append(snippet_info["path"])

        prompt = (
            f"{self._blog_header()}\n\n"
            f"ŞU ANA KADAR YAZILAN BÖLÜMLER:\n{prev_summary}\n"
            f"ÖNCEKİ BÖLÜMÜN SON PARAGRAFI (akış için, tekrar etme):\n{last_tail or '(yok)'}\n\n"
            f"--- ŞİMDİ YAZILACAK BÖLÜM ---\n"
            f"Bölüm başlığı (sistem ekleyecek, sen yazma): {sec['heading']}\n"
            f"Bölümün amacı: {sec['goal']}\n"
            f"Anlatılması gereken noktalar:\n" + "\n".join(f"  - {kp}" for kp in sec["key_points"]) + "\n"
            f"Hedef kelime sayısı: ~{sec['estimated_words']}\n\n"
            f"KAYNAK DOSYA ANALİZLERİ (tek doğru kaynağın bunlar):\n{self._file_context(paths)}\n"
        )

        if snippet_info:
            prompt += (
                f"\nKOD KESİTİ: VAR. Dosya: {snippet_info['path']}\n"
                f"Kesitin açıklaması: {snippet_info['explanation']}\n"
                f"Aşağıdaki kod yalnızca KENDİ ANLAMAN içindir. Cevabına KOPYALAMA; sadece {SNIPPET_TOKEN} yer tutucusunu koy:\n"
                f"<kod>\n{snippet_info['code']}\n</kod>\n"
            )
        else:
            prompt += "\nKOD KESİTİ: YOK. Yer tutucu kullanma.\n"

        if feedback:
            prompt += f"\nEDİTÖR İSTEĞİ: {feedback}\n"

        text = self._generate(prompt, sec["heading"])
        text, notes = inject_snippet(text, snippet_md)
        for n in notes:
            self.warnings.append(f"'{sec['heading']}': {n}")

        wc = word_count(re.sub(r"```.*?```", "", text, flags=re.DOTALL))
        target = sec["estimated_words"]
        if wc < target * 0.5 or wc > target * 2:
            self.warnings.append(f"'{sec['heading']}': uzunluk hedeften çok farklı ({wc} kelime, hedef ~{target})")

        return text, snippet_info

    def write_conclusion(self, feedback: Optional[str] = None) -> str:
        o = self.outline
        prompt = (
            f"{self._blog_header()}\n\n"
            f"YAZILACAK KISIM: SONUÇ (başlıksız, doğrudan paragraflarla başla)\n"
            f"Hedef kelime sayısı: ~{CONCLUSION_WORDS}\n"
            f"Sonuç fikri: {o['conclusion_idea']}\n\n"
            f"Yazıda anlatılan bölümler:\n{self._roadmap()}\n\n"
            "Yeni teknik iddia ekleme; yazıdan çıkarılan dersleri topla, okuyucuya bir sonraki adımı öner. "
            "Kod kesiti yok, yer tutucu kullanma."
        )
        if feedback:
            prompt += f"\n\nEDİTÖR İSTEĞİ: {feedback}"
        return self._generate(prompt, "Sonuç")


# =====================================================================
# 4. BİRLEŞTİRME VE ÇIKTI
# =====================================================================

def assemble_markdown(outline: dict, repository: str, parts: Dict[str, str]) -> str:
    out = [f"# {outline['chosen_title']}", "", f"*{outline['subtitle']}*", "", parts["intro"], ""]
    for i, sec in enumerate(outline["sections"]):
        out += [f"## {sec['heading']}", "", parts[f"sec_{i}"], ""]
    out += ["## Sonuç", "", parts["conclusion"], ""]
    if repository:
        out += ["---", "", f"Projenin kaynak kodu: [github.com/{repository}](https://github.com/{repository})", ""]
    return "\n".join(out)


def final_checks(markdown: str, outline: dict) -> List[str]:
    issues = []
    if SNIPPET_TOKEN in markdown:
        issues.append("Metinde kullanılmamış yer tutucu kaldı!")
    if markdown.count("```") % 2 != 0:
        issues.append("Kod bloğu çitleri dengesiz (``` sayısı tek).")
    if re.search(r"\[(?:Bölüm|Section|\.\.\.)\]", markdown):
        issues.append("Şablon artığı gibi görünen köşeli parantezli metin var.")
    return issues


def print_summary(outline: dict, parts: Dict[str, str]):
    print("\n" + "-" * 65)
    print(f" {'Giriş':<45} {word_count(parts['intro']):>5} kelime")
    for i, sec in enumerate(outline["sections"], 1):
        text = re.sub(r"```.*?```", "", parts[f"sec_{i - 1}"], flags=re.DOTALL)
        has_code = "```" in parts[f"sec_{i - 1}"]
        print(f" {i}. {sec['heading'][:42]:<42} {word_count(text):>5} kelime {'[kod]' if has_code else ''}")
    print(f" {'Sonuç':<45} {word_count(parts['conclusion']):>5} kelime")
    print("-" * 65)


# =====================================================================
# 5. ANA AKIŞ
# =====================================================================

def main():
    print("=" * 65)
    print(" AŞAMA 3: BLOG YAZARI (WRITER AGENT)")
    print("=" * 65)

    outline_file = select_outline_file()
    if not os.path.exists(outline_file):
        raise FileNotFoundError(f"'{outline_file}' bulunamadı! Önce plan_blog.py çalıştırın.")

    outline_data = load_json(outline_file)
    if not outline_data.get("outline", {}).get("sections"):
        raise ValueError("Outline dosyasında bölüm yok. plan_blog.py çıktısını kontrol edin.")

    analysis_file = outline_data.get("source_analysis_file", "")
    if not analysis_file or not os.path.exists(analysis_file):
        guess = outline_file.replace("_outline.json", "_analysis.json")
        if os.path.exists(guess):
            analysis_file = guess
        else:
            raise FileNotFoundError(f"Analiz dosyası bulunamadı ('{analysis_file}'). Aynı dizinde olduğundan emin olun.")

    analysis = load_json(analysis_file)
    code_contents = load_code_contents(outline_file, analysis_file)

    outline = outline_data["outline"]
    print(f"\nBaşlık       : {outline['chosen_title']}")
    print(f"Bölüm sayısı : {len(outline['sections'])}")
    print(f"Analiz dosyası: {analysis_file}")
    if code_contents:
        print(f"Kesit doğrulama: AÇIK ({len(code_contents)} kaynak dosya yüklendi)")
    else:
        print("Kesit doğrulama: KAPALI (*_code_contents.json bulunamadı, kesitler kaynakla karşılaştırılamayacak)")

    writer = BlogWriter(outline_data, analysis, code_contents)
    print(f"Model        : {writer.model_name}\n")

    base_name = re.sub(r"_outline\.json$", "", outline_file)
    cache_file = f"{base_name}_draft_cache.json"
    parts: Dict[str, str] = {}
    all_warnings: List[str] = []

    # Yarım kalan çalışmayı devam ettir
    if os.path.exists(cache_file):
        resume = input(f"Önceki yarım kalmış taslak bulundu ('{cache_file}'). Devam edilsin mi? (Enter = Evet / n = Sıfırdan): ").strip().lower()
        if resume != "n":
            parts = load_json(cache_file).get("parts", {})
            print(f"  {len(parts)} parça önbellekten yüklendi.\n")

    def save_cache():
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"parts": parts}, f, ensure_ascii=False, indent=2)

    start_time = time.time()
    sections = outline["sections"]

    try:
        if "intro" not in parts:
            print("[Giriş] yazılıyor...")
            parts["intro"] = writer.write_intro()
            save_cache()

        for i, sec in enumerate(sections):
            key = f"sec_{i}"
            if key in parts:
                print(f"[{i + 1}/{len(sections)}] (önbellekten) {sec['heading']}")
                continue
            print(f"[{i + 1}/{len(sections)}] Yazılıyor: {sec['heading']}")
            previous = [(s["heading"], parts[f"sec_{j}"]) for j, s in enumerate(sections[:i]) if f"sec_{j}" in parts]
            text, snip = writer.write_section(i, previous)
            parts[key] = text
            save_cache()
            print(f"   ✓ {word_count(text)} kelime" + (f" | kod: {snip['path']}" if snip else ""))
            time.sleep(0.4)

        if "conclusion" not in parts:
            print("[Sonuç] yazılıyor...")
            parts["conclusion"] = writer.write_conclusion()
            save_cache()

    except KeyboardInterrupt:
        print("\nİşlem durduruldu. İlerleme kaydedildi; tekrar çalıştırınca kaldığı yerden devam eder.")
        save_cache()
        return

    # ---- İnteraktif düzenleme döngüsü ----
    while True:
        print_summary(outline, parts)
        if writer.warnings:
            print("[UYARILAR]")
            for w in writer.warnings:
                print(f"  ! {w}")
            all_warnings.extend(writer.warnings)
            writer.warnings = []   # ekranda tekrar göstermemek için; meta dosyaya all_warnings ile yazılır
        print("\nBir bölümü yeniden yazdırmak ister misiniz?")
        print("  Bölüm no (1-N) | i = giriş | s = sonuç | Enter = bitir")
        choice = input("> ").strip().lower()
        if choice == "":
            break

        feedback = input("Geri bildirim (isteğe bağlı, Enter = yok): ").strip() or None
        try:
            if choice == "i":
                parts["intro"] = writer.write_intro(feedback)
            elif choice == "s":
                parts["conclusion"] = writer.write_conclusion(feedback)
            elif choice.isdigit() and 1 <= int(choice) <= len(sections):
                idx = int(choice) - 1
                previous = [(s["heading"], parts[f"sec_{j}"]) for j, s in enumerate(sections[:idx])]
                parts[f"sec_{idx}"], _ = writer.write_section(idx, previous, feedback)
            else:
                print("Geçersiz seçim.")
                continue
            save_cache()
        except Exception as e:
            print(f"   ✗ [HATA] {str(e)[:300]}")

    # ---- Birleştir, kontrol et, kaydet ----
    markdown = assemble_markdown(outline, analysis.get("repository", ""), parts)
    issues = final_checks(markdown, outline)

    draft_file = f"{base_name}_draft.md"
    with open(draft_file, "w", encoding="utf-8") as f:
        f.write(markdown)

    code_blocks = len(re.findall(r"```", markdown)) // 2
    total_words = word_count(re.sub(r"```.*?```", "", markdown, flags=re.DOTALL))

    meta = {
        "repository": analysis.get("repository"),
        "title": outline["chosen_title"],
        "subtitle": outline["subtitle"],
        "tags": outline.get("tags", []),
        "word_count": total_words,
        "estimated_read_minutes": max(1, round(total_words / 200)),
        "code_blocks": code_blocks,
        "final_check_issues": issues,
        "writer_warnings": all_warnings,
        "writer_model": writer.model_name,
        "draft_file": draft_file,
    }
    meta_file = f"{base_name}_draft_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if os.path.exists(cache_file):
        os.remove(cache_file)

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f" AŞAMA 3 TAMAMLANDI! ({elapsed:.1f} saniye)")
    print(f" Taslak        : {draft_file}")
    print(f" Meta          : {meta_file}")
    print(f" Kelime        : ~{total_words} (~{meta['estimated_read_minutes']} dk okuma)")
    print(f" Kod bloğu     : {code_blocks}")
    print(f" Etiketler     : {', '.join(meta['tags'])}")
    if issues:
        print(" Son kontrol sorunları:")
        for i in issues:
            print(f"   ! {i}")
    print("=" * 65)


if __name__ == "__main__":
    main()