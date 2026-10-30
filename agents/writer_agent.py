"""
BLOG YAZARI AJANI (agents/writer_agent.py)
"""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from agents.base_agent import BaseAgent
from core.utils import (
    SNIPPET_TOKEN,
    assemble_markdown,
    clean_model_text,
    detect_language,
    inject_snippet,
    verify_snippet,
    word_count,
)
from schemas.blog_schemas import BlogOutline

INTRO_WORDS = 180
CONCLUSION_WORDS = 160

WRITER_SYSTEM_PROMPT = """Sen Medium'da yazılım mimarisi üzerine derinlikli makaleler yazan kıdemli bir Yazılım Mühendisisin (Senior Software Engineer).
Sana verilen blog planının TEK BİR bölümünü (veya giriş/sonuç kısmını) yazacaksın.

Kurallara sıkı sıkıya uy:
1. "NE" DEĞİL "NEDEN VE NASIL" ANLAT: Dosyanın ne yaptığını yüzeysel özetleme. Hangi mühendislik problemini çözdüğünü, neden bu mimariyi seçtiğini (trade-off), uç durumları (edge-case) somut fonksiyon, değişken ve parametre isimleriyle (`RunnableConfig`, `set_config`, `CROSS JOIN LATERAL` vb.) anlat.
2. TEKRARDAN KAÇIN: Önceki bölümlerde işlenen teknik detayları bu bölümde KESİNLİKLE tekrar açıklama; gerekirse "önceki adımda kurduğumuz yapı" diyerek doğrudan yeni konuya geç.
3. KOD YAZMA, YER TUTUCU KULLAN: Cevabında KESİNLİKLE ``` ile kod bloğu yazma. Bölüm için kod kesiti verilmişse, görünmesini istediğin yere TEK BİR SATIRDA tam olarak şunu yaz: [[KOD_KESITI]]
   - Yer tutucudan ÖNCE: Sana verilen 'Kesitin açıklaması' cümlesini ASLA kopyalama ve "Aşağıdaki kod şunu yapar" gibi mekanik cümleler kurma. Problemi ortaya koyup okuyucuyu koda hazırlayan doğal bir geçiş yap.
   - Yer tutucudan SONRA: Kodu satır satır tercüme etme. Kodun içindeki en akıllıca mühendislik detayını, güvenlik kontrolünü veya performans optimizasyonunu analiz et.
   - Kod kesiti verilmediyse [[KOD_KESITI]] yer tutucusunu HİÇ kullanma.
4. GÖRSEL RİTİM VE FORMAT: Sadece uzun paragraf yığınları yazma. Önemli teknik kavramları **kalın** yaz, satır içi kod (`apiFetch`, `ciftlik_id`) kullan ve çok adımlı mantıkları anlatırken kısa madde işaretlerinden yararlan. Bölüm başlığını (##) sen yazma, sistem ekleyecek.
5. SADECE GERÇEĞE DAYAN: Analizde ve kaynak kodda olmayan hiçbir kütüphaneyi, özelliği veya metriği uydurma. Varsa somut sayıları (timeout süreleri, limitler, yüzdeler) mutlaka kullan.
6. DİL VE UZUNLUK: Akıcı, samimi ama teknik ağırlığı yüksek birinci tekil şahıs ("kurguladım", "tercih ettim") Türkçe kullan. Hedef kelime sayısına sadık kal (±%20). Sadece Markdown metni döndür."""


class WriterAgent(BaseAgent):
    def __init__(
        self,
        outline_data: Union[BlogOutline, dict],
        analysis: dict,
        code_contents: Optional[Dict[str, str]] = None,
        audience: Optional[str] = None,
        tone: Optional[str] = None,
    ):
        super().__init__(
            model_env_var="WRITER_MODEL",
            default_model="gemma4:31b-cloud",
            temperature=0.6,
            format_schema=None,
            agent_role="writer",
        )

        if isinstance(outline_data, BlogOutline):
            self.outline = outline_data.model_dump()
            self.audience = audience or "yazılım geliştiriciler"
            self.tone = tone or "öğretici ama samimi"
        elif isinstance(outline_data, dict) and "outline" in outline_data:
            raw_o = outline_data["outline"]
            self.outline = raw_o.model_dump() if isinstance(raw_o, BlogOutline) else raw_o
            self.audience = audience or outline_data.get("audience", "yazılım geliştiriciler")
            self.tone = tone or outline_data.get("tone", "öğretici ama samimi")
        else:
            self.outline = outline_data
            self.audience = audience or "yazılım geliştiriciler"
            self.tone = tone or "öğretici ama samimi"

        self.analysis = analysis
        self.code_contents = code_contents or {}
        self.by_path = {f["path"]: f for f in analysis.get("file_analyses", [])}
        self.warnings: List[str] = []

    def _blog_header(self) -> str:
        o = self.outline
        focus = o.get("selected_focus_area", "Genel Mimari ve Mühendislik Pratikleri")
        return (
            f"BLOG BAŞLIĞI: {o['chosen_title']}\n"
            f"SEÇİLEN ODAK ALANI: {focus}\n"
            f"ANA AÇI (TEZ): {o['core_angle']}\n"
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
            raw_code = self.code_contents.get(p, "")
            code_preview = raw_code[:3500] + ("\n... [kırpıldı]" if len(raw_code) > 3500 else "")
            blocks.append(
                f"=== DOSYA: {p} ===\n"
                f"Mimari Alan: {f.get('architectural_domain', 'Genel Mimari')}\n"
                f"Sorumluluk: {f.get('responsibility', '')}\n"
                f"Kütüphane/Hook: {', '.join(f.get('libraries_and_hooks', [])[:10])}\n"
                f"Mühendislik Detayları:\n  - " + "\n  - ".join(f.get("engineering_highlights", [])[:6]) + "\n"
                f"Gerçek Kod Referansı (Sadece teknik detay/değişken ismi öğrenmek için, cevaba kopyalama):\n"
                f"<kaynak_kod>\n{code_preview}\n</kaynak_kod>"
            )
        return "\n\n".join(blocks) or "(kaynak dosya bilgisi yok)"

    def prepare_snippet(self, section: dict) -> Tuple[Optional[dict], Optional[str]]:
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

        lang = detect_language(path)
        md = f"```{lang}\n{code}\n```\n*Kaynak: `{path}`*"
        return {"path": path, "code": code, "explanation": snippet.get("explanation", "")}, md

    def _generate(self, user_prompt: str, heading: Optional[str], max_retries: int = 3, min_words: int = 40) -> str:
        messages = [("system", WRITER_SYSTEM_PROMPT), ("user", user_prompt)]

        def _process_output(raw: str) -> str:
            text, notes = clean_model_text(raw, heading)
            if word_count(text) < min_words:
                raise ValueError(f"Çıktı çok kısa ({word_count(text)} kelime)")
            for n in notes:
                self.warnings.append(f"'{heading or 'metin'}': {n}")
            return text

        return self.invoke_with_retry(
            messages=messages,
            response_processor=_process_output,
            max_retries=max_retries,
        )

    def write_intro(self, feedback: Optional[str] = None) -> str:
        o = self.outline
        prompt = (
            f"{self._blog_header()}\n\n"
            f"YAZILACAK KISIM: GİRİŞ (başlıksız, doğrudan paragraflarla başla)\n"
            f"Hedef kelime sayısı: ~{INTRO_WORDS}\n"
            f"Girişin açılış fikri: {o['hook']}\n\n"
            f"Yazının ilerleyişi:\n{self._roadmap()}\n\n"
            "Giriş okuyucuyu bir problemle/merakla yakalamalı, seçilen odak alanını tanıtmalı ve yazıdan ne kazanacağını söylemeli."
        )
        if feedback:
            prompt += f"\n\nEDİTÖR İSTEĞİ: {feedback}"
        text = self._generate(prompt, "Giriş")
        text, _ = inject_snippet(text, None)
        return text

    def write_section(
        self,
        idx: int,
        previous_parts: List[Tuple[str, str]],
        feedback: Optional[str] = None,
    ) -> Tuple[str, Optional[dict]]:
        sec = self.outline["sections"][idx]
        snippet_info, snippet_md = self.prepare_snippet(sec)

        prev_summary = "\n".join(f"  - {h}" for h, _ in previous_parts) or "  (bu ilk bölüm)"
        
        paths = list(sec.get("source_files", []))
        if snippet_info and snippet_info["path"] not in paths:
            paths.append(snippet_info["path"])

        prompt = (
            f"{self._blog_header()}\n\n"
            f"ŞU ANA KADAR YAZILAN BÖLÜM BAŞLIKLARI:\n{prev_summary}\n\n"
            f"--- ŞİMDİ YAZILACAK BÖLÜM ---\n"
            f"Bölüm başlığı (sen yazma): {sec['heading']}\n"
            f"Bölümün amacı: {sec['goal']}\n"
            f"Anlatılması gereken noktalar:\n" + "\n".join(f"  - {kp}" for kp in sec["key_points"]) + "\n"
            f"Hedef kelime sayısı: ~{sec['estimated_words']}\n\n"
            f"KAYNAK DOSYA ANALİZLERİ VE KODLARI:\n{self._file_context(paths)}\n"
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

        return text, snippet_info

    def write_conclusion(self, feedback: Optional[str] = None) -> str:
        o = self.outline
        prompt = (
            f"{self._blog_header()}\n\n"
            f"YAZILACAK KISIM: SONUÇ (başlıksız, doğrudan paragraflarla başla)\n"
            f"Hedef kelime sayısı: ~{CONCLUSION_WORDS}\n"
            f"Sonuç fikri: {o['conclusion_idea']}\n\n"
            f"Yazıda anlatılan bölümler:\n{self._roadmap()}\n\n"
            "Yeni teknik iddia ekleme; yazıdan çıkarılan dersleri topla, okuyucuya bir sonraki adımı öner."
        )
        if feedback:
            prompt += f"\n\nEDİTÖR İSTEĞİ: {feedback}"
        return self._generate(prompt, "Sonuç")

    def run(
        self,
        initial_parts: Optional[Dict[str, str]] = None,
        on_part_complete: Optional[Callable[[str, str, Dict[str, str]], None]] = None,
    ) -> Dict[str, Any]:
        parts: Dict[str, str] = dict(initial_parts) if initial_parts else {}
        sections = self.outline["sections"]

        if "intro" not in parts:
            parts["intro"] = self.write_intro()
            if on_part_complete:
                on_part_complete("intro", parts["intro"], parts)

        for i, sec in enumerate(sections):
            key = f"sec_{i}"
            if key not in parts:
                previous = [
                    (s["heading"], parts[f"sec_{j}"])
                    for j, s in enumerate(sections[:i])
                    if f"sec_{j}" in parts
                ]
                text, _ = self.write_section(i, previous)
                parts[key] = text
                if on_part_complete:
                    on_part_complete(key, text, parts)

        if "conclusion" not in parts:
            parts["conclusion"] = self.write_conclusion()
            if on_part_complete:
                on_part_complete("conclusion", parts["conclusion"], parts)

        markdown = assemble_markdown(self.outline, self.analysis.get("repository", ""), parts)
        return {
            "parts": parts,
            "markdown": markdown,
            "warnings": list(self.warnings),
        }