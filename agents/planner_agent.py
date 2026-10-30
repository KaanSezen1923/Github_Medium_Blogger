"""
BLOG PLANLAYICI AJAN (agents/planner_agent.py)
"""

import json
import os
from typing import List, Optional, Union

from agents.base_agent import BaseAgent
from core.utils import build_planner_context, select_context_files
from schemas.blog_schemas import BlogOutline, validate_outline

DEFAULT_AUDIENCE = os.getenv("BLOG_AUDIENCE", "orta seviye yazılım geliştiriciler")
DEFAULT_TONE = os.getenv("BLOG_TONE", "öğretici ama samimi, birinci tekil şahıs (projeyi yapan kişi anlatıyor)")

OUTLINE_SYSTEM_PROMPT = """Sen deneyimli bir Teknik İçerik Editörü ve Medium yazarısın.
Görevin, bir GitHub reposunun mimari analizinden yola çıkarak ilgi çekici bir Medium blog yazısının İSKELETİNİ (outline) tasarlamaktır. Yazının kendisini yazmayacaksın, sadece planını çıkaracaksın.

Kurallara sıkı sıkıya uy:
1. ODAK ALANI BELİRLE (ÇORBA YAPMA): Her şeyden biraz bahseden yüzeysel bir proje tanıtımı yapma!
   - Önce `recommended_focus_areas` listesine bu repodan çıkabilecek en güçlü 3 spesifik mühendislik konusunu yaz.
   - Eğer kullanıcı özel bir odak alanı (`İSTENEN ODAK ALANI`) belirttiyse `selected_focus_area` olarak onu kullan; belirtmediyse önerdiğin 3 alandan en derinlikli olanını `selected_focus_area` olarak seç ve TÜM YAZI PLANINI bu odak alanı etrafında kurgula.
2. HİKÂYE KUR, DOSYA GEZMESİ YAPMA: Yazı "dosya dosya anlatım" olmamalı. Seçilen odak alanı (`selected_focus_area`) ve ortak bir tez (`core_angle`) etrafında, okuyucuyu bir problemden çözüme götüren akış kur. Birbiriyle ilgili dosyaları aynı bölümde birleştir.
3. SADECE GERÇEĞE DAYAN: Sadece sana verilen dosya analizlerindeki bulguları kullan. Analizde olmayan bir özelliği, kütüphaneyi veya mimariyi uydurma.
4. DOSYA YOLLARI: `source_files` ve `use_snippet_from` alanlarında SADECE sana verilen dosya listesindeki yolları, birebir aynı yazımla kullan.
5. KOD KESİTİ: `use_snippet_from` yalnızca listede "snippet: var" olarak işaretli dosyalar için doldurulabilir. Her kod kesiti en fazla bir bölümde kullanılsın.
6. AKIŞ: Giriş (hook) -> projenin bağlamı/problemi -> seçilen odak alanındaki en değerli 3-5 teknik konu -> öğrenilen dersler.
7. BAŞLIK: Somut, teknik bir vaat içersin ve seçilen odak alanını yansıtsın. "X Projemi Nasıl Yaptım" gibi jenerik kalıplardan kaçın.
8. DİL: Tüm içeriği akıcı ve profesyonel Türkçe ile yaz. Evrensel teknik terimler İngilizce kalabilir."""

SCHEMA_JSON = json.dumps(BlogOutline.model_json_schema(), ensure_ascii=False, indent=2)

OUTLINE_SYSTEM_PROMPT_FULL = OUTLINE_SYSTEM_PROMPT + f"""

ÇIKTI FORMATI (ZORUNLU):
Yalnızca aşağıdaki JSON şemasına uyan TEK BİR JSON nesnesi döndür.
- Kök seviyede doğrudan şemadaki alanlar olmalı. Başka bir anahtarın içine SARMA.

Şema:
{SCHEMA_JSON}
"""


class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            model_env_var="OUTLINE_MODEL",
            default_model="deepseek-v3.1:671b-cloud",
            temperature=0.4,
            format_schema=BlogOutline.model_json_schema(),
            agent_role="planner",
        )
        self.last_warnings: List[str] = []

    def run(
        self,
        analysis: dict,
        audience: str = DEFAULT_AUDIENCE,
        tone: str = DEFAULT_TONE,
        focus_area: Optional[str] = None,
        previous_outline: Optional[Union[BlogOutline, dict]] = None,
        feedback: Optional[str] = None,
        max_retries: int = 3,
    ) -> BlogOutline:
        context_files = select_context_files(analysis, focus_area=focus_area)
        if not context_files:
            raise ValueError("Analiz verisinde planlama için kullanılabilecek dosya bulunamadı.")

        context = build_planner_context(analysis, context_files)

        focus_directive = (
            f"İSTENEN ODAK ALANI (ZORUNLU): {focus_area}\n"
            f"(Yazının tamamını ve bölümlerini özellikle '{focus_area}' konusunu merkeze alarak kurgula.)\n"
            if focus_area
            else "İSTENEN ODAK ALANI: Otomatik Keşif (Projedeki en özgün 3 teknik odak alanını `recommended_focus_areas` içinde belirle, en güçlüsünü `selected_focus_area` olarak seç ve planı onun üzerine kur.)\n"
        )

        user_prompt = (
            f"HEDEF KİTLE: {audience}\n"
            f"TON: {tone}\n"
            f"{focus_directive}\n"
            f"{context}\n\n"
            "Bu bilgilere dayanarak blog yazısının iskeletini çıkar ve SADECE şemaya uyan JSON döndür."
        )

        if previous_outline and feedback:
            prev_dict = (
                previous_outline.model_dump()
                if isinstance(previous_outline, BlogOutline)
                else previous_outline
            )
            user_prompt += (
                "\n\n--- GERİ BİLDİRİM ---\n"
                f"Önceki taslak:\n{json.dumps(prev_dict, ensure_ascii=False)}\n\n"
                f"Editörün isteği: {feedback}\n"
                "Önceki taslağı bu isteğe göre revize et."
            )

        messages = [("system", OUTLINE_SYSTEM_PROMPT_FULL), ("user", user_prompt)]

        def _process_output(raw: str) -> BlogOutline:
            outline_obj = BlogOutline.from_raw_response(raw)
            outline_dict = outline_obj.model_dump()
            if len(outline_dict["sections"]) < 2:
                raise ValueError("Outline en az 2 bölüm içermeli.")
            self.last_warnings = validate_outline(outline_dict, analysis)
            return BlogOutline(**outline_dict)

        return self.invoke_with_retry(
            messages=messages,
            response_processor=_process_output,
            max_retries=max_retries,
        )