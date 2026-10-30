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
Görevin, bir GitHub reposunun mimari analizinden yola çıkarak ilgi çekici bir Medium blog yazısının İSKELETİNİ (outline) tasarlamaktır.

Kurallara sıkı sıkıya uy:
1. ODAK ALANI BELİRLE: `recommended_focus_areas` listesine bu repodan çıkabilecek en güçlü 3 spesifik mühendislik konusunu yaz. Belirtilmişse `selected_focus_area` olarak onu kullan.
2. BÖLÜMLER ARASI BENZERSİZLİK (TEKRARI ÖNLE): Her bölümde farklı bir mimari katmanı veya problemi ele al. Aynı kütüphaneyi, stratejiyi veya teknik detayı (örn. LLM stratejisi veya arama optimizasyonu) birden fazla bölümün `key_points` listesine kesinlikle yazma.
3. HİKÂYE KUR: Yazı "dosya dosya anlatım" olmamalı. Seçilen odak alanı etrafında okuyucuyu problemden çözüme götüren mantıksal bir akış kur.
4. SADECE GERÇEĞE DAYAN: Sadece sana verilen dosya analizlerindeki bulguları kullan. Uydurma özellik ekleme.
5. DOSYA YOLLARI: `source_files` ve `use_snippet_from` alanlarında SADECE verilen dosya listesindeki yolları birebir kullan.
6. DİL: Tüm içeriği akıcı ve profesyonel Türkçe ile yaz."""

SCHEMA_JSON = json.dumps(BlogOutline.model_json_schema(), ensure_ascii=False, indent=2)

OUTLINE_SYSTEM_PROMPT_FULL = OUTLINE_SYSTEM_PROMPT + f"""

ÇIKTI FORMATI (ZORUNLU):
Yalnızca aşağıdaki JSON şemasına uyan TEK BİR JSON nesnesi döndür.

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
            if focus_area
            else "İSTENEN ODAK ALANI: Otomatik Keşif\n"
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