"""
MİMARİ ANALİZ AJANI (agents/analyzer_agent.py)
"""

import json
from agents.base_agent import BaseAgent
from schemas.blog_schemas import FileAnalysisSchema

MAX_CONTENT_CHARS = 12000

ANALYZER_SYSTEM_PROMPT = """Sen kıdemli bir Yazılım Mimarı (Senior Software Architect) ve Teknik İçerik Editörüsün.
Görevin, bir GitHub reposundan alınan kaynak kod dosyasını incelemek ve bu projeyi anlatacak bir Medium blog yazısı için derinlikli teknik malzeme çıkarmaktır.

Analiz ederken şu kurallara sıkı sıkıya uy:
1. MİMARİ ALANI BELİRLE (`architectural_domain`): Dosyanın projedeki ana teknik kategorisini net bir başlıkla etiketle (Örn: 'AI/LLM Ajan Orkestrasyonu', 'State Yönetimi ve Önbellekleme', 'Veri Doğrulama ve Şema Yönetimi', 'API ve Ağ Katmanı', 'UI/UX Performansı').
2. YÜZEYSEL OLMA: "Bu dosya profil sayfasıdır" veya "Bu dosya sohbet bileşenidir" gibi sığ cümleler kurma. Bunun yerine; "Kullanıcının sinema DNA'sını çıkaran, backend'de AI persona verisi henüz oluşmamışsa kullanıcının favori listesinden ağırlıklı tür/yönetmen hesaplayarak istemci tarafında fallback profil üreten bileşendir" gibi kodun içindeki gerçek algoritmayı anlatan analizler yap.
3. MÜHENDİSLİK PRATİKLERİNİ YAKALA: Kod içindeki Optimistic UI Update (iyimser güncelleme), Debouncing, Caching (staleTime), Custom Hooks, Context API, Fallback Mekanizmaları, Polling, Token Interceptor gibi mühendislik detaylarını tespit et ve `engineering_highlights` listesine ekle.
4. KOD KESİTİ (SNIPPET) SEÇİMİ: Asla tüm dosyayı kopyalama! Sadece en öğretici, en temiz veya en akıllıca yazılmış 10 ila 25 satırlık gerçek kod bloğunu seç. Sıradan import satırlarını veya uzun standart JSX/CSS kalabalığını alma; doğrudan algoritmayı veya mimari çözümü gösteren kısmı al.
5. ADİL PUANLAMA (`blog_worthiness_score`):
   - 1-4 Puan: Basit 404 sayfaları, standart login/signup formları, basit listeleme kartları veya sarmalayıcı (wrapper) bileşenler.
   - 5-6 Puan: Standart API çağrıları, temel routing yapıları, orta düzey bileşenler.
   - 7-10 Puan: Karmaşık state yönetimi, AI/sohbet entegrasyonu, akıllı arama/filtreleme, özel algoritmalar, custom hook'lar ve performans optimizasyonları içeren dosyalar.
6. DİL: Kod kesitleri ve evrensel teknik terimler hariç tüm açıklamaları akıcı ve profesyonel bir Türkçe ile yaz."""

SCHEMA_JSON = json.dumps(FileAnalysisSchema.model_json_schema(), ensure_ascii=False, indent=2)

ANALYZER_SYSTEM_PROMPT_FULL = ANALYZER_SYSTEM_PROMPT + f"""

ÇIKTI FORMATI (ZORUNLU):
Yalnızca aşağıdaki JSON şemasına uyan TEK BİR JSON nesnesi döndür.
- Kök seviyede doğrudan şu alanlar olmalı: architectural_domain, responsibility, libraries_and_hooks, engineering_highlights, blog_worthiness_score, best_snippet.
- Başka bir anahtarın (örn. "file_analysis") içine SARMA.
- Şemada olmayan alan ekleme.
- best_snippet bir nesnedir: {{"code": "...", "explanation": "..."}} veya null.

Şema:
{SCHEMA_JSON}
"""


class AnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            model_env_var="ANALYZER_MODEL",
            default_model="qwen3-coder:480b-cloud",
            temperature=0.0,
            format_schema=FileAnalysisSchema.model_json_schema(),
            agent_role="analyzer",
        )

    def run(
        self,
        repo_name: str,
        file_path: str,
        file_content: str,
        max_retries: int = 3,
    ) -> FileAnalysisSchema:
        if len(file_content) > MAX_CONTENT_CHARS:
            file_content = file_content[:MAX_CONTENT_CHARS] + "\n... [dosya kırpıldı]"

        user_prompt = (
            f"Proje Adı: {repo_name}\n"
            f"Dosya Yolu: {file_path}\n\n"
            f"Dosya İçeriği:\n```\n{file_content}\n```\n\n"
            "Bu dosyayı analiz et ve SADECE şemaya uyan JSON döndür."
        )
        messages = [("system", ANALYZER_SYSTEM_PROMPT_FULL), ("user", user_prompt)]

        return self.invoke_with_retry(
            messages=messages,
            response_processor=FileAnalysisSchema.from_raw_response,
            max_retries=max_retries,
        )