"""
MERKEZİ ORKESTRATÖR VE KONTROLCÜ (main.py)

Uçtan uca akışı yöneten saf kontrolcü (Controller):
1. GitHubFetcher ile repoyu belleğe çeker ve önbelleğe yazar.
2. AnalyzerAgent ile dosyaları analiz eder.
3. PlannerAgent ile odak alanını ve blog iskeletini (BlogOutline) oluşturur.
4. WriterAgent ile bölümleri ve kod kesitlerini birleştirip Markdown üretir.
5. Sonuçları (JSON / Markdown) diske kaydeder.
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(override=True)

from agents.analyzer_agent import AnalyzerAgent
from agents.planner_agent import DEFAULT_AUDIENCE, DEFAULT_TONE, PlannerAgent
from agents.writer_agent import WriterAgent
from core.github_fetcher import GitHubFetcher, parse_repo_url
from core.utils import (
    SNIPPET_TOKEN,
    final_checks,
    partition_repo_files,
    word_count,
)
from schemas.blog_schemas import BlogOutline

TOTAL_STAGES = 5
MIN_ANALYZED_FILES = 3
MAX_OUTLINE_ATTEMPTS = 3


# =====================================================================
# I/O, ÖNBELLEK VE CLI YARDIMCILARI
# =====================================================================

def banner(n: int, title: str):
    print("\n" + "=" * 65)
    print(f" [{n}/{TOTAL_STAGES}] {title}")
    print("=" * 65)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_paths(workdir: str, repo: str) -> Dict[str, str]:
    base = os.path.join(workdir, repo)
    return {
        "code": f"{base}_code_contents.json",
        "analysis": f"{base}_analysis.json",
        "analysis_partial": f"{base}_analysis_partial.json",
        "outline": f"{base}_outline.json",
        "draft_cache": f"{base}_draft_cache.json",
        "draft": f"{base}_draft.md",
        "meta": f"{base}_draft_meta.json",
    }


def check_environment():
    host = os.getenv("OLLAMA_HOST") or "[https://ollama.com](https://ollama.com)"
    if "ollama.com" in host and not os.getenv("OLLAMA_API_KEY"):
        print("UYARI: OLLAMA_API_KEY .env içinde bulunamadı. Bulut modeller 401 hatası verebilir.")
    if not os.getenv("GITHUB_TOKEN"):
        print("UYARI: GITHUB_TOKEN .env içinde yok. GitHub rate limit düşük olacak.")


def print_outline(outline: dict):
    print("\n" + "=" * 65)
    print(f" ODAK ALANI: {outline.get('selected_focus_area', '-')}")
    if outline.get("recommended_focus_areas"):
        print(" ALTERNATİF ODAKLAR:")
        for idx, fa in enumerate(outline["recommended_focus_areas"], 1):
            marker = "★" if fa == outline.get("selected_focus_area") else " "
            print(f"   [{idx}] {marker} {fa}")
    print("-" * 65)
    print(f" BAŞLIK    : {outline['chosen_title']}")
    print(f" ALT BAŞLIK: {outline['subtitle']}")
    print(f" OKUMA     : ~{outline['estimated_read_minutes']} dk | ETİKETLER: {', '.join(outline['tags'])}")
    print("-" * 65)
    for i, sec in enumerate(outline["sections"], 1):
        print(f" {i}. {sec['heading']} (~{sec['estimated_words']} kelime)")
        print(f"    Amaç    : {sec['goal']}")
        print(f"    Dosyalar: {', '.join(sec['source_files']) or '-'}")
        if sec.get("use_snippet_from"):
            print(f"    Kod     : {sec['use_snippet_from']}")
    print("-" * 65)
    print(f" SONUÇ     : {outline['conclusion_idea']}")
    print("=" * 65)


# =====================================================================
# AŞAMA 1: VERİ ÇEKME (core/github_fetcher.py)
# =====================================================================

def stage_fetch(owner: str, repo: str, path: str, fresh: bool) -> dict:
    banner(1, "REPO GITHUB'DAN ÇEKİLİYOR")

    if os.path.exists(path) and not fresh:
        data = load_json(path)
        print(f"Önbellekten yüklendi: {path} ({len(data.get('files', []))} dosya)")
        return data

    fetcher = GitHubFetcher()
    print(f"Repo indiriliyor: {owner}/{repo}...")

    def on_progress(idx: int, total: int, file_path: str, size: int):
        print(f"  [{idx}/{total}] {file_path} ({size / 1024:.1f} KB)")

    data = fetcher.fetch_repository(owner, repo, progress_callback=on_progress)
    save_json(path, data)
    print(f"Kaydedildi: {path} ({data['total_size_kb']} KB)")
    return data


# =====================================================================
# AŞAMA 2: DOSYA ANALİZİ (agents/analyzer_agent.py)
# =====================================================================

def stage_analyze(repo_data: dict, paths: Dict[str, str], workers: int, fresh: bool) -> dict:
    banner(2, "DOSYALAR ANALYZER AGENT İLE İNCELENİYOR")

    if os.path.exists(paths["analysis"]) and not fresh:
        data = load_json(paths["analysis"])
        print(f"Önbellekten yüklendi: {paths['analysis']} ({data.get('analyzed_files_count')} dosya analizli)")
        return data

    repo_name = repo_data.get("repository", "Bilinmeyen_Repo")
    files = repo_data.get("files", [])
    directory_structure = repo_data.get("directory_structure") or [f["path"] for f in files]

    dependency_manifests, markdown_docs, analyzable, _ = partition_repo_files(files)

    print(f"Proje: {repo_name} | Toplam dosya: {len(files)} | Analiz edilecek: {len(analyzable)}")
    if not analyzable:
        raise RuntimeError("Analiz edilecek kod dosyası bulunamadı.")

    partial: Dict[str, dict] = {}
    if os.path.exists(paths["analysis_partial"]) and not fresh:
        partial = load_json(paths["analysis_partial"])
        print(f"Önceki çalışmadan {len(partial)} dosya analizi önbellekten yüklendi.")

    todo = [f for f in analyzable if f["path"] not in partial]
    failed: List[dict] = []

    if todo:
        analyzer = AnalyzerAgent()
        print(f"Model: {analyzer.model_name} ({workers} paralel istek)...\n")

        def analyze_task(f_info: dict):
            try:
                result_schema = analyzer.run(
                    repo_name=repo_name,
                    file_path=f_info["path"],
                    file_content=f_info["content"],
                )
                return f_info, result_schema.model_dump(), None
            except Exception as e:
                return f_info, None, str(e)

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(analyze_task, f) for f in todo]
            for fut in as_completed(futures):
                f_info, res_dict, err = fut.result()
                done += 1
                if err:
                    print(f"  [{done}/{len(todo)}] ✗ {f_info['path']}: {err[:150]}")
                    failed.append({"path": f_info["path"], "error": err[:500]})
                else:
                    partial[f_info["path"]] = {"path": f_info["path"], "size": f_info.get("size", 0), **res_dict}
                    save_json(paths["analysis_partial"], partial)
                    dom = res_dict.get("architectural_domain", "Genel Mimari")
                    print(f"  [{done}/{len(todo)}] ✓ {f_info['path']} [{dom}] (puan: {res_dict['blog_worthiness_score']}/10)")

    analyzed = [partial[f["path"]] for f in analyzable if f["path"] in partial]
    min_needed = min(MIN_ANALYZED_FILES, len(analyzable))
    if len(analyzed) < min_needed:
        raise RuntimeError(f"Yalnızca {len(analyzed)} dosya analiz edilebildi (en az {min_needed} gerekli).")

    analyzed.sort(key=lambda x: x.get("blog_worthiness_score", 0), reverse=True)
    highlights: List[str] = []
    for item in analyzed:
        for h in item.get("engineering_highlights", []):
            if h not in highlights:
                highlights.append(h)

    analysis_data = {
        "repository": repo_name,
        "description": repo_data.get("description"),
        "stargazers_count": repo_data.get("stargazers_count", 0),
        "languages": repo_data.get("languages", []),
        "directory_structure": directory_structure,
        "dependency_manifests": dependency_manifests,
        "markdown_docs": markdown_docs,
        "project_wide_highlights": highlights,
        "top_architectural_files": [
            {
                "path": i["path"],
                "architectural_domain": i.get("architectural_domain", "Genel Mimari"),
                "score": i["blog_worthiness_score"],
                "responsibility": i["responsibility"],
                "engineering_highlights": i["engineering_highlights"],
            }
            for i in analyzed if i.get("blog_worthiness_score", 0) >= 7
        ],
        "analyzed_files_count": len(analyzed),
        "failed_files": failed,
        "file_analyses": analyzed,
    }
    save_json(paths["analysis"], analysis_data)
    if os.path.exists(paths["analysis_partial"]) and not failed:
        os.remove(paths["analysis_partial"])

    print(f"\nKaydedildi: {paths['analysis']}")
    return analysis_data


# =====================================================================
# AŞAMA 3: BLOG PLANLAMA (agents/planner_agent.py)
# =====================================================================

def _outline_quality_ok(outline: dict) -> bool:
    secs = outline["sections"]
    return len(secs) >= 3 and all(s["source_files"] for s in secs)


def _interactive_review(
    planner: PlannerAgent,
    analysis: dict,
    audience: str,
    tone: str,
    focus_area: Optional[str],
    outline: dict,
) -> dict:
    feedback, need_generate = None, False
    while True:
        if need_generate:
            outline_model = planner.run(
                analysis=analysis,
                audience=audience,
                tone=tone,
                focus_area=focus_area,
                previous_outline=outline if feedback else None,
                feedback=feedback,
            )
            outline = outline_model.model_dump()
            for w in planner.last_warnings:
                print(f"  ! {w}")
            print_outline(outline)
            need_generate = False

        print("\n  Enter = Onayla | r = Yeniden üret | f <n> = Odak seç | t <n> = Başlık seç | Diğer = Geri bildirim")
        choice = input("> ").strip()
        if choice == "":
            return outline
        if choice.lower() == "r":
            outline, feedback, need_generate = None, None, True
        elif re.fullmatch(r"f\s+\d+", choice.lower()):
            idx = int(choice.split()[1]) - 1
            focus_list = outline.get("recommended_focus_areas", [])
            if 0 <= idx < len(focus_list):
                focus_area = focus_list[idx]
                print(f"Yeni odak alanı seçildi: {focus_area} -> İskelet bu odağa göre yeniden üretiliyor...")
                outline, feedback, need_generate = None, None, True
            else:
                print("Geçersiz odak numarası.")
        elif re.fullmatch(r"t\s+\d+", choice.lower()):
            idx = int(choice.split()[1]) - 1
            if 0 <= idx < len(outline["title_options"]):
                outline["chosen_title"] = outline["title_options"][idx]
                print(f"Başlık seçildi: {outline['chosen_title']}")
            else:
                print("Geçersiz numara.")
        else:
            feedback, need_generate = choice, True


def stage_outline(
    analysis: dict,
    audience: str,
    tone: str,
    focus_area: Optional[str],
    review: bool,
    path: str,
    fresh: bool,
    analysis_path: str,
) -> dict:
    banner(3, "PLANNER AGENT İLE BLOG İSKELETİ HAZIRLANIYOR")

    if os.path.exists(path) and not fresh and not review and not focus_area:
        data = load_json(path)
        print(f"Önbellekten yüklendi: {path} | Başlık: {data['outline']['chosen_title']}")
        return data

    planner = PlannerAgent()
    print(f"Model: {planner.model_name} | Hedef kitle: {audience} | Odak: {focus_area or 'Otomatik Keşif'}")

    outline = None
    for attempt in range(1, MAX_OUTLINE_ATTEMPTS + 1):
        print(f"Outline üretiliyor (deneme {attempt}/{MAX_OUTLINE_ATTEMPTS})...")
        outline_model: BlogOutline = planner.run(
            analysis=analysis,
            audience=audience,
            tone=tone,
            focus_area=focus_area,
        )
        outline = outline_model.model_dump()
        for w in planner.last_warnings:
            print(f"  ! {w}")
        if _outline_quality_ok(outline):
            break
        print("  Kalite kapısı: bölüm sayısı <3 veya kaynak dosyasız bölüm var, yeniden deneniyor.")
    else:
        kept = [s for s in outline["sections"] if s["source_files"]]
        if len(kept) < 2:
            raise RuntimeError("Geçerli bir outline üretilemedi.")
        outline["sections"] = kept

    print_outline(outline)
    if review:
        outline = _interactive_review(planner, analysis, audience, tone, focus_area, outline)

    outline_data = {
        "repository": analysis.get("repository"),
        "source_analysis_file": analysis_path,
        "audience": audience,
        "tone": tone,
        "focus_area": outline.get("selected_focus_area"),
        "outline": outline,
    }
    save_json(path, outline_data)
    print(f"\nKaydedildi: {path}")
    return outline_data


# =====================================================================
# AŞAMA 4: BLOG YAZIMI (agents/writer_agent.py)
# =====================================================================

def stage_write(outline_data: dict, analysis: dict, repo_data: dict, paths: Dict[str, str], fresh: bool) -> Tuple[Dict[str, str], List[str], str]:
    banner(4, "WRITER AGENT İLE BLOG YAZISI YAZILIYOR")

    code_contents = {f["path"]: f.get("content", "") for f in repo_data.get("files", [])}
    writer = WriterAgent(outline_data=outline_data, analysis=analysis, code_contents=code_contents)
    sections = outline_data["outline"]["sections"]
    print(f"Model: {writer.model_name} | Bölüm sayısı: {len(sections)} | Odak: {outline_data['outline'].get('selected_focus_area', '-')}")

    initial_parts: Dict[str, str] = {}
    if os.path.exists(paths["draft_cache"]):
        if fresh:
            os.remove(paths["draft_cache"])
        else:
            initial_parts = load_json(paths["draft_cache"]).get("parts", {})
            print(f"Önbellekten {len(initial_parts)} tamamlanmış parça yüklendi.")

    def on_part_complete(part_key: str, text: str, current_parts: Dict[str, str]):
        save_json(paths["draft_cache"], {"parts": current_parts})
        label = "Giriş" if part_key == "intro" else ("Sonuç" if part_key == "conclusion" else part_key)
        print(f"  ✓ [{label}] tamamlandı ve önbelleğe alındı ({word_count(text)} kelime)")

    result = writer.run(initial_parts=initial_parts, on_part_complete=on_part_complete)
    return result["parts"], result["warnings"], result["markdown"]


# =====================================================================
# AŞAMA 5: TESLİM VE KAYIT
# =====================================================================

def stage_deliver(markdown: str, outline: dict, analysis: dict, warnings: List[str], paths: Dict[str, str], show: bool, elapsed: float):
    banner(5, "TASLAK DİSKE KAYDEDİLİYOR")

    issues = final_checks(markdown)
    if SNIPPET_TOKEN in markdown:
        markdown = markdown.replace(SNIPPET_TOKEN, "")
        issues = [i for i in issues if "yer tutucu" not in i.lower()] + ["Kullanılmamış yer tutucu otomatik temizlendi."]

    with open(paths["draft"], "w", encoding="utf-8") as f:
        f.write(markdown)

    total_words = word_count(re.sub(r"```.*?```", "", markdown, flags=re.DOTALL))
    code_blocks = markdown.count("```") // 2
    meta = {
        "repository": analysis.get("repository"),
        "focus_area": outline.get("selected_focus_area"),
        "recommended_focus_areas": outline.get("recommended_focus_areas", []),
        "title": outline["chosen_title"],
        "subtitle": outline["subtitle"],
        "tags": outline.get("tags", []),
        "word_count": total_words,
        "estimated_read_minutes": max(1, round(total_words / 200)),
        "code_blocks": code_blocks,
        "final_check_issues": issues,
        "writer_warnings": warnings,
        "draft_file": paths["draft"],
    }
    save_json(paths["meta"], meta)

    if os.path.exists(paths["draft_cache"]):
        os.remove(paths["draft_cache"])

    if show:
        print("\n" + "#" * 65)
        print("#  BLOG YAZISI")
        print("#" * 65 + "\n")
        print(markdown)
        print("\n" + "#" * 65)

    print("\n" + "=" * 65)
    print(f" TAMAMLANDI! (toplam {elapsed / 60:.1f} dakika)")
    print(f" Odak Alanı : {outline.get('selected_focus_area', '-')}")
    print(f" Başlık     : {outline['chosen_title']}")
    print(f" Uzunluk    : ~{total_words} kelime (~{meta['estimated_read_minutes']} dk okuma) | {code_blocks} kod bloğu")
    print(f" Taslak     : {paths['draft']}")
    print(f" Meta       : {paths['meta']}")
    print("=" * 65)


# =====================================================================
# ANA KONTROLCÜ GİRİŞ NOKTASI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GitHub reposundan Medium blog yazısı üretir.")
    p.add_argument("url", nargs="?", help="GitHub repo URL'si (verilmezse sorulur)")
    p.add_argument("--audience", default=None, help="Hedef kitle")
    p.add_argument("--tone", default=None, help="Yazı tonu")
    p.add_argument("--focus", default=None, help="Blog yazısının odaklanacağı spesifik teknik/mimari alan")
    p.add_argument("--review", action="store_true", help="Outline aşamasında onay/geri bildirim iste")
    p.add_argument("--workers", type=int, default=3, help="Paralel LLM isteği sayısı")
    p.add_argument("--fresh", action="store_true", help="Önbelleği yok say ve baştan çalış")
    p.add_argument("--output-dir", default="output", help="Çıktı klasörü")
    p.add_argument("--no-print", action="store_true", help="Bitince yazıyı ekrana basma")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print(" GITHUB REPO -> MEDIUM BLOG YAZISI (PIPELINE CONTROLLER)")
    print("=" * 65)

    url = args.url or input("GitHub Repo URL'sini girin: ").strip()
    try:
        owner, repo = parse_repo_url(url)
    except ValueError as e:
        print(f"\n✗ HATA: {e}")
        sys.exit(1)

    check_environment()

    workdir = os.path.join(args.output_dir, f"{owner}__{repo}")
    os.makedirs(workdir, exist_ok=True)
    paths = build_paths(workdir, repo)

    audience = args.audience or DEFAULT_AUDIENCE
    tone = args.tone or DEFAULT_TONE

    start = time.time()
    try:
        repo_data = stage_fetch(owner, repo, paths["code"], args.fresh)
        analysis = stage_analyze(repo_data, paths, args.workers, args.fresh)
        outline_data = stage_outline(
            analysis=analysis,
            audience=audience,
            tone=tone,
            focus_area=args.focus,
            review=args.review,
            path=paths["outline"],
            fresh=args.fresh,
            analysis_path=paths["analysis"],
        )

        if os.path.exists(paths["draft"]) and not args.fresh and not args.focus:
            print(f"\n[4/{TOTAL_STAGES}] Taslak zaten var ({paths['draft']}). Yeniden yazmak için --fresh kullanın.")
            with open(paths["draft"], "r", encoding="utf-8") as f:
                markdown = f.read()
            warnings: List[str] = []
        else:
            _, warnings, markdown = stage_write(outline_data, analysis, repo_data, paths, args.fresh or bool(args.focus))

        stage_deliver(markdown, outline_data["outline"], analysis, warnings, paths, not args.no_print, time.time() - start)

    except KeyboardInterrupt:
        print("\n\nİşlem kullanıcı tarafından durduruldu. Aynı komutu çalıştırarak kaldığınız yerden devam edebilirsiniz.")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ HATA: {e}")
        print("Aynı komutu tekrar çalıştırdığınızda tamamlanan aşamalar önbellekten yüklenerek devam edecektir.")
        sys.exit(1)


if __name__ == "__main__":
    main()