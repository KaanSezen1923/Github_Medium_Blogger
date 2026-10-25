"""
GITHUB REPO -> MEDIUM BLOG YAZISI (Uçtan Uca Pipeline)

Tek komutla tüm aşamaları sırayla çalıştırır:
    1. try.py          -> Repoyu GitHub'dan çek, filtrele
    2. analyze_repo.py -> Her dosyayı LLM ile analiz et
    3. plan_blog.py    -> Blog iskeletini (outline) çıkar
    4. write_blog.py   -> Bölüm bölüm yaz, kod kesitlerini doğrulayıp göm
    5. Teslim          -> Markdown taslağı kaydet ve ekrana bas

Bu dosya diğer dört dosyayı DEĞİŞTİRMEDEN onların sınıf/fonksiyonlarını kullanır.
Hepsi (try.py, analyze_repo.py, plan_blog.py, write_blog.py, logger.py, .env) aynı dizinde olmalıdır.

Kullanım:
    python main.py https://github.com/kullanici/repo
    python main.py                                   # URL'yi sorar
    python main.py <url> --review                    # outline aşamasında durup onay ister
    python main.py <url> --audience "yeni başlayanlar" --tone "samimi, hikâye anlatan"
    python main.py <url> --workers 4                 # analizde paralel istek sayısı
    python main.py <url> --fresh                     # önbelleği yok say, baştan çalış

Hata olursa veya Ctrl+C yaparsanız aynı komutu tekrar çalıştırın: tamamlanan aşamalar ve
dosyalar önbellekten alınır, kalan yerden devam edilir.

Çıktılar: output/<sahip>__<repo>/ klasörüne yazılır.
"""

import argparse
import base64
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

import analyze_repo   # noqa: E402
import plan_blog      # noqa: E402
import write_blog     # noqa: E402

TOTAL_STAGES = 5
MIN_ANALYZED_FILES = 3          # Bu kadar dosya analiz edilemezse pipeline durur
MAX_OUTLINE_ATTEMPTS = 3        # Kalite kapısını geçemezse outline en fazla bu kadar üretilir


# =====================================================================
# YARDIMCILAR
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


def load_fetcher():
    """'try' Python'da ayrılmış bir kelime olduğu için try.py'yi importlib ile yüklüyoruz."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "try.py")
    if not os.path.exists(path):
        raise FileNotFoundError("try.py bulunamadı. main.py ile aynı dizinde olmalı.")
    spec = importlib.util.spec_from_file_location("github_fetcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_repo_url(url: str) -> Tuple[str, str]:
    parsed = urlparse(url.strip())
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Geçersiz GitHub Repo URL'si. Örnek: https://github.com/kullanici/repo")
    return parts[0], re.sub(r"\.git$", "", parts[1])


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
    host = (os.getenv("OLLAMA_HOST") or "https://ollama.com")
    if "ollama.com" in host and not os.getenv("OLLAMA_API_KEY"):
        print("UYARI: OLLAMA_API_KEY .env içinde bulunamadı. Bulut modeller 401 hatası verebilir.")
    if not os.getenv("GITHUB_TOKEN"):
        print("UYARI: GITHUB_TOKEN .env içinde yok. GitHub rate limit düşük olacak.")


# =====================================================================
# AŞAMA 1: REPO ÇEKME
# =====================================================================

def stage_fetch(owner: str, repo: str, fetcher, path: str, fresh: bool) -> dict:
    banner(1, "REPO GITHUB'DAN ÇEKİLİYOR")

    if os.path.exists(path) and not fresh:
        data = load_json(path)
        print(f"Önbellekten yüklendi: {path} ({len(data.get('files', []))} dosya)")
        return data

    print(f"Repo bilgileri alınıyor: {owner}/{repo}...")
    repo_info = fetcher.gh_get(f"/repos/{owner}/{repo}")
    branch = repo_info.get("default_branch", "main")
    languages = fetcher.gh_get(f"/repos/{owner}/{repo}/languages")

    print(f"Dosya ağacı alınıyor (Branch: {branch})...")
    tree_data = fetcher.gh_get(f"/repos/{owner}/{repo}/git/trees/{branch}", recursive=1)
    if tree_data.get("truncated"):
        print("UYARI: Repo çok büyük, GitHub dosya ağacını kırptı. Bazı dosyalar eksik olabilir.")

    blobs = [
        e for e in tree_data.get("tree", [])
        if e["type"] == "blob" and fetcher.is_allowed_file(e["path"], e.get("size", 0))
    ]
    if not blobs:
        raise RuntimeError("Filtreleme sonrası indirilecek dosya kalmadı. Repo desteklenen dosya türlerini içermiyor olabilir.")

    print(f"İndirilecek dosya sayısı: {len(blobs)}")
    files, total_size = [], 0
    for idx, blob in enumerate(blobs, 1):
        size = blob.get("size", 0)
        print(f"  [{idx}/{len(blobs)}] {blob['path']} ({size / 1024:.1f} KB)")
        try:
            blob_data = fetcher.gh_get(f"/repos/{owner}/{repo}/git/blobs/{blob['sha']}")
            content = base64.b64decode(blob_data["content"]).decode("utf-8", errors="replace")
            total_size += size
        except Exception as e:
            content = f"[İndirme/Çözümleme Hatası: {e}]"
        files.append({"path": blob["path"], "size": size, "sha": blob["sha"], "content": content})

    data = {
        "repository": repo_info.get("full_name"),
        "description": repo_info.get("description"),
        "stargazers_count": repo_info.get("stargazers_count"),
        "languages": list(languages.keys()),
        "total_files": len(files),
        "total_size_kb": round(total_size / 1024, 2),
        "directory_structure": [f["path"] for f in files],
        "files": files,
    }
    save_json(path, data)
    print(f"Kaydedildi: {path} ({data['total_size_kb']} KB)")
    return data


# =====================================================================
# AŞAMA 2: DOSYA ANALİZİ
# =====================================================================

def stage_analyze(repo_data: dict, paths: Dict[str, str], workers: int, fresh: bool) -> dict:
    banner(2, "DOSYALAR LLM İLE ANALİZ EDİLİYOR")

    if os.path.exists(paths["analysis"]) and not fresh:
        data = load_json(paths["analysis"])
        print(f"Önbellekten yüklendi: {paths['analysis']} ({data.get('analyzed_files_count')} dosya analizli)")
        return data

    repo_name = repo_data.get("repository", "Bilinmeyen_Repo")
    files = repo_data.get("files", [])
    directory_structure = repo_data.get("directory_structure") or [f["path"] for f in files]

    dependency_manifests, markdown_docs, analyzable = {}, {}, []
    for f_info in files:
        path = f_info["path"]
        fname = path.split("/")[-1].lower()
        content = f_info.get("content", "")
        if fname in analyze_repo.META_FILENAMES:
            try:
                dependency_manifests[path] = json.loads(content) if fname.endswith(".json") else content
            except Exception:
                dependency_manifests[path] = content
            continue
        if fname.endswith(".md"):
            markdown_docs[path] = content
            continue
        ok, _ = analyze_repo.should_analyze_with_llm(f_info)
        if ok:
            analyzable.append(f_info)

    print(f"Proje: {repo_name} | Toplam dosya: {len(files)} | Analiz edilecek: {len(analyzable)}")
    if not analyzable:
        raise RuntimeError("Analiz edilecek kod dosyası bulunamadı.")

    # Kaldığı yerden devam: daha önce başarıyla analiz edilenleri atla
    partial: Dict[str, dict] = {}
    if os.path.exists(paths["analysis_partial"]) and not fresh:
        partial = load_json(paths["analysis_partial"])
        print(f"Önceki çalışmadan {len(partial)} dosya analizi bulundu, atlanacak.")

    todo = [f for f in analyzable if f["path"] not in partial]
    failed: List[dict] = []

    if todo:
        analyzer = analyze_repo.LLMAnalyzer()
        print(f"Model çağrıları başlıyor ({workers} paralel istek)...\n")

        def work(f_info):
            try:
                res = analyzer.analyze_file(repo_name=repo_name, file_path=f_info["path"], file_content=f_info["content"])
                return f_info, res, None
            except Exception as e:
                return f_info, None, str(e)

        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(work, f) for f in todo]
            for fut in as_completed(futures):
                f_info, res, err = fut.result()
                done += 1
                if err:
                    print(f"  [{done}/{len(todo)}] ✗ {f_info['path']}: {err[:150]}")
                    failed.append({"path": f_info["path"], "error": err[:500]})
                else:
                    partial[f_info["path"]] = {"path": f_info["path"], "size": f_info.get("size", 0), **res}
                    save_json(paths["analysis_partial"], partial)
                    print(f"  [{done}/{len(todo)}] ✓ {f_info['path']} (puan: {res['blog_worthiness_score']}/10)")

    analyzed = [partial[f["path"]] for f in analyzable if f["path"] in partial]
    min_needed = min(MIN_ANALYZED_FILES, len(analyzable))
    if len(analyzed) < min_needed:
        raise RuntimeError(
            f"Yalnızca {len(analyzed)} dosya analiz edilebildi (en az {min_needed} gerekli). "
            "Komutu tekrar çalıştırın; başarılı olanlar atlanır, başarısızlar yeniden denenir. "
            "Sorun sürerse .env içinde OLLAMA_MODEL değerini değiştirmeyi deneyin."
        )
    if failed:
        print(f"\nUYARI: {len(failed)} dosya analiz edilemedi, bunlarsız devam ediliyor.")

    analyzed.sort(key=lambda x: x.get("blog_worthiness_score", 0), reverse=True)
    highlights: List[str] = []
    for item in analyzed:
        for h in item.get("engineering_highlights", []):
            if h not in highlights:
                highlights.append(h)

    analysis = {
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
                "path": i["path"], "score": i["blog_worthiness_score"],
                "responsibility": i["responsibility"], "engineering_highlights": i["engineering_highlights"],
            }
            for i in analyzed if i.get("blog_worthiness_score", 0) >= 7
        ],
        "analyzed_files_count": len(analyzed),
        "failed_files": failed,
        "file_analyses": analyzed,
    }
    save_json(paths["analysis"], analysis)
    if os.path.exists(paths["analysis_partial"]) and not failed:
        os.remove(paths["analysis_partial"])

    print(f"\nKaydedildi: {paths['analysis']}")
    print("En değerli dosyalar:")
    for item in analyzed[:5]:
        print(f"  ★ [{item['blog_worthiness_score']}/10] {item['path']}")
    return analysis


# =====================================================================
# AŞAMA 3: OUTLINE
# =====================================================================

def _outline_quality_ok(outline: dict) -> bool:
    secs = outline["sections"]
    return len(secs) >= 3 and all(s["source_files"] for s in secs)


def _review_outline(planner, context, audience, tone, outline, analysis) -> dict:
    """--review modunda: kullanıcı outline'ı onaylar veya geri bildirimle revize ettirir."""
    feedback, need_generate = None, False
    while True:
        if need_generate:
            outline = planner.plan(context=context, audience=audience, tone=tone,
                                   previous_outline=outline if feedback else None, feedback=feedback)
            for w in plan_blog.validate_outline(outline, analysis):
                print(f"  ! {w}")
            plan_blog.print_outline(outline)
            need_generate = False

        print("\n  Enter = Onayla | r = Yeniden üret | t <n> = Başlık seç | Diğer = Geri bildirim")
        choice = input("> ").strip()
        if choice == "":
            return outline
        if choice.lower() == "r":
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


def stage_outline(analysis: dict, audience: str, tone: str, review: bool, path: str, fresh: bool, analysis_path: str) -> dict:
    banner(3, "BLOG İSKELETİ (OUTLINE) HAZIRLANIYOR")

    if os.path.exists(path) and not fresh and not review:
        data = load_json(path)
        print(f"Önbellekten yüklendi: {path} | Başlık: {data['outline']['chosen_title']}")
        return data

    context_files = plan_blog.select_context_files(analysis)
    if not context_files:
        print(f"Puanı {plan_blog.MIN_SCORE_IN_CONTEXT}+ dosya yok, eşik düşürülüyor.")
        plan_blog.MIN_SCORE_IN_CONTEXT = 1
        context_files = plan_blog.select_context_files(analysis)
    context = plan_blog.build_context(analysis, context_files)
    print(f"Bağlama {len(context_files)} dosya alındı (~{len(context)} karakter). Hedef kitle: {audience}")

    planner = plan_blog.OutlinePlanner()
    print(f"Model: {planner.model_name}")

    outline = None
    for attempt in range(1, MAX_OUTLINE_ATTEMPTS + 1):
        print(f"Outline üretiliyor (deneme {attempt}/{MAX_OUTLINE_ATTEMPTS})...")
        outline = planner.plan(context=context, audience=audience, tone=tone)
        for w in plan_blog.validate_outline(outline, analysis):
            print(f"  ! {w}")
        if _outline_quality_ok(outline):
            break
        print("  Kalite kapısı: bölüm sayısı <3 veya kaynak dosyasız bölüm var, yeniden deneniyor.")
    else:
        # Son denemede de kapıyı geçemedi: kaynaksız bölümleri at, en az 2 bölüm kalıyorsa devam et
        kept = [s for s in outline["sections"] if s["source_files"]]
        if len(kept) < 2:
            raise RuntimeError("Geçerli bir outline üretilemedi. OUTLINE_MODEL için daha güçlü bir model deneyin.")
        outline["sections"] = kept
        print(f"  Kaynaksız bölümler çıkarıldı, {len(kept)} bölümle devam ediliyor.")

    plan_blog.print_outline(outline)
    if review:
        outline = _review_outline(planner, context, audience, tone, outline, analysis)

    data = {
        "repository": analysis.get("repository"),
        "source_analysis_file": analysis_path,
        "audience": audience,
        "tone": tone,
        "outline": outline,
    }
    save_json(path, data)
    print(f"\nKaydedildi: {path}")
    return data


# =====================================================================
# AŞAMA 4: YAZIM
# =====================================================================

def stage_write(outline_data: dict, analysis: dict, repo_data: dict, paths: Dict[str, str], fresh: bool) -> Tuple[Dict[str, str], List[str], str]:
    banner(4, "BLOG YAZISI BÖLÜM BÖLÜM YAZILIYOR")

    code_contents = {f["path"]: f.get("content", "") for f in repo_data.get("files", [])}
    writer = write_blog.BlogWriter(outline_data, analysis, code_contents)
    outline = outline_data["outline"]
    sections = outline["sections"]
    print(f"Model: {writer.model_name} | Bölüm sayısı: {len(sections)} | Kesit doğrulama: AÇIK")

    parts: Dict[str, str] = {}
    if os.path.exists(paths["draft_cache"]):
        if fresh:
            os.remove(paths["draft_cache"])
        else:
            parts = load_json(paths["draft_cache"]).get("parts", {})
            print(f"Yarım kalan taslaktan {len(parts)} parça yüklendi, devam ediliyor.")

    def checkpoint():
        save_json(paths["draft_cache"], {"parts": parts})

    if "intro" not in parts:
        print("[Giriş] yazılıyor...")
        parts["intro"] = writer.write_intro()
        checkpoint()

    for i, sec in enumerate(sections):
        key = f"sec_{i}"
        if key in parts:
            print(f"[{i + 1}/{len(sections)}] (önbellekten) {sec['heading']}")
            continue
        print(f"[{i + 1}/{len(sections)}] Yazılıyor: {sec['heading']}")
        previous = [(s["heading"], parts[f"sec_{j}"]) for j, s in enumerate(sections[:i])]
        text, snip = writer.write_section(i, previous)
        parts[key] = text
        checkpoint()
        print(f"   ✓ {write_blog.word_count(text)} kelime" + (f" | kod: {snip['path']}" if snip else ""))
        time.sleep(0.3)

    if "conclusion" not in parts:
        print("[Sonuç] yazılıyor...")
        parts["conclusion"] = writer.write_conclusion()
        checkpoint()

    markdown = write_blog.assemble_markdown(outline, analysis.get("repository", ""), parts)
    return parts, writer.warnings, markdown


# =====================================================================
# AŞAMA 5: TESLİM
# =====================================================================

def stage_deliver(markdown: str, outline: dict, analysis: dict, warnings: List[str], paths: Dict[str, str], show: bool, elapsed: float):
    banner(5, "TASLAK KAYDEDİLİYOR")

    issues = write_blog.final_checks(markdown, outline)
    if write_blog.SNIPPET_TOKEN in markdown:
        markdown = markdown.replace(write_blog.SNIPPET_TOKEN, "")   # güvenlik ağı: yazıda yer tutucu kalmasın
        issues = [i for i in issues if "yer tutucu" not in i.lower()] + ["Kullanılmamış yer tutucu otomatik temizlendi."]

    with open(paths["draft"], "w", encoding="utf-8") as f:
        f.write(markdown)

    total_words = write_blog.word_count(re.sub(r"```.*?```", "", markdown, flags=re.DOTALL))
    code_blocks = markdown.count("```") // 2
    meta = {
        "repository": analysis.get("repository"),
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
    print(f" Başlık     : {outline['chosen_title']}")
    print(f" Uzunluk    : ~{total_words} kelime (~{meta['estimated_read_minutes']} dk okuma) | {code_blocks} kod bloğu")
    print(f" Etiketler  : {', '.join(meta['tags'])}")
    print(f" Taslak     : {paths['draft']}")
    print(f" Meta       : {paths['meta']}")
    if warnings:
        print(f" Yazar uyarıları ({len(warnings)}):")
        for w in warnings[:8]:
            print(f"   ! {w}")
        if len(warnings) > 8:
            print(f"   ... ve {len(warnings) - 8} uyarı daha (meta dosyasında)")
    if issues:
        print(" Son kontrol:")
        for i in issues:
            print(f"   ! {i}")
    print("-" * 65)
    print(" Yayınlamadan önce yazıyı okuyup kontrol edin: model analizde olmayan bir")
    print(" iddia eklemiş olabilir; bunu otomatik olarak yakalayamıyoruz.")
    print("=" * 65)


# =====================================================================
# ANA AKIŞ
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GitHub reposundan Medium blog yazısı üretir.")
    p.add_argument("url", nargs="?", help="GitHub repo URL'si (verilmezse sorulur)")
    p.add_argument("--audience", default=None, help="Hedef kitle (varsayılan: .env BLOG_AUDIENCE)")
    p.add_argument("--tone", default=None, help="Yazı tonu (varsayılan: .env BLOG_TONE)")
    p.add_argument("--review", action="store_true", help="Outline aşamasında durup onay/geri bildirim iste")
    p.add_argument("--workers", type=int, default=3, help="Analiz aşamasında paralel LLM isteği sayısı (varsayılan 3)")
    p.add_argument("--fresh", action="store_true", help="Önbelleği yok say, tüm aşamaları baştan çalıştır")
    p.add_argument("--output-dir", default="output", help="Çıktı klasörü (varsayılan: output)")
    p.add_argument("--no-print", action="store_true", help="Bitince yazının tamamını ekrana basma")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print(" GITHUB REPO  ->  MEDIUM BLOG YAZISI")
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

    audience = args.audience or plan_blog.DEFAULT_AUDIENCE
    tone = args.tone or plan_blog.DEFAULT_TONE

    print(f"\nRepo       : {owner}/{repo}")
    print(f"Çıktı      : {workdir}")
    print(f"Hedef kitle: {audience}")
    print(f"Ton        : {tone}")

    start = time.time()
    try:
        fetcher = load_fetcher()
        repo_data = stage_fetch(owner, repo, fetcher, paths["code"], args.fresh)
        analysis = stage_analyze(repo_data, paths, args.workers, args.fresh)
        outline_data = stage_outline(analysis, audience, tone, args.review, paths["outline"], args.fresh, paths["analysis"])

        if os.path.exists(paths["draft"]) and not args.fresh:
            print(f"\n[4/{TOTAL_STAGES}] Taslak zaten var ({paths['draft']}). Yeniden yazmak için --fresh kullanın.")
            with open(paths["draft"], "r", encoding="utf-8") as f:
                markdown = f.read()
            warnings: List[str] = []
        else:
            _, warnings, markdown = stage_write(outline_data, analysis, repo_data, paths, args.fresh)

        stage_deliver(markdown, outline_data["outline"], analysis, warnings, paths, not args.no_print, time.time() - start)

    except KeyboardInterrupt:
        print("\n\nDurduruldu. Aynı komutu tekrar çalıştırırsanız kaldığı yerden devam eder.")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ HATA: {e}")
        print("Aynı komutu tekrar çalıştırırsanız tamamlanan aşamalar atlanır ve kaldığı yerden devam edilir.")
        sys.exit(1)


if __name__ == "__main__":
    main()