"""
GITHUB REPO -> MEDIUM BLOG YAZISI  |  Streamlit Arayüzü

Çalıştırma:
    pip install streamlit
    streamlit run app.py

Gereksinimler: main.py, try.py, analyze_repo.py, plan_blog.py, write_blog.py, logger.py ve .env
dosyaları bu dosyayla aynı dizinde olmalıdır. Arayüz, main.py içindeki aşama fonksiyonlarını
kullanır; mantık değişmez, sadece terminal yerine tarayıcıda çalışır.

Akış:
    1. Repo URL'sini gir, "Blog yazısı oluştur"a bas.
    2. (İsteğe bağlı) "Outline'ı önce incelemek istiyorum" açıksa, yazım başlamadan iskeleti
       görür, başlık seçer, geri bildirimle revize ettirir ve onaylarsın.
    3. Yazı oluşunca Önizleme / Markdown / Detaylar sekmelerinde görür, .md olarak indirirsin.
"""

import contextlib
import glob
import io
import json
import os
import threading
import time

import streamlit as st

st.set_page_config(page_title="Repo → Medium Blog", page_icon="✍️", layout="wide")

import main as pipeline   # noqa: E402  (set_page_config'ten sonra, aşama fonksiyonları burada)
import plan_blog          # noqa: E402

OUTPUT_DIR = "output"
STAGE_PROGRESS = {1: 0.05, 2: 0.15, 3: 0.45, 4: 0.55, 5: 0.95}


# =====================================================================
# CANLI LOG (print çıktılarını arayüze akıtır)
# =====================================================================

def _in_script_thread() -> bool:
    """Sadece Streamlit'in ana script thread'i arayüz elemanlarını güncelleyebilir."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        try:
            return get_script_run_ctx(suppress_warning=True) is not None
        except TypeError:
            return get_script_run_ctx() is not None
    except Exception:
        return True


class LiveLog(io.TextIOBase):
    """stdout'u yakalar; satırları biriktirir ve son birkaç satırı canlı olarak gösterir."""

    def __init__(self, placeholder, tail: int = 16, max_lines: int = 600):
        self.placeholder = placeholder
        self.tail = tail
        self.max_lines = max_lines
        self.lines = []
        self._partial = ""
        self._lock = threading.Lock()
        self._last_refresh = 0.0

    def writable(self):
        return True

    def write(self, s):
        if not s:
            return 0
        with self._lock:
            self._partial += s
            chunks = self._partial.split("\n")
            self._partial = chunks.pop()
            for c in chunks:
                c = c.rstrip()
                if c:
                    self.lines.append(c)
            if len(self.lines) > self.max_lines:
                self.lines = self.lines[-self.max_lines:]
        # Worker thread'lerden (paralel analiz) arayüze dokunma; sadece ana thread güncellesin
        if time.time() - self._last_refresh > 0.2 and _in_script_thread():
            self.refresh()
        return len(s)

    def flush(self):
        pass

    def refresh(self):
        self._last_refresh = time.time()
        with self._lock:
            text = "\n".join(self.lines[-self.tail:]) or "..."
        try:
            self.placeholder.code(text, language=None)
        except Exception:
            pass


# =====================================================================
# DURUM YÖNETİMİ VE DOSYA YARDIMCILARI
# =====================================================================

def init_state():
    defaults = {"phase": "input", "ctx": None, "outline_data": None, "result": None, "outline_version": 0}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def remove_files(paths: dict, keys):
    for k in keys:
        p = paths.get(k)
        if p and os.path.exists(p):
            os.remove(p)


def make_ctx(url: str, settings: dict) -> dict:
    owner, repo = pipeline.parse_repo_url(url)   # geçersizse ValueError
    workdir = os.path.join(OUTPUT_DIR, f"{owner}__{repo}")
    os.makedirs(workdir, exist_ok=True)
    return {
        "url": url, "owner": owner, "repo": repo,
        "paths": pipeline.build_paths(workdir, repo),
        "audience": settings["audience"], "tone": settings["tone"],
        "workers": settings["workers"], "fresh": settings["fresh"],
        "modified": False,
    }


def invalidate_if_style_changed(ctx: dict) -> bool:
    """Aynı repo için hedef kitle/ton değiştiyse outline ve yazı önbelleğini temizler."""
    paths = ctx["paths"]
    if not os.path.exists(paths["outline"]):
        return False
    try:
        old = pipeline.load_json(paths["outline"])
    except Exception:
        return False
    if old.get("audience") != ctx["audience"] or old.get("tone") != ctx["tone"]:
        remove_files(paths, ["outline", "draft", "draft_cache", "meta"])
        return True
    return False


def get_fallback_context(analysis: dict):
    context_files = plan_blog.select_context_files(analysis)
    if not context_files:
        plan_blog.MIN_SCORE_IN_CONTEXT = 1
        context_files = plan_blog.select_context_files(analysis)
    return plan_blog.build_context(analysis, context_files)


# =====================================================================
# PIPELINE ÇALIŞTIRICILAR
# =====================================================================

def run_prepare(ctx: dict, bar):
    """Aşama 1-3: repo çek, analiz et, outline çıkar."""
    paths, fresh = ctx["paths"], ctx["fresh"]

    if fresh:
        remove_files(paths, ["draft", "draft_cache", "meta"])
    elif invalidate_if_style_changed(ctx):
        print("Hedef kitle/ton değişti: outline ve yazı yeniden üretilecek.")

    bar.progress(STAGE_PROGRESS[1], text="1/5 · Repo GitHub'dan çekiliyor")
    fetcher = pipeline.load_fetcher()
    repo_data = pipeline.stage_fetch(ctx["owner"], ctx["repo"], fetcher, paths["code"], fresh)

    bar.progress(STAGE_PROGRESS[2], text="2/5 · Dosyalar analiz ediliyor (en uzun aşama)")
    analysis = pipeline.stage_analyze(repo_data, paths, ctx["workers"], fresh)

    bar.progress(STAGE_PROGRESS[3], text="3/5 · Blog iskeleti hazırlanıyor")
    outline_data = pipeline.stage_outline(
        analysis, ctx["audience"], ctx["tone"], False, paths["outline"], fresh, paths["analysis"]
    )
    return outline_data


def run_regenerate_outline(ctx: dict, outline_data: dict, feedback, bar) -> dict:
    """Review ekranında: outline'ı geri bildirimle revize eder veya sıfırdan üretir."""
    paths = ctx["paths"]
    analysis = pipeline.load_json(paths["analysis"])
    bar.progress(0.5, text="Outline yeniden hazırlanıyor")
    context = get_fallback_context(analysis)
    planner = plan_blog.OutlinePlanner()
    previous = outline_data["outline"] if feedback else None
    outline = planner.plan(context=context, audience=ctx["audience"], tone=ctx["tone"],
                           previous_outline=previous, feedback=feedback or None)
    for w in plan_blog.validate_outline(outline, analysis):
        print(f"  ! {w}")
    new_data = dict(outline_data)
    new_data["outline"] = outline
    return new_data


def run_write(ctx: dict, outline_data: dict, bar) -> dict:
    """Aşama 4-5: yazıyı yaz, kaydet ve sonucu arayüz için hazırla."""
    paths = ctx["paths"]
    repo_data = pipeline.load_json(paths["code"])
    analysis = pipeline.load_json(paths["analysis"])
    outline = outline_data["outline"]
    start = time.time()

    bar.progress(STAGE_PROGRESS[4], text="4/5 · Yazı bölüm bölüm yazılıyor")
    if os.path.exists(paths["draft"]):
        print(f"Mevcut taslak bulundu, yeniden yazılmıyor: {paths['draft']}")
        with open(paths["draft"], "r", encoding="utf-8") as f:
            markdown = f.read()
        warnings = []
    else:
        _, warnings, markdown = pipeline.stage_write(outline_data, analysis, repo_data, paths, False)

    bar.progress(STAGE_PROGRESS[5], text="5/5 · Taslak kaydediliyor")
    pipeline.stage_deliver(markdown, outline, analysis, warnings, paths, False, time.time() - start)
    bar.progress(1.0, text="Tamamlandı")
    return build_result(paths, outline, analysis)


def build_result(paths: dict, outline: dict, analysis: dict) -> dict:
    with open(paths["draft"], "r", encoding="utf-8") as f:
        markdown = f.read()
    meta = pipeline.load_json(paths["meta"]) if os.path.exists(paths["meta"]) else {}
    files = [
        {
            "Dosya": f["path"],
            "Puan": f.get("blog_worthiness_score"),
            "Öne çıkan": ", ".join(f.get("engineering_highlights", [])[:2]),
        }
        for f in analysis.get("file_analyses", [])
    ]
    return {
        "markdown": markdown,
        "meta": meta,
        "outline": outline,
        "files": files,
        "draft_path": paths["draft"],
        "repository": analysis.get("repository"),
    }


def load_history_result(draft_path: str) -> dict:
    base = draft_path[: -len("_draft.md")]
    with open(draft_path, "r", encoding="utf-8") as f:
        markdown = f.read()
    meta = pipeline.load_json(f"{base}_draft_meta.json") if os.path.exists(f"{base}_draft_meta.json") else {}
    outline, files = None, []
    if os.path.exists(f"{base}_outline.json"):
        outline = pipeline.load_json(f"{base}_outline.json").get("outline")
    if os.path.exists(f"{base}_analysis.json"):
        analysis = pipeline.load_json(f"{base}_analysis.json")
        files = [
            {"Dosya": f["path"], "Puan": f.get("blog_worthiness_score"),
             "Öne çıkan": ", ".join(f.get("engineering_highlights", [])[:2])}
            for f in analysis.get("file_analyses", [])
        ]
    return {"markdown": markdown, "meta": meta, "outline": outline, "files": files,
            "draft_path": draft_path, "repository": meta.get("repository")}


def list_history():
    return sorted(glob.glob(os.path.join(OUTPUT_DIR, "*", "*_draft.md")), key=os.path.getmtime, reverse=True)


# =====================================================================
# ÇALIŞMA KUTUSU (durum + ilerleme + canlı log)
# =====================================================================

def run_with_status(title: str, job):
    """
    job(bar) -> sonuç döndürür. Canlı log ve ilerleme çubuğuyla çalıştırır.
    Dönüş: (sonuç | None, hata_mesajı | None)
    """
    result, error = None, None
    with st.status(title, expanded=True) as status:
        bar = st.progress(0.0, text="Başlıyor")
        log_placeholder = st.empty()
        log = LiveLog(log_placeholder)
        try:
            with contextlib.redirect_stdout(log):
                result = job(bar)
            log.refresh()
            status.update(label="Tamamlandı", state="complete", expanded=False)
        except Exception as e:
            error = str(e)
            log.refresh()
            status.update(label="Hata oluştu", state="error", expanded=True)
    return result, error


def show_error(error: str):
    st.error(f"**Hata:** {error}")
    st.info("Aynı repo ile tekrar denerseniz tamamlanan aşamalar atlanır ve kaldığı yerden devam edilir.")


# =====================================================================
# ARAYÜZ BİLEŞENLERİ
# =====================================================================

def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        audience = st.text_input("Hedef kitle", value=plan_blog.DEFAULT_AUDIENCE)
        tone = st.text_input("Yazı tonu", value=plan_blog.DEFAULT_TONE)
        review = st.checkbox("Outline'ı yazmadan önce incelemek istiyorum", value=False,
                             help="Açıksa iskelet çıkınca durur; başlık seçebilir, geri bildirim verebilir ve onaylayabilirsiniz.")
        workers = st.slider("Paralel analiz isteği", 1, 8, 3,
                            help="Daha yüksek değer analizi hızlandırır ama API limitine takılma riskini artırır.")
        fresh = st.checkbox("Önbelleği yok say (baştan çalıştır)", value=False,
                            help="Aynı repo için daha önce üretilen analiz/outline/yazıyı kullanmaz.")

        st.divider()
        st.caption("Ortam durumu")
        if os.getenv("OLLAMA_API_KEY") or "ollama.com" not in (os.getenv("OLLAMA_HOST") or "https://ollama.com"):
            st.success("Ollama erişimi hazır", icon="✅")
        else:
            st.warning("OLLAMA_API_KEY .env içinde yok", icon="⚠️")
        if os.getenv("GITHUB_TOKEN"):
            st.success("GitHub token hazır", icon="✅")
        else:
            st.warning("GITHUB_TOKEN yok (düşük rate limit)", icon="⚠️")

        history = list_history()
        if history:
            st.divider()
            st.caption("Önceki yazılar")
            choice = st.selectbox(
                "Geçmiş", history, label_visibility="collapsed",
                format_func=lambda p: os.path.basename(os.path.dirname(p)).replace("__", " / "),
            )
            if st.button("Aç"):
                st.session_state.result = load_history_result(choice)
                st.session_state.phase = "done"
                st.rerun()

    return {"audience": audience.strip() or plan_blog.DEFAULT_AUDIENCE,
            "tone": tone.strip() or plan_blog.DEFAULT_TONE,
            "review": review, "workers": workers, "fresh": fresh}


def render_input(settings: dict):
    with st.form("repo_form"):
        url = st.text_input("GitHub repo URL'si", placeholder="https://github.com/kullanici/repo")
        submitted = st.form_submit_button("✍️ Blog yazısı oluştur", type="primary")

    if not submitted:
        return

    if not url.strip():
        st.warning("Lütfen bir GitHub repo URL'si girin.")
        return

    try:
        ctx = make_ctx(url.strip(), settings)
    except ValueError as e:
        st.error(str(e))
        return

    st.session_state.result = None
    st.session_state.outline_version += 1

    def job(bar):
        outline_data = run_prepare(ctx, bar)
        if settings["review"]:
            return {"outline_data": outline_data, "result": None}
        return {"outline_data": outline_data, "result": run_write(ctx, outline_data, bar)}

    out, err = run_with_status(f"{ctx['owner']}/{ctx['repo']} işleniyor", job)
    if err:
        show_error(err)
        return

    st.session_state.ctx = ctx
    st.session_state.outline_data = out["outline_data"]
    if settings["review"]:
        st.session_state.phase = "review"
    else:
        st.session_state.result = out["result"]
        st.session_state.phase = "done"
    st.rerun()


def render_review():
    ctx = st.session_state.ctx
    outline_data = st.session_state.outline_data
    outline = outline_data["outline"]
    ver = st.session_state.outline_version

    st.subheader(f"📝 Outline incelemesi · {ctx['owner']}/{ctx['repo']}")
    st.caption("Yazı henüz yazılmadı. İskeleti inceleyin; onayladığınızda bölümler bu plana göre yazılır.")

    titles = outline["title_options"]
    idx = titles.index(outline["chosen_title"]) if outline["chosen_title"] in titles else 0
    chosen = st.selectbox("Başlık", titles, index=idx, key=f"title_{ver}")
    if chosen != outline["chosen_title"]:
        outline["chosen_title"] = chosen
        ctx["modified"] = True

    st.markdown(f"**Alt başlık:** {outline['subtitle']}")
    st.markdown(f"**Ana açı:** {outline['core_angle']}")
    st.markdown(f"**Giriş fikri:** {outline['hook']}")

    for i, sec in enumerate(outline["sections"], 1):
        with st.expander(f"{i}. {sec['heading']}  (~{sec['estimated_words']} kelime)"):
            st.markdown(f"**Amaç:** {sec['goal']}")
            for kp in sec["key_points"]:
                st.markdown(f"- {kp}")
            st.caption("Dosyalar: " + (", ".join(sec["source_files"]) or "-"))
            if sec.get("use_snippet_from"):
                st.caption(f"Kod kesiti: {sec['use_snippet_from']}")

    st.markdown(f"**Sonuç fikri:** {outline['conclusion_idea']}")
    st.caption(f"Etiketler: {', '.join(outline['tags'])}")
    st.divider()

    feedback = st.text_area("Geri bildirim (isteğe bağlı)", key=f"fb_{ver}",
                            placeholder="Örn: 5 bölüme indir, auth kısmını öne al, daha az teknik ol")
    c1, c2, c3 = st.columns(3)
    approve = c1.button("✅ Onayla ve yazmaya başla", type="primary")
    revise = c2.button("🔁 Geri bildirimle revize et", disabled=not feedback.strip())
    regen = c3.button("🎲 Sıfırdan üret")

    if revise or regen:
        fb = feedback.strip() if revise else None

        def job(bar):
            return run_regenerate_outline(ctx, outline_data, fb, bar)

        new_data, err = run_with_status("Outline yeniden hazırlanıyor", job)
        if err:
            show_error(err)
            return
        ctx["modified"] = True
        pipeline.save_json(ctx["paths"]["outline"], new_data)
        st.session_state.outline_data = new_data
        st.session_state.outline_version += 1
        st.rerun()

    if approve:
        paths = ctx["paths"]
        if ctx["modified"]:
            remove_files(paths, ["draft", "draft_cache", "meta"])   # eski yazı yeni outline'a uymaz
        pipeline.save_json(paths["outline"], outline_data)

        out, err = run_with_status("Yazı yazılıyor", lambda bar: run_write(ctx, outline_data, bar))
        if err:
            show_error(err)
            return
        st.session_state.result = out
        st.session_state.phase = "done"
        st.rerun()

    if st.button("← Başa dön"):
        st.session_state.phase = "input"
        st.rerun()


def render_result(r: dict):
    meta = r.get("meta", {})
    outline = r.get("outline") or {}

    st.success(f"Blog yazısı hazır · {r.get('repository') or ''}", icon="🎉")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kelime", meta.get("word_count", "-"))
    c2.metric("Okuma süresi", f"{meta.get('estimated_read_minutes', '-')} dk")
    c3.metric("Kod bloğu", meta.get("code_blocks", "-"))
    c4.metric("Bölüm", len(outline.get("sections", [])) or "-")

    title = meta.get("title") or outline.get("chosen_title") or "blog"
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in title)[:60] or "blog"
    st.download_button("⬇️ Markdown olarak indir", r["markdown"], file_name=f"{safe_name}.md",
                       mime="text/markdown", type="primary")

    tab_preview, tab_md, tab_details = st.tabs(["📖 Önizleme", "📝 Markdown", "🔎 Detaylar"])

    with tab_preview:
        st.markdown(r["markdown"])

    with tab_md:
        st.caption("Sağ üstteki simgeyle kopyalayıp Medium editörüne yapıştırabilirsiniz. "
                   "Başlık ve kod bloklarını yapıştırdıktan sonra kontrol edin.")
        st.code(r["markdown"], language="markdown")

    with tab_details:
        if meta.get("tags"):
            st.markdown("**Etiketler:** " + ", ".join(meta["tags"]))
        if meta.get("final_check_issues"):
            st.warning("Son kontrol sorunları:\n\n" + "\n".join(f"- {i}" for i in meta["final_check_issues"]))
        if meta.get("writer_warnings"):
            with st.expander(f"Yazar uyarıları ({len(meta['writer_warnings'])})"):
                for w in meta["writer_warnings"]:
                    st.markdown(f"- {w}")
        if r.get("files"):
            st.markdown("**Analiz edilen dosyalar**")
            st.dataframe(sorted(r["files"], key=lambda x: x.get("Puan") or 0, reverse=True),
                         hide_index=True)
        st.caption(f"Dosya: {r.get('draft_path')}")
        st.info("Yayınlamadan önce yazıyı okuyun: model, analizde olmayan bir iddia ekleyebilir.")

    if st.button("🆕 Yeni repo"):
        st.session_state.phase = "input"
        st.session_state.result = None
        st.rerun()


# =====================================================================
# ANA UYGULAMA
# =====================================================================

def app():
    init_state()
    settings = render_sidebar()

    st.title("✍️ GitHub Repo → Medium Blog Yazısı")
    st.caption("Repo URL'sini girin; kod analiz edilir, iskelet çıkarılır ve yazı bölüm bölüm yazılır. "
               "Kod kesitleri modelin değil, kaynak dosyadan alınır ve doğrulanır.")

    if st.session_state.phase == "review":
        render_review()
        return

    if st.session_state.result:
        render_result(st.session_state.result)
        return

    render_input(settings)


if __name__ == "__main__":
    app()