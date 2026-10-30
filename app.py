import os
import streamlit as st
import requests
from agents.planner_agent import DEFAULT_AUDIENCE, DEFAULT_TONE

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3000")

st.set_page_config(page_title="GitHub to Medium", page_icon="✍️", layout="wide")

# Session state başlatma
if "user" not in st.session_state:
    st.session_state.user = None
if "token" not in st.session_state:
    st.session_state.token = None
if "blog_result" not in st.session_state:
    st.session_state.blog_result = None


def get_auth_headers() -> dict:
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}


def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "unhealthy", "error": f"HTTP {resp.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "unreachable", "error": str(e)}


st.title("🚀 GitHub Reposundan Medium Blog Yazısına")
st.markdown("Bu araç, FastAPI backend (`localhost:3000`) üzerinden yapay zeka ajanlarını kullanarak GitHub kodunuzu analiz eder ve teknik bir Medium yazısı çıkarır.")

# --- YAN MENÜ (SIDEBAR) ---
with st.sidebar:
    # 1. API Durumu
    health = check_api_health()
    if health.get("status") == "healthy":
        st.success("🟢 Backend & DB Bağlantısı Aktif")
    elif health.get("status") == "unhealthy":
        st.warning(f"🟡 Backend açık ancak DB sorunu var: {health.get('error', '')}")
    else:
        st.error("🔴 FastAPI sunucusuna (`localhost:3000`) ulaşılamıyor. Lütfen `python api.py` komutunu çalıştırın.")

    st.divider()

    # 2. Kullanıcı Giriş / Kayıt Alanı
    st.header("👤 Kullanıcı Hesabı")
    if st.session_state.user:
        user_info = st.session_state.user
        st.success(f"Hoş geldin, **{user_info['username']}**!")
        st.caption(f"📧 {user_info['email']}")
        if st.button("🚪 Çıkış Yap", use_container_width=True):
            st.session_state.user = None
            st.session_state.token = None
            st.rerun()
    else:
        auth_tab1, auth_tab2 = st.tabs(["Giriş Yap", "Kayıt Ol"])

        with auth_tab1:
            with st.form("login_form"):
                login_email = st.text_input("E-posta")
                login_password = st.text_input("Şifre", type="password")
                login_submit = st.form_submit_button("Giriş Yap", use_container_width=True)

                if login_submit:
                    if not login_email or not login_password:
                        st.warning("Lütfen e-posta ve şifrenizi girin.")
                    else:
                        try:
                            res = requests.post(
                                f"{API_BASE_URL}/login",
                                json={"email": login_email, "password": login_password},
                                timeout=10,
                            )
                            if res.status_code == 200:
                                body = res.json()
                                st.session_state.user = body["user"]
                                st.session_state.token = body.get("access_token")
                                st.toast("Giriş başarılı! 🎉")
                                st.rerun()
                            else:
                                detail = res.json().get("detail", "Giriş başarısız.")
                                st.error(detail)
                        except requests.exceptions.RequestException as e:
                            st.error(f"Sunucu hatası: {e}")

        with auth_tab2:
            with st.form("signup_form"):
                signup_username = st.text_input("Kullanıcı Adı")
                signup_email = st.text_input("E-posta")
                signup_password = st.text_input("Şifre (en az 6 karakter)", type="password")
                signup_submit = st.form_submit_button("Kayıt Ol", use_container_width=True)

                if signup_submit:
                    if not signup_username or not signup_email or not signup_password:
                        st.warning("Lütfen tüm alanları doldurun.")
                    else:
                        try:
                            res = requests.post(
                                f"{API_BASE_URL}/signup",
                                json={
                                    "username": signup_username,
                                    "email": signup_email,
                                    "password": signup_password,
                                },
                                timeout=10,
                            )
                            if res.status_code in (200, 201):
                                body = res.json()
                                st.session_state.user = body["user"]
                                st.session_state.token = body.get("access_token")
                                st.toast("Kayıt başarılı ve giriş yapıldı! 🎉")
                                st.rerun()
                            else:
                                detail = res.json().get("detail", "Kayıt başarısız.")
                                st.error(detail)
                        except requests.exceptions.RequestException as e:
                            st.error(f"Sunucu hatası: {e}")

    st.divider()

    # 3. Blog Üretim Ayarları
    st.header("⚙️ Blog Ayarları")
    repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/owner/repo")
    focus = st.text_input(
        "Odak Alanı",
        placeholder="Boş bırakırsanız yapay zeka belirler",
        help="Örn: State Yönetimi, AI Ajanları, Performans vb.",
    )
    audience = st.text_input("Hedef Kitle", value=DEFAULT_AUDIENCE)
    tone = st.text_input("Yazı Tonu", value=DEFAULT_TONE)
    workers = st.slider("Paralel Analiz (Worker)", min_value=1, max_value=10, value=3)

    start_button = st.button("✨ Blog Üretimini Başlat", type="primary", use_container_width=True)


# --- ANA AKIŞ (API İSTEĞİ) ---
if start_button:
    if not repo_url.strip():
        st.error("Lütfen geçerli bir GitHub Repo URL'si girin.")
        st.stop()

    payload = {
        "repo_url": repo_url.strip(),
        "user_id": st.session_state.user["user_id"] if st.session_state.user else None,
        "focus_area": focus.strip() if focus.strip() else None,
        "audience": audience,
        "tone": tone,
        "workers": workers,
    }

    with st.spinner("🚀 FastAPI backend üzerinde repo indiriliyor, analiz ediliyor ve blog yazılıyor... (Bu işlem birkaç dakika sürebilir)"):
        try:
            response = requests.post(
                f"{API_BASE_URL}/write",
                json=payload,
                headers=get_auth_headers(),
                timeout=900,
            )
            if response.status_code == 200:
                st.session_state.blog_result = response.json()
                st.toast("Blog yazısı başarıyla üretildi! 🎉")
            else:
                try:
                    err_detail = response.json().get("detail", response.text)
                except Exception:
                    err_detail = response.text
                st.error(f"API Hatası ({response.status_code}): {err_detail}")
        except requests.exceptions.ConnectionError:
            st.error("FastAPI sunucusuna (`http://localhost:3000`) bağlanılamadı. Sunucunun çalıştığından emin olun.")
        except requests.exceptions.Timeout:
            st.error("İstek zaman aşımına uğradı.")
        except Exception as e:
            st.error(f"Beklenmeyen bir hata oluştu: {e}")


# --- SONUÇLARIN GÖSTERİMİ ---
if st.session_state.blog_result:
    data = st.session_state.blog_result
    outline_data = data.get("outline", {})
    analysis_data = data.get("analysis", {})
    final_markdown = data.get("markdown_content", "")
    repo_name = data.get("repo_name", "repo")

    tab_result, tab_outline, tab_analysis = st.tabs([
        "🎉 1. Makale (Sonuç)",
        "📝 2. Blog İskeleti (Outline)",
        "🔍 3. Analiz Özeti",
    ])

    with tab_result:
        st.subheader(data.get("title") or "🎉 Makaleniz Hazır!")
        if data.get("subtitle"):
            st.caption(data["subtitle"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Odak Alanı", data.get("focus_area") or "-")
        col2.metric("Tahmini Okuma", f"~{data.get('estimated_read_minutes', '-')} dk")
        col3.metric("Kayıt ID (DB)", f"Repo #{data.get('repo_id')} / Blog #{data.get('blog_id')}")

        if data.get("tags"):
            st.markdown("**Etiketler:** " + ", ".join(f"`{t}`" for t in data["tags"]))

        if data.get("warnings"):
            with st.expander("⚠️ Uyarılar ve Kontroller"):
                for w in data["warnings"]:
                    st.write(f"- {w}")

        st.download_button(
            label="📥 Markdown (MD) Olarak İndir",
            data=final_markdown,
            file_name=f"{repo_name}_blog.md",
            mime="text/markdown",
            type="primary",
        )

        st.markdown("---")
        st.markdown(final_markdown, unsafe_allow_html=True)

    with tab_outline:
        st.subheader("Blog İskeleti (Outline)")
        st.info(f"🎯 **Seçilen Odak Alanı:** {outline_data.get('selected_focus_area', '-')}")
        if outline_data.get("recommended_focus_areas"):
            st.caption(f"💡 **Alternatif Odaklar:** {', '.join(outline_data['recommended_focus_areas'])}")

        st.write(f"**Seçilen Başlık:** {outline_data.get('chosen_title', '-')}")
        st.write(f"**Alt Başlık:** {outline_data.get('subtitle', '-')}")

        for i, sec in enumerate(outline_data.get("sections", []), 1):
            with st.expander(f"Bölüm {i}: {sec.get('heading', '')}"):
                st.write(f"**Amaç:** {sec.get('goal', '')}")
                st.write(f"**Hedef Dosyalar:** {', '.join(sec.get('source_files', []))}")
                if sec.get("use_snippet_from"):
                    st.code(f"Kod Kesiti: {sec['use_snippet_from']}")

    with tab_analysis:
        st.subheader("Kod Analiz Özeti")
        if analysis_data:
            st.success(f"✅ Toplam {analysis_data.get('analyzed_files_count', 0)} dosya analiz edildi.")
            if analysis_data.get("languages"):
                st.write("**Diller:** " + ", ".join(analysis_data["languages"]))

            file_analyses = analysis_data.get("file_analyses", [])
            if file_analyses:
                st.markdown("**En Yüksek Puanlı Dosyalar:**")
                st.json([
                    f"{r['path']} - {r.get('architectural_domain', '')} (Puan: {r.get('blog_worthiness_score', 0)})"
                    for r in file_analyses[:10]
                ])
        else:
            st.info("Analiz detayı bulunamadı.")
