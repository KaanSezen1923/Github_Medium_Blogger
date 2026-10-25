import json
import os
import streamlit as st

# Sayfa yapılandırması
st.set_page_config(
    page_title="Süt Sihirbazı - Proje Görselleştirici",
    page_icon="🥛",
    layout="wide",
)


@st.cache_data
def load_repo_data(json_file_path: str):
  """JSON dosyasını önbelleğe alarak yükler."""
  if not os.path.exists(json_file_path):
    return None
  with open(json_file_path, "r", encoding="utf-8") as f:
    return json.load(f)


# Veri dosyasının adı
json_filename = "Cinewave_React_Website_code_contents.json"
data = load_repo_data(json_filename)

if data is None:
  st.error(
      f"❌ '{json_filename}' dosyası bulunamadı! Lütfen aynı dizinde olduğundan"
      " emin olun."
  )
  st.stop()

# --- BAŞLIK VE GENEL BİLGİLER ---
st.title("🥛 Süt Sihirbazı - Proje Analiz ve Görselleştirme Paneli")
st.markdown(
    f"**GitHub Reposu:** `{data.get('repository', 'Bilinmiyor')}` | **Yıldız"
    f" Sayısı:** ⭐ {data.get('stargazers_count', 0)}"
)

if data.get("description"):
  st.info(f"**Açıklama:** {data['description']}")

st.markdown("---")

# --- KENAR ÇUCUĞU (SIDEBAR) FİLTRELERİ ---
st.sidebar.header("🔍 Filtreleme ve Arama")

# Dil seçimi filtresi
languages = data.get("languages", [])
selected_lang = st.sidebar.selectbox(
    "Programlama Diline Göre Filtrele", ["Tümü"] + languages
)

# Dosya adı / yolunda arama
search_query = st.sidebar.text_input("Dosya Adında Ara", "").lower()

# Dosyaları işleme
files = data.get("files", [])

# Filtreleme mantığı
filtered_files = []
for file in files:
  path = file.get("path", "")
  ext = os.path.splitext(path)[1].lower()

  # Dil filtresi (Basit bir uzantı eşleme mantığı)
  if selected_lang != "Tümü":
    lang_lower = selected_lang.lower()
    if lang_lower == "python" and ext != ".py":
      continue
    elif lang_lower in ["typescript", "javascript"] and ext not in [
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
    ]:
      continue
    elif lang_lower == "dockerfile" and "dockerfile" not in path.lower():
      continue

  # Arama filtresi
  if search_query and search_query not in path.lower():
    continue

  filtered_files.append(file)

# --- ANA PANEL: SEKME YAPISI ---
tab1, tab2, tab3 = st.tabs(
    ["📁 Dosya İçerikleri", "📊 İstatistikler & Özet", "📄 JSON Veri İnceleme"]
)

with tab1:
  st.subheader(f"Dosya Gezgini (Gösterilen: {len(filtered_files)} dosya)")

  if not filtered_files:
    st.warning("Seçilen kriterlere uygun dosya bulunamadı.")
  else:
    # Dosya yollarını seçmek için selectbox veya liste
    file_paths = [f["path"] for f in filtered_files]
    selected_path = st.selectbox("İncelemek istediğiniz dosyayı seçin:", file_paths)

    # Seçilen dosyanın detayını bul
    selected_file = next(
        (f for f in filtered_files if f["path"] == selected_path), None
    )

    if selected_file:
      col1, col2 = st.columns([3, 1])
      with col1:
        st.markdown(f"**Dosya Yolu:** `{selected_file['path']}`")
      with col2:
        st.markdown(
            f"**Boyut:** `{selected_file['size']} bytes` | **SHA:**"
            f" `{selected_file['sha'][:7]}`"
        )

      # Uzantıya göre dil tespiti (Streamlit st.code için)
      ext = os.path.splitext(selected_file["path"])[1].lower()
      lang_map = {
          ".py": "python",
          ".js": "javascript",
          ".ts": "typescript",
          ".tsx": "tsx",
          ".jsx": "jsx",
          ".json": "json",
          ".yml": "yaml",
          ".yaml": "yaml",
          ".sql": "sql",
          ".Dockerfile": "dockerfile",
      }
      code_lang = lang_map.get(ext, "text")
      if "dockerfile" in selected_file["path"].lower():
        code_lang = "dockerfile"

      st.code(selected_file["content"], language=code_lang, line_numbers=True)

with tab2:
  st.subheader("📈 Proje İstatistikleri")

  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric(
        label="Toplam Dosya Sayısı", value=len(files)
    )  #
  with col2:
    total_size = sum(f.get("size", 0) for f in files)
    st.metric(label="Toplam Kod Boyutu", value=f"{total_size / 1024:.2f} KB")
  with col3:
    st.metric(label="Kullanılan Dil Sayısı", value=len(languages))

  st.markdown("---")
  st.markdown("### 🗂️ Kullanılan Teknolojiler / Diller")
  if languages:
    st.write(languages)

  # Klasör dağılımı analizi
  st.markdown("### 📂 Klasör Dağılımı")
  dir_counts = {}
  for f in files:
    parts = f["path"].split("/")
    folder = parts[0] if len(parts) > 1 else "Kök Dizin (Root)"
    dir_counts[folder] = dir_counts.get(folder, 0) + 1

  st.bar_chart(dir_counts)

with tab3:
  st.subheader("🔍 Ham JSON Veri Yapısı")
  st.markdown(
      "Aşağıdaki alandan JSON içerisindeki anahtarları ve yapısal verileri"
      " inceleyebilirsiniz."
  )
  if st.checkbox("Ham JSON Gövdesini Göster"):
    st.json(data)