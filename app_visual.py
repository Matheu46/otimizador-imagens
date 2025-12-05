import os
import streamlit as st
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import cv2

# ========================= CONFIG =========================

PASTA_ENTRADA = "./entrada"
os.makedirs(PASTA_ENTRADA, exist_ok=True)

PASTA_SAIDA = "./saida"
os.makedirs(PASTA_SAIDA, exist_ok=True)

st.set_page_config(layout="wide", page_title="Visualizador de Zoom com Detecção de Rosto")
st.title("📸 Visualizador de Zoom + Detecção de Rosto + Upload")
st.markdown("---")

# ========================= FACE DETECTOR =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detectar_rosto(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    faces = face_cascade.detectMultiScale(
        img_cv, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

# ========================= FUNÇÃO DE PROCESSAMENTO =========================

def processar_imagem(img_original, zoom_fator, tamanho_max):
    """Processa 1 imagem igual ao imageMagic.py e retorna o objeto PIL final."""
    img_original = ImageOps.exif_transpose(img_original)

    if img_original.mode != "RGB":
        img_original = img_original.convert("RGB")

    w_orig, h_orig = img_original.size

    # ---------- DETECÇÃO DE ROSTO ----------
    face = detectar_rosto(img_original)

    if face is not None:
        x, y, w, h = face
        cx = x + w / 2
        cy = y + h / 2
    else:
        cx = cy = None

    # ---------- CÁLCULO DO CROP ----------
    janela_w = w_orig / zoom_fator
    janela_h = h_orig / zoom_fator

    if cx is None:
        left = (w_orig - janela_w) / 2
        top = 0
    else:
        left = cx - janela_w / 2
        top = cy - janela_h / 2

    left = max(0, min(left, w_orig - janela_w))
    top = max(0, min(top, h_orig - janela_h))

    right = left + janela_w
    bottom = top + janela_h

    # ---------- CORTE ----------
    img_crop = img_original.crop((left, top, right, bottom))

    # ---------- REDIMENSIONAR ----------
    img_final = img_crop.copy()
    img_final.thumbnail((tamanho_max, tamanho_max), Image.Resampling.LANCZOS)

    return img_final, (left, top, right, bottom), face


# ========================= SIDEBAR =========================

st.sidebar.header("Upload das Imagens")
uploads = st.sidebar.file_uploader(
    "Arraste uma ou várias imagens",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True
)

limpar_pasta = st.sidebar.checkbox(
    "Limpar pasta 'entrada/' antes de salvar uploads",
    value=True
)

st.sidebar.header("Configurações")
tamanho_max = st.sidebar.number_input("Tamanho Máximo (px)", value=1000, step=100)
zoom_fator = st.sidebar.slider("Zoom (-z)", min_value=1.0, max_value=5.0, value=1.0, step=0.1)

processar_agora = st.sidebar.button("🚀 Processar tudo agora")

# ========================= TRATAR UPLOADS =========================

if uploads:
    if limpar_pasta:
        for f in os.listdir(PASTA_ENTRADA):
            try:
                os.remove(os.path.join(PASTA_ENTRADA, f))
            except:
                pass

    for arquivo in uploads:
        with open(os.path.join(PASTA_ENTRADA, arquivo.name), "wb") as f:
            f.write(arquivo.getbuffer())

    st.success(f"✔ {len(uploads)} imagem(ns) salva(s) na pasta 'entrada/'")


# ========================= CARREGAR TODAS AS IMAGENS =========================

arquivos = [f for f in os.listdir(PASTA_ENTRADA)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
arquivos.sort()

if len(arquivos) == 0:
    st.info("Nenhuma imagem na pasta 'entrada/'. Envie arquivos no painel lateral.")
    st.stop()

# ========================= PROCESSAMENTO AUTOMÁTICO =========================

if processar_agora:
    st.warning("Processando imagens... aguarde alguns segundos.")

    for filename in arquivos:
        input_path = os.path.join(PASTA_ENTRADA, filename)
        img_original = Image.open(input_path)

        img_final, _, _ = processar_imagem(img_original, zoom_fator, tamanho_max)

        output_path = os.path.join(PASTA_SAIDA, os.path.splitext(filename)[0] + ".jpg")
        img_final.save(output_path, "JPEG", quality=80, optimize=True)

    st.success(f"✔ Todas as imagens foram processadas e salvas na pasta '{PASTA_SAIDA}'!")
    st.balloons()


# ========================= PREVIEW DAS IMAGENS =========================

st.subheader(f"🖼 {len(arquivos)} imagem(ns) encontrada(s) na pasta 'entrada/'")

for filename in arquivos:
    st.markdown(f"### 📌 {filename}")

    input_path = os.path.join(PASTA_ENTRADA, filename)
    img_original = Image.open(input_path)

    img_final, crop_box, face = processar_imagem(img_original, zoom_fator, tamanho_max)

    left, top, right, bottom = crop_box

    # ---------- PREVIEW ----------
    img_preview = img_original.copy()
    draw = ImageDraw.Draw(img_preview)

    if face is not None:
        x, y, w, h = face
        draw.rectangle([x, y, x + w, y + h], outline="yellow", width=4)

    draw.rectangle([left, top, right, bottom], outline="red", width=4)

    # ---------- EXIBIÇÃO ----------
    col1, col2 = st.columns(2)

    with col1:
        st.caption("Original + Detecção")
        st.image(img_preview, use_column_width=True)

    with col2:
        st.caption(f"Resultado ({img_final.width}x{img_final.height})")
        st.image(img_final, use_column_width=True)

    st.markdown("---")
