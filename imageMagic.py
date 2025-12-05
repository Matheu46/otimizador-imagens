import os
import argparse
import sys
import numpy as np
from PIL import Image, ImageOps
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detectar_rosto(img_pil):
    """Recebe uma imagem PIL e retorna (cx, cy) do maior rosto encontrado ou None."""
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    faces = face_cascade.detectMultiScale(
        img_cv,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None, None

    # Pega o maior rosto
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

    cx = x + w / 2
    cy = y + h / 2
    return cx, cy


def processar_imagens(pasta_entrada, pasta_saida, tamanho_max, zoom, qualidade, preview):
    if not os.path.exists(pasta_entrada):
        print(f"Erro: Pasta '{pasta_entrada}' não encontrada.")
        return

    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    arquivos = [f for f in os.listdir(pasta_entrada) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    arquivos.sort()

    print(f"--- Processando com ZOOM {zoom}x ---")
    print(f"Tamanho Máximo: {tamanho_max}px (lado maior)")

    for i, filename in enumerate(arquivos):
        input_path = os.path.join(pasta_entrada, filename)
        output_path = os.path.join(pasta_saida, filename)

        try:
            with Image.open(input_path) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode in ("RGBA", "P"): img = img.convert("RGB")

                w_orig, h_orig = img.size
                
                # ========== 1. DETECTAR ROSTO ==========
                rosto_cx, rosto_cy = detectar_rosto(img)

                if rosto_cx is not None:
                    print(f"👤 Rosto detectado em {filename} — ajustando crop.")
                else:
                    print(f"➡ Nenhum rosto encontrado em {filename} — usando crop padrão.")

                # ========== 2. CALCULAR JANELA DE ZOOM ==========
                janela_w = w_orig / zoom
                janela_h = h_orig / zoom

                if rosto_cx is None:
                    # Crop padrão (centralizado horizontal e colado no topo)
                    left = (w_orig - janela_w) / 2
                    top = 0
                else:
                    # Crop centralizado no rosto
                    left = rosto_cx - janela_w / 2
                    top = rosto_cy - janela_h / 2

                # Limitar crop para não ultrapassar a imagem
                left = max(0, min(left, w_orig - janela_w))
                top = max(0, min(top, h_orig - janela_h))

                right = left + janela_w
                bottom = top + janela_h

                # ========== 3. CORTAR ==========
                img = img.crop((left, top, right, bottom))

                # ========== 4. REDIMENSIONAR ==========
                img.thumbnail((tamanho_max, tamanho_max), Image.Resampling.LANCZOS)

                # ========== 5. PREVIEW ==========
                if preview and i == 0:
                    print(f"👁️  Preview ({filename})...")
                    img.show()
                    confirma = input(f">> O Zoom de {zoom}x ficou bom? [s/n]: ").strip().lower()
                    if confirma != 's':
                        print("❌ Cancelado. Ajuste o zoom ou tamanho e tente novamente.")
                        sys.exit()

                # ========== 6. SALVAR ==========
                final_output = os.path.splitext(output_path)[0] + ".jpg"
                img.save(final_output, "JPEG", quality=qualidade, optimize=True)

                if not preview:
                    print(f"✔ {filename} -> {img.size[0]}x{img.size[1]}")

        except Exception as e:
            print(f"✖ Erro em {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Voltamos a ter apenas UM argumento de tamanho
    parser.add_argument("-t", "--tamanho", type=int, required=True, help="Tamanho máximo em pixels (ex: 800)")
    parser.add_argument("-z", "--zoom", type=float, default=1.0, help="Fator de aproximação (1.0 = Original, 2.0 = Metade)")
    parser.add_argument("-q", "--qualidade", type=int, default=80)
    parser.add_argument("-i", "--input", type=str, default="./entrada")
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()
    
    processar_imagens(args.input, "./saida", args.tamanho, args.zoom, args.qualidade, args.preview)