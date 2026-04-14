from fastapi import FastAPI, UploadFile
import shutil
import cv2
import numpy as np
import easyocr
import re

app = FastAPI()

# 🔥 OCR leve
reader = easyocr.Reader(['en'], gpu=False)

# 🔥 template check
check_template = cv2.imread("imgs/check.jpg", 0)
if check_template is None:
    raise Exception("check.jpg não encontrado em /imgs")


@app.get("/")
def home():
    return {"status": "ok"}


# -------------------------------
# 🔧 UTIL
# -------------------------------

def crop_rel(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    return img[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]


# -------------------------------
# 🗺️ MAPA
# -------------------------------

def detectar_mapa(img):
    left = crop_rel(img, 0.0, 0.0, 0.3, 1.0)
    gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    results = reader.readtext(gray)

    texto = " ".join([r[1].lower() for r in results])

    if "plano" in texto:
        return "Plano Divino"
    elif "zona" in texto:
        return "Zona Proibida"
    elif "sant" in texto:
        return "Santuário"

    return "Desconhecido"


# -------------------------------
# ✔ CHECK
# -------------------------------

def tem_check(img_gray):
    try:
        img_gray = cv2.resize(img_gray, (60, 60))
        template = cv2.resize(check_template, (30, 30))

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        return np.max(res) > 0.5
    except:
        return False


# -------------------------------
# 🔢 PROGRESSO
# -------------------------------

def extrair_progresso(crop):
    try:
        numero_area = crop[70:120, 10:120]

        numero_area = cv2.cvtColor(numero_area, cv2.COLOR_BGR2GRAY)

        results = reader.readtext(numero_area)

        for (_, text, _) in results:
            match = re.search(r'(\d+)/(\d+)', text)
            if match:
                atual = int(match.group(1))
                total = int(match.group(2))

                if atual < total:
                    return atual, total

        return None, None

    except:
        return None, None


# -------------------------------
# 🚀 MAIN
# -------------------------------

@app.post("/analisar")
async def analisar(file: UploadFile):
    try:
        path = f"/tmp/{file.filename}"

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(path)

        if img is None:
            return {"erro": "Imagem inválida"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 🗺️ detectar mapa
        mapa = detectar_mapa(img)

        # 🔥 posições relativas (ajustar fino depois)
        bosses = [
            {"nome": "E-Bolt 5", "x": 0.60, "y": 0.65},
            {"nome": "Centro 1", "x": 0.70, "y": 0.65},
            {"nome": "Direita 1", "x": 0.80, "y": 0.65},
            {"nome": "Topo", "x": 0.70, "y": 0.20},
            {"nome": "Centro Esq", "x": 0.60, "y": 0.40},
            {"nome": "Centro Dir", "x": 0.80, "y": 0.40},
        ]

        resultados = []

        h, w = img.shape[:2]

        for boss in bosses:
            x = int(boss["x"] * w)
            y = int(boss["y"] * h)

            crop = img[y:y+120, x:x+120]

            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            crop_gray = gray[y:y+120, x:x+120]

            if not tem_check(crop_gray):

                atual, total = extrair_progresso(crop)

                progresso = None
                nivel = None

                if atual is not None:
                    progresso = f"{atual}/{total}"
                    nivel = atual + 1

                resultados.append({
                    "boss": boss["nome"],
                    "progresso": progresso,
                    "nivel": nivel
                })

        return {
            "mapa": mapa,
            "faltando": resultados
        }

    except Exception as e:
        return {"erro": str(e)}