from fastapi import FastAPI, UploadFile
import shutil
import cv2
import numpy as np
import easyocr
import re

app = FastAPI()

# 🔥 OCR leve só pra número
reader = easyocr.Reader(['en'], gpu=False)

# 🔥 template check
check_template = cv2.imread("imgs/check.jpg", 0)

if check_template is None:
    raise Exception("check.jpg não encontrado em /imgs")


@app.get("/")
def home():
    return {"status": "ok"}


def tem_check(img_gray):
    try:
        img_gray = cv2.resize(img_gray, (60, 60))
        template = cv2.resize(check_template, (30, 30))

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        return np.max(res) > 0.5
    except:
        return False


def extrair_progresso(crop):
    try:
        # 🔥 área onde fica o número dentro do card
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

        # 🔥 MAPA DE BOSSES (AJUSTAR UMA VEZ)
        bosses = [
            {"nome": "E-Bolt 5", "x": 750, "y": 450},
            {"nome": "Centro 1", "x": 950, "y": 450},
            {"nome": "Direita 1", "x": 1150, "y": 450},
            {"nome": "Topo", "x": 950, "y": 150},
            {"nome": "Centro Esquerda", "x": 850, "y": 300},
            {"nome": "Centro Direita", "x": 1050, "y": 300},
        ]

        resultados = []

        for boss in bosses:
            x = boss["x"]
            y = boss["y"]

            crop = img[y:y+120, x:x+120]

            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            crop_gray = gray[y:y+120, x:x+120]

            if not tem_check(crop_gray):

                atual, total = extrair_progresso(crop)

                nivel = None
                progresso = None

                if atual is not None:
                    progresso = f"{atual}/{total}"
                    nivel = atual + 1

                resultados.append({
                    "boss": boss["nome"],
                    "progresso": progresso,
                    "nivel": nivel
                })

        return {
            "faltando": resultados
        }

    except Exception as e:
        return {"erro": str(e)}