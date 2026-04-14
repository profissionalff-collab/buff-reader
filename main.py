from fastapi import FastAPI, UploadFile
import shutil
import cv2
import numpy as np

app = FastAPI()

# 🔥 carrega template do check
check_template = cv2.imread("imgs/check.jpg", 0)


@app.get("/")
def home():
    return {"status": "ok"}


def tem_check(img_gray):
    res = cv2.matchTemplate(img_gray, check_template, cv2.TM_CCOEFF_NORMED)
    return np.max(res) > 0.6  # ajuste fino


@app.post("/analisar")
async def analisar(file: UploadFile):
    try:
        path = f"/tmp/{file.filename}"

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 🔥 COORDENADAS FIXAS (ajustar 1x só)
        bosses = [
            {"nome": "boss1", "x": 900, "y": 150},
            {"nome": "boss2", "x": 800, "y": 300},
            {"nome": "boss3", "x": 1000, "y": 300},
            {"nome": "boss4", "x": 700, "y": 450},
            {"nome": "boss5", "x": 900, "y": 450},
            {"nome": "boss6", "x": 1100, "y": 450},
        ]

        faltando = []

        for boss in bosses:
            x = boss["x"]
            y = boss["y"]

            # tamanho fixo do card
            crop = gray[y:y+120, x:x+120]

            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            if not tem_check(crop):
                faltando.append(boss["nome"])

        return {
            "faltando": faltando
        }

    except Exception as e:
        return {"erro": str(e)}