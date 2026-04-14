from fastapi import FastAPI, UploadFile
import shutil
import cv2
import numpy as np

app = FastAPI()

# 🔥 carrega template (garante que existe)
check_template = cv2.imread("imgs/check.jpg", 0)

if check_template is None:
    raise Exception("check.jpg não encontrado em /imgs")


@app.get("/")
def home():
    return {"status": "ok"}


def tem_check(img_gray):
    try:
        h_img, w_img = img_gray.shape
        h_tmp, w_tmp = check_template.shape

        # 🔥 evita erro do OpenCV
        if h_img < h_tmp or w_img < w_tmp:
            return False

        # 🔥 aplica template matching
        res = cv2.matchTemplate(img_gray, check_template, cv2.TM_CCOEFF_NORMED)
        max_val = np.max(res)

        return max_val > 0.6  # ajuste fino depois

    except:
        return False


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

        # 🔥 COORDENADAS FIXAS (AJUSTAR UMA VEZ)
        bosses = [
            {"nome": "boss_topo_direita", "x": 950, "y": 150},
            {"nome": "boss_centro", "x": 850, "y": 300},
            {"nome": "boss_direita", "x": 1050, "y": 300},
            {"nome": "boss_esquerda", "x": 750, "y": 450},
            {"nome": "boss_meio", "x": 950, "y": 450},
            {"nome": "boss_direita2", "x": 1150, "y": 450},
        ]

        faltando = []

        for boss in bosses:
            x = boss["x"]
            y = boss["y"]

            # 🔥 tamanho do card
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