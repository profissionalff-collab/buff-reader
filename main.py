from fastapi import FastAPI, UploadFile
import shutil
import cv2
import numpy as np
import os

app = FastAPI()

# 🔥 load templates de bosses
boss_templates = {}
for file in os.listdir("imgs/bosses"):
    name = file.split(".")[0]
    img = cv2.imread(f"imgs/bosses/{file}", 0)
    boss_templates[name] = img

# 🔥 check
check_template = cv2.imread("imgs/check.jpg", 0)


@app.get("/")
def home():
    return {"status": "ok"}


def tem_check(img_gray):
    img_gray = cv2.resize(img_gray, (60, 60))
    template = cv2.resize(check_template, (30, 30))

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    return np.max(res) > 0.5


def identificar_boss(crop_gray):
    best_match = None
    best_score = 0

    for nome, template in boss_templates.items():

        try:
            template = cv2.resize(template, (50, 50))
            crop = cv2.resize(crop_gray, (100, 100))

            res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            score = np.max(res)

            if score > best_score:
                best_score = score
                best_match = nome

        except:
            continue

    if best_score > 0.5:
        return best_match

    return None


@app.post("/analisar")
async def analisar(file: UploadFile):
    try:
        path = f"/tmp/{file.filename}"

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        resultados = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # 🔥 filtra tamanho dos cards
            if 80 < w < 300 and 80 < h < 300:

                crop = img[y:y+h, x:x+w]
                crop_gray = gray[y:y+h, x:x+w]

                if crop.shape[0] == 0:
                    continue

                if not tem_check(crop_gray):

                    boss = identificar_boss(crop_gray)

                    if boss:
                        resultados.append({
                            "boss": boss
                        })

        return {
            "faltando": resultados
        }

    except Exception as e:
        return {"erro": str(e)}