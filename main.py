from fastapi import FastAPI, UploadFile
import shutil
import cv2
import numpy as np

app = FastAPI()


@app.get("/")
def home():
    return {"status": "ok"}


def tem_check_azul(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return cv2.countNonZero(mask) > 80


@app.post("/analisar")
async def analisar(file: UploadFile):
    try:
        path = f"/tmp/{file.filename}"

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = cv2.imread(path)

        if img is None:
            return {"erro": "Imagem inválida"}

        h, w = img.shape[:2]

        # 🔥 cortar área central onde ficam os bosses
        area = img[int(h*0.15):int(h*0.85), int(w*0.25):int(w*0.85)]

        gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)

        # detectar bordas
        edges = cv2.Canny(gray, 50, 150)

        # encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bosses_detectados = []

        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)

            # 🔥 filtrar só caixas do tamanho dos bosses
            if 60 < w_box < 200 and 60 < h_box < 200:

                crop = area[y:y+h_box, x:x+w_box]

                tem_check = tem_check_azul(crop)

                bosses_detectados.append({
                    "x": x,
                    "y": y,
                    "check": tem_check
                })

        # ordenar por posição (top → bottom, left → right)
        bosses_detectados = sorted(bosses_detectados, key=lambda b: (b["y"], b["x"]))

        faltando = [b for b in bosses_detectados if not b["check"]]

        return {
            "total_detectados": len(bosses_detectados),
            "faltando": faltando
        }

    except Exception as e:
        return {"erro": str(e)}
