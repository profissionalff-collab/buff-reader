from fastapi import FastAPI, UploadFile
import shutil
import cv2
import numpy as np

app = FastAPI()


@app.get("/")
def home():
    return {"status": "ok"}


def tem_check_azul(img):
    # converte pra HSV (melhor pra cor)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # faixa de azul
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # conta pixels azuis
    azul_pixels = cv2.countNonZero(mask)

    return azul_pixels > 50  # ajuste fino depois


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

        # 🔥 cortar área dos bosses (ajustável)
        area = img[int(h*0.25):int(h*0.75), int(w*0.30):int(w*0.80)]

        # dividir grid 3x3
        rows, cols = 3, 3
        cell_h = area.shape[0] // rows
        cell_w = area.shape[1] // cols

        faltando = []

        for i in range(rows):
            for j in range(cols):
                y1 = i * cell_h
                y2 = (i + 1) * cell_h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w

                cell = area[y1:y2, x1:x2]

                if not tem_check_azul(cell):
                    faltando.append((i, j))

        return {
            "faltando_posicoes": faltando
        }

    except Exception as e:
        return {"erro": str(e)}
