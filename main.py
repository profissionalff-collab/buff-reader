from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import shutil
import cv2
from ultralytics import YOLO

from services.extract_bosses import extrair_bosses
from services.detector import identificar_boss, tem_check

model = YOLO("best.pt")  # seu modelo treinado
app = FastAPI()

# 🔥 carregar template check
check_template = cv2.imread("imgs/check.jpg", 0)


@app.get("/")
def home():
    return {"status": "ok"}


# -------------------------------
# 📦 EXTRAIR BOSSES
# -------------------------------

@app.post("/extrair")
async def extrair(file: UploadFile):
    path = f"/tmp/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    zip_path = extrair_bosses(path)

    return FileResponse(zip_path, filename="bosses.zip")
#
# detectar bosses
#
@app.post("/detect")
async def detect(file: UploadFile):
    path = f"/tmp/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(path)

    bosses = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            bosses.append(name)

    return {
        "bosses": bosses
    }
####
#match bosses com cv2 match
####
app.post('/match', async (req, res) => {
  const { image, template } = req.body;

  // chama script python
  const { exec } = require('child_process');

  exec(`python match.py ${image} ${template}`, (err, stdout) => {
    if (err) return res.status(500).send(err.message);
    res.json({ result: stdout });
  });
});

# -------------------------------
# 🔍 ANALISAR BOSSES
# -------------------------------

@app.post("/analisar")
async def analisar(file: UploadFile):
    path = f"/tmp/{file.filename}"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 mesma lógica do extractor
    edges = cv2.Canny(gray, 50, 150)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    combined = cv2.bitwise_or(edges, thresh)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resultados = []
    ids = set()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        ratio = w / h

        if 80 < w < 300 and 80 < h < 300 and 0.7 < ratio < 1.3:

            crop = img[y:y+h, x:x+w]
            crop_gray = gray[y:y+h, x:x+w]

            if crop.shape[0] == 0:
                continue

            if not tem_check(crop_gray, check_template):

                boss = identificar_boss(crop_gray)

                if boss and boss not in ids:
                    ids.add(boss)
                    resultados.append({"boss": boss})

    return {"faltando": resultados}
