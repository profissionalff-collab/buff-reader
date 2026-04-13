from fastapi import FastAPI, UploadFile
import shutil
import cv2
import pytesseract
import re

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)

    # binarização leve
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh

def limpar_texto(texto):
    texto = texto.replace("O", "0")
    texto = texto.replace("I", "1")
    texto = texto.replace("l", "1")
    return texto

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

        # 🔥 corta só região útil (centro)
        img = img[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)]

        # 🔥 reduz resolução (acelera MUITO)
        img = cv2.resize(img, (800, 600))

        proc = preprocess(img)

        # 🔥 OCR leve
        texto = pytesseract.image_to_string(proc)

        texto = limpar_texto(texto)
        linhas = texto.split("\n")

        # 🗺️ MAPA
        texto_full = texto.lower()

        mapa = "Desconhecido"
        if "plano divino" in texto_full:
            mapa = "Plano Divino"
        elif "zona proibida" in texto_full:
            mapa = "Zona Proibida"
        elif "sant" in texto_full:
            mapa = "Santuário"

        # 🎯 ESTÁGIO
        estagio = None
        for i, linha in enumerate(linhas):
            if "progr" in linha.lower():
                if i > 0:
                    match = re.search(r'(\d+)', linhas[i-1])
                    if match:
                        estagio = int(match.group(1))

        # 🔢 PROGRESSO (ignora completos)
        candidatos = []

        for linha in linhas:
            match = re.search(r'(\d+)/(\d+)', linha)
            if match:
                atual = int(match.group(1))
                total = int(match.group(2))

                if atual != total:  # ignora 9/9 etc
                    candidatos.append((atual, total))

        progresso = None
        nivel = None

        if candidatos:
            atual, total = sorted(candidatos, key=lambda x: x[0])[0]
            progresso = f"{atual}/{total}"
            nivel = atual + 1

        return {
            "mapa": mapa,
            "estagio": estagio,
            "progresso": progresso,
            "nivel": nivel
        }

    except Exception as e:
        return {"erro": str(e)}
