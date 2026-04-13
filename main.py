from fastapi import FastAPI, UploadFile
import shutil
import cv2
import pytesseract
import re

# garante path do tesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok"}


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)
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

        # corta área útil
        img = img[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)]

        # reduz resolução
        img = cv2.resize(img, (800, 600))

        proc = preprocess(img)

        texto = pytesseract.image_to_string(proc, config="--psm 6")
        texto = limpar_texto(texto)

        linhas = texto.split("\n")
        linhas = [l for l in linhas if "/" in l or "progr" in l.lower()]

        texto_full = " ".join(linhas).lower()

        # mapa
        mapa = "Desconhecido"
        if "plano divino" in texto_full:
            mapa = "Plano Divino"
        elif "zona proibida" in texto_full:
            mapa = "Zona Proibida"
        elif "sant" in texto_full:
            mapa = "Santuário"

        # estágio
        estagio = None
        for i, linha in enumerate(linhas):
            if "progr" in linha.lower():
                if i > 0:
                    match = re.search(r'(\d+)', linhas[i-1])
                    if match:
                        estagio = int(match.group(1))

        # progresso válido
        candidatos = []

        for linha in linhas:
            match = re.search(r'(\d+)/(\d+)', linha)
            if match:
                atual = int(match.group(1))
                total = int(match.group(2))

                if atual < total:
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
