from fastapi import FastAPI, UploadFile
import shutil
import easyocr
import cv2
import numpy as np
import re

app = FastAPI()

# 🔥 inicia OCR uma vez só (importantíssimo)
reader = easyocr.Reader(['en'], gpu=False)


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

        # 🔥 corta área útil
        img = img[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)]

        # 🔥 reduz resolução (MUITO importante)
        img = cv2.resize(img, (800, 600))

        proc = preprocess(img)

        # 🔥 OCR
        results = reader.readtext(proc)

        textos = []
        for (_, text, _) in results:
            text = limpar_texto(text)
            textos.append(text)

        # 🔥 filtra só o necessário
        textos = [t for t in textos if "/" in t or "progr" in t.lower()]

        texto_full = " ".join(textos).lower()

        # 🗺️ mapa
        mapa = "Desconhecido"
        if "plano divino" in texto_full:
            mapa = "Plano Divino"
        elif "zona proibida" in texto_full:
            mapa = "Zona Proibida"
        elif "sant" in texto_full:
            mapa = "Santuário"

        # 🎯 estágio
        estagio = None
        for i, text in enumerate(textos):
            if "progr" in text.lower():
                if i > 0:
                    match = re.search(r'(\d+)', textos[i-1])
                    if match:
                        estagio = int(match.group(1))

        # 🔢 progresso válido
        candidatos = []

        for text in textos:
            match = re.search(r'(\d+)/(\d+)', text)
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
