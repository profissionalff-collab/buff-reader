from fastapi import FastAPI, UploadFile
import shutil
import easyocr
import cv2
import numpy as np
import re

app = FastAPI()

reader = easyocr.Reader(['en'])

@app.get("/")
def home():
    return {"status": "ok"}

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def corrigir_texto(texto):
    texto = texto.replace("O", "0")
    texto = texto.replace("I", "1")
    texto = texto.replace("l", "1")
    texto = texto.replace("00", "%")  # tentativa simples %
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

        # 🔥 preprocess
        proc = preprocess(img)

        # 🔍 OCR
        results = reader.readtext(proc)

        textos = []
        for (_, text, _) in results:
            text = corrigir_texto(text)
            textos.append(text)

        texto_full = " ".join(textos).lower()

        # 🗺️ MAPA
        mapa = "Desconhecido"
        if "plano divino" in texto_full:
            mapa = "Plano Divino"
        elif "zona proibida" in texto_full:
            mapa = "Zona Proibida"
        elif "sant" in texto_full:
            mapa = "Santuário"

        # 🎯 ESTÁGIO (baseado em "Progr")
        estagio = None
        for i, text in enumerate(textos):
            if "progr" in text.lower():
                if i > 0:
                    anterior = textos[i - 1]
                    match = re.search(r'(\d+)', anterior)
                    if match:
                        estagio = int(match.group(1))

        # 🔢 PROGRESSO (ignora completos)
        candidatos = []

        for text in textos:
            match = re.search(r'(\d+)/(\d+)', text)
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
            "nivel": nivel,
            "texto_detectado": textos
        }

    except Exception as e:
        return {"erro": str(e)}
