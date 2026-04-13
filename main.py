from fastapi import FastAPI, UploadFile
import shutil
import easyocr
import cv2
import numpy as np
import re
import os

app = FastAPI()

# inicia OCR uma vez só (IMPORTANTE pra performance)
reader = easyocr.Reader(['en'])

@app.get("/")
def home():
    return {"status": "ok"}

@app.post("/analisar")
async def analisar(file: UploadFile):
    try:
        # salvar imagem temporária
        temp_path = f"/tmp/{file.filename}"

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # carregar imagem
        img = cv2.imread(temp_path)

        if img is None:
            return {"erro": "Imagem inválida"}

        # 🔥 OCR
        results = reader.readtext(img)

        texto_detectado = []
        progresso = None

        for (bbox, text, prob) in results:
            texto_detectado.append(text)

            # procura padrão tipo 0/6, 1/6 etc
            match = re.search(r'(\d)/(\d)', text)
            if match:
                atual = int(match.group(1))
                total = int(match.group(2))
                progresso = f"{atual}/{total}"

        # 🎯 converter progresso em nível
        nivel = None
        if progresso:
            atual, total = map(int, progresso.split("/"))
            nivel = atual + 1  # regra: 0/6 = nível 1

        # 🧠 (placeholder futuro)
        mapa = "Desconhecido"
        chefe = "Desconhecido"

        return {
            "mapa": mapa,
            "chefe": chefe,
            "progresso": progresso,
            "nivel": nivel,
            "texto_detectado": texto_detectado
        }

    except Exception as e:
        return {"erro": str(e)}
