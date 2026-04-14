from fastapi import FastAPI, UploadFile
import shutil
import easyocr
import cv2
import re

app = FastAPI()

reader = easyocr.Reader(['en'], gpu=False)


@app.get("/")
def home():
    return {"status": "ok"}


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # melhora contraste
    gray = cv2.convertScaleAbs(gray, alpha=2, beta=0)

    # blur leve
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold adaptativo (melhor)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

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

        img_original = cv2.imread(path)

        if img_original is None:
            return {"erro": "Imagem inválida"}

        h, w = img_original.shape[:2]

        # 🔥 CORTE SEGURO (não perde info)
        img = img_original[int(h*0.20):int(h*0.70), int(w*0.30):int(w*0.80)]

        # 🔥 resize (performance)
        img = cv2.resize(img, (800, 600))

        proc = preprocess(img)

        # 🔥 DEBUG (opcional, mas útil)
        cv2.imwrite("/tmp/debug_crop.png", img)
        cv2.imwrite("/tmp/debug_proc.png", proc)

        # 🔍 OCR
        results = reader.readtext(proc)

        textos = []

        for (_, text, prob) in results:
            if prob < 0.3:
                continue

            text = limpar_texto(text)

            if len(text) < 3:
                continue

            textos.append(text)

        # 🔥 filtra só relevantes
        textos = [t for t in textos if "/" in t or "progr" in t.lower()]

        texto_full = " ".join(textos).lower()

        # 🗺️ MAPA
        mapa = "Desconhecido"
        if "plano divino" in texto_full:
            mapa = "Plano Divino"
        elif "zona proibida" in texto_full:
            mapa = "Zona Proibida"
        elif "sant" in texto_full:
            mapa = "Santuário"

        # 🎯 ESTÁGIO
        estagio = None
        for i, text in enumerate(textos):
            if "progr" in text.lower():
                if i > 0:
                    match = re.search(r'(\d+)', textos[i-1])
                    if match:
                        estagio = int(match.group(1))

        # 🔢 PROGRESSO (robusto)
        candidatos = []

        for text in textos:
            matches = re.findall(r'(\d+)/(6|7|9)', text)

            for m in matches:
                try:
                    atual = int(m[0])
                    total = int(m[1])

                    if atual < total:
                        candidatos.append((atual, total))
                except:
                    pass

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
            "debug_textos": textos
        }

    except Exception as e:
        return {"erro": str(e)}
