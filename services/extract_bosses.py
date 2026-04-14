import cv2
import numpy as np
import zipfile
import os

# 🔥 carrega template do check
check_template = cv2.imread("imgs/check.jpg", 0)


def tem_check(img_gray):
    """Verifica se o boss está concluído (tem check azul)"""
    try:
        img_gray = cv2.resize(img_gray, (60, 60))
        template = cv2.resize(check_template, (30, 30))

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        return np.max(res) > 0.5
    except:
        return False


def extrair_bosses(path_img):
    """
    Recebe imagem, detecta bosses SEM check e retorna caminho do zip
    """

    img = cv2.imread(path_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 DETECÇÃO ROBUSTA (resolve Nyx)
    edges = cv2.Canny(gray, 50, 150)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    combined = cv2.bitwise_or(edges, thresh)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boss_files = []
    count = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        ratio = w / h

        # 🔥 filtro leve (não matar boss)
        if 80 < w < 300 and 80 < h < 300 and 0.7 < ratio < 1.3:

            crop = img[y:y+h, x:x+w]
            crop_gray = gray[y:y+h, x:x+w]

            if crop.shape[0] == 0:
                continue

            # 🔥 pega só bosses NÃO concluídos
            if not tem_check(crop_gray):

                filename = f"/tmp/boss_{count}.png"
                cv2.imwrite(filename, crop)

                boss_files.append(filename)
                count += 1

    # 🔥 cria zip
    zip_path = "/tmp/bosses.zip"

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in boss_files:
            zipf.write(f, os.path.basename(f))

    return zip_path