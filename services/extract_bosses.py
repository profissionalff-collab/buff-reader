import cv2
import numpy as np
import zipfile
import os

# 🔥 carregar template do check
check_template = cv2.imread("imgs/check.jpg", 0)

if check_template is None:
    raise Exception("check.jpg não encontrado em imgs/check.jpg")


def tem_check(img_gray):
    """
    Verifica se o boss está concluído (tem check azul)
    """
    try:
        img_gray = cv2.resize(img_gray, (60, 60))
        template = cv2.resize(check_template, (30, 30))

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        return np.max(res) > 0.5
    except:
        return False


def extrair_bosses(path_img):
    """
    Recebe uma imagem do jogo e:
    - Detecta os cards dos bosses
    - Remove os que já têm check
    - Retorna um ZIP com os bosses restantes
    """

    img = cv2.imread(path_img)

    if img is None:
        raise Exception("Imagem inválida")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 leve blur melhora contorno
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 🔥 CANNY AJUSTADO (funciona melhor pro seu caso)
    edges = cv2.Canny(gray, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boss_files = []
    count = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        ratio = w / h

        # 🔥 filtro leve (não matar boss tipo Nyx)
        if 80 < w < 300 and 80 < h < 300 and 0.7 < ratio < 1.3:

            crop = img[y:y+h, x:x+w]
            crop_gray = gray[y:y+h, x:x+w]

            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            # 🔥 ignorar bosses já concluídos
            if not tem_check(crop_gray):

                filename = f"/tmp/boss_{count}.png"
                cv2.imwrite(filename, crop)

                boss_files.append(filename)
                count += 1

    # 🔥 criar zip com os bosses encontrados
    zip_path = "/tmp/bosses.zip"

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in boss_files:
            zipf.write(f, os.path.basename(f))

    print(f"Bosses extraídos: {count}")

    return zip_path