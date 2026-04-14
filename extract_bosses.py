import cv2
import numpy as np
import os

# cria pasta se não existir
os.makedirs("imgs/bosses", exist_ok=True)

# carrega template do check
check_template = cv2.imread("imgs/check.jpg", 0)


def tem_check(img_gray):
    try:
        img_gray = cv2.resize(img_gray, (60, 60))
        template = cv2.resize(check_template, (30, 30))

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        return np.max(res) > 0.5
    except:
        return False


def extrair_bosses(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # filtrar tamanho do card
        if 80 < w < 300 and 80 < h < 300:

            crop = img[y:y+h, x:x+w]
            crop_gray = gray[y:y+h, x:x+w]

            if crop.shape[0] == 0:
                continue

            # 🔥 pega só bosses SEM check
            if not tem_check(crop_gray):

                # 🔥 salva imagem
                filename = f"imgs/bosses/boss_{count}.png"
                cv2.imwrite(filename, crop)

                print(f"Salvo: {filename}")
                count += 1

    print(f"\nTotal extraído: {count}")


# ---------------------------
# USO
# ---------------------------

extrair_bosses("print.png")