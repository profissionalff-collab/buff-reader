import cv2
import numpy as np
import zipfile

# 🔥 template do check
check_template = cv2.imread("imgs/check.jpg", 0)

if check_template is None:
    raise Exception("check.jpg não encontrado em imgs/check.jpg")


def tem_check(img_gray):
    img_gray = cv2.resize(img_gray, (60, 60))
    template = cv2.resize(check_template, (30, 30))

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    return np.max(res) > 0.5


def extrair_bosses(path_img):
    img = cv2.imread(path_img)

    if img is None:
        raise Exception("Imagem inválida")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    boss_files = []
    count = 0

    # -------------------------------
    # 🔥 PASSADA 1 — CONTORNO NORMAL
    # -------------------------------
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray_blur, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w2, h2 = cv2.boundingRect(cnt)
        ratio = w2 / h2

        if 80 < w2 < 300 and 80 < h2 < 300 and 0.7 < ratio < 1.3:

            crop = img[y:y+h2, x:x+w2]
            crop_gray = gray[y:y+h2, x:x+w2]

            if crop.shape[0] == 0:
                continue

            if not tem_check(crop_gray):
                filename = f"/tmp/boss_{count}.png"
                cv2.imwrite(filename, crop)

                boss_files.append(filename)
                count += 1

    # --------------------------------------
    # 🔥 PASSADA 2 — TOPO (RESOLVE NYX)
    # --------------------------------------
    top_img = img[0:int(h * 0.35), 0:w]
    top_gray = gray[0:int(h * 0.35), 0:w]

    top_blur = cv2.GaussianBlur(top_gray, (3, 3), 0)
    edges_top = cv2.Canny(top_blur, 30, 100)

    contours_top, _ = cv2.findContours(edges_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours_top:
        x, y, w2, h2 = cv2.boundingRect(cnt)
        ratio = w2 / h2

        if 80 < w2 < 300 and 80 < h2 < 300 and 0.7 < ratio < 1.3:

            crop = top_img[y:y+h2, x:x+w2]
            crop_gray = top_gray[y:y+h2, x:x+w2]

            if crop.shape[0] == 0:
                continue

            if not tem_check(crop_gray):
                filename = f"/tmp/boss_top_{count}.png"
                cv2.imwrite(filename, crop)

                boss_files.append(filename)
                count += 1

    # --------------------------------------
    # 🔥 REMOVE DUPLICADOS
    # --------------------------------------
    unique_files = []

    for f in boss_files:
        img1 = cv2.imread(f, 0)
        is_duplicate = False

        for uf in unique_files:
            img2 = cv2.imread(uf, 0)

            diff = cv2.absdiff(img1, img2)
            if np.mean(diff) < 5:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_files.append(f)

    # --------------------------------------
    # 🔥 ZIP FINAL
    # --------------------------------------
    zip_path = "/tmp/bosses.zip"

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in unique_files:
            zipf.write(f, f.split("/")[-1])

    return zip_path