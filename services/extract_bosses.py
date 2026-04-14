import cv2
import numpy as np
import zipfile

check_template = cv2.imread("imgs/check.jpg", 0)


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

    # 🔥 tamanho do card (ajustável)
    step = 120

    for y in range(0, h - step, 40):   # overlap ajuda MUITO
        for x in range(0, w - step, 40):

            crop = img[y:y+step, x:x+step]
            crop_gray = gray[y:y+step, x:x+step]

            if crop.shape[0] != step or crop.shape[1] != step:
                continue

            # 🔥 ignora os já completos
            if tem_check(crop_gray):
                continue

            # 🔥 filtro leve pra evitar lixo
            std = np.std(crop_gray)
            if std < 15:
                continue

            filename = f"/tmp/boss_{count}.png"
            cv2.imwrite(filename, crop)

            boss_files.append(filename)
            count += 1

    # 🔥 remove duplicados (muito importante)
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

    # 🔥 zip final
    zip_path = "/tmp/bosses.zip"

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in unique_files:
            zipf.write(f, f.split("/")[-1])

    return zip_path