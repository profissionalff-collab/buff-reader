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

    # 🔥 tamanho do card (ajusta se precisar)
    step = 120

    # 🔥 varredura (grid)
    for y in range(0, h - step, 40):
        for x in range(0, w - step, 40):

            crop = img[y:y+step, x:x+step]
            crop_gray = gray[y:y+step, x:x+step]

            if crop.shape[0] != step or crop.shape[1] != step:
                continue

            # -----------------------
            # 🔥 FILTROS INTELIGENTES
            # -----------------------

            # contraste mínimo (remove fundo)
            std = np.std(crop_gray)
            if std < 25:
                continue

            # densidade de borda (remove linha)
            edges = cv2.Canny(crop_gray, 50, 150)
            edge_density = np.sum(edges > 0) / crop_gray.size
            if edge_density < 0.02:
                continue

            # brilho médio (remove UI vazia)
            mean_color = np.mean(crop)
            if mean_color < 40 or mean_color > 200:
                continue

            # -----------------------
            # 🔥 IGNORA COMPLETOS
            # -----------------------
            if tem_check(crop_gray):
                continue

            # -----------------------
            # 🔥 SALVA
            # -----------------------
            filename = f"/tmp/boss_{count}.png"
            cv2.imwrite(filename, crop)

            boss_files.append(filename)
            count += 1

    # -----------------------
    # 🔥 REMOVE DUPLICADOS
    # -----------------------
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

    # -----------------------
    # 🔥 ZIP FINAL
    # -----------------------
    zip_path = "/tmp/bosses.zip"

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for f in unique_files:
            zipf.write(f, f.split("/")[-1])

    return zip_path