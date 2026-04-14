import cv2
import numpy as np
import os

# 🔥 carregar templates de bosses
boss_templates = {}

if os.path.exists("imgs/bosses"):
    for file in os.listdir("imgs/bosses"):
        if file.endswith(".png") or file.endswith(".jpg"):
            name = file.split(".")[0]
            img = cv2.imread(f"imgs/bosses/{file}", 0)
            boss_templates[name] = img


def tem_check(img_gray, check_template):
    """Verifica check"""
    img_gray = cv2.resize(img_gray, (60, 60))
    template = cv2.resize(check_template, (30, 30))

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    return np.max(res) > 0.5


def identificar_boss(crop_gray):
    """Identifica qual boss é via template matching"""
    best_match = None
    best_score = 0

    for nome, template in boss_templates.items():
        try:
            template = cv2.resize(template, (50, 50))
            crop = cv2.resize(crop_gray, (100, 100))

            res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            score = np.max(res)

            if score > best_score:
                best_score = score
                best_match = nome

        except:
            continue

    if best_score > 0.5:
        return best_match

    return None