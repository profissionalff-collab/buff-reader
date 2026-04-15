import cv2
import sys
import numpy as np

img_path = sys.argv[1]
template_path = sys.argv[2]

img = cv2.imread(img_path, 0)
template = cv2.imread(template_path, 0)

res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
loc = np.where(res >= threshold)

print(len(loc[0]))  # quantidade encontrada
