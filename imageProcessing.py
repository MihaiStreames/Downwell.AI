import pytesseract
import cv2 as cv

# -------PATHS---------
path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = path
# ---------------------

def extract_hp(screen):
    vals = {'44': 4, '34': 3, '5': 2, '14': 1}

    hp_area = screen[49:67, 30:194]
    hp_area = cv.cvtColor(hp_area, cv.COLOR_BGR2GRAY)
    hp_area = cv.threshold(hp_area, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    hp_area = cv.resize(hp_area, (0, 0), fx=2, fy=2)

    hp = pytesseract.image_to_string(hp_area, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
    hp = hp.replace(' ', '').replace('\n', '')

    if hp in vals:
        hp = vals[hp]
    else:
        hp = 0

    return hp

def find_player(screen, templates):
    screen_gray = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
    max_val = -1
    max_loc = None
    best_template = None

    for template in templates:
        result = cv.matchTemplate(screen_gray, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val_current, _, max_loc_current = cv.minMaxLoc(result)

        if max_val_current > max_val:
            max_val = max_val_current
            max_loc = max_loc_current
            best_template = template

    top_left = max_loc
    bottom_right = (top_left[0] + best_template.shape[1], top_left[1] + best_template.shape[0])

    return top_left, bottom_right