import pytesseract
import cv2 as cv

# -------PATHS---------
#path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = path
# ---------------------

def extract_info(screen):
    hp = extract_hp(screen)
    gems = extract_gems(screen)

    return hp, gems

def extract_hp(screen):
    hp_vals = {'44': 4, '34': 3, '5': 2, '14': 1}

    hp_area = screen[49:67, 30:194]
    hp_area = cv.cvtColor(hp_area, cv.COLOR_BGR2GRAY)
    hp_area = cv.threshold(hp_area, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    hp_area = cv.resize(hp_area, (0, 0), fx=2, fy=2)

    hp = pytesseract.image_to_string(hp_area, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
    hp = hp.replace(' ', '').replace('\n', '')

    if hp in hp_vals:
        hp = hp_vals[hp]
    else:
        hp = 0

    return hp

def extract_gems(screen):
    gems_area = screen[49:67, 225:300]
    gems_area = cv.cvtColor(gems_area, cv.COLOR_BGR2GRAY)
    gems_area = cv.threshold(gems_area, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    gems_area = cv.resize(gems_area, (0, 0), fx=2, fy=2)

    gems = pytesseract.image_to_string(gems_area, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
    gems = gems.replace(' ', '').replace('\n', '')

    try:
        gems = int(gems)
    except ValueError:
        gems = 0

    return gems