import pytesseract
import cv2 as cv

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

def killed_enemies(screen):
    #enemy_templates = load_enemy_templates()

    killed_enemies = 0
    #for enemy_template in enemy_templates:
    #    matches = template_matches(screen, enemy_template)
    #    killed_enemies += len(matches)

    return killed_enemies

def template_matches(screen, enemy_template):
    pass

def shop_side_room(screen):
    #shop_template = load_shop_template()
    #side_room_template = load_side_room_template()

    #shop_matches = template_matches(screen, shop_template)
    #side_room_matches = template_matches(screen, side_room_template)

    #return len(shop_matches) > 0 or len(side_room_matches) > 0
    pass

# Loaders
def load_shop_template():
    pass

def load_side_room_template():
    pass

def load_enemy_templates():
    pass