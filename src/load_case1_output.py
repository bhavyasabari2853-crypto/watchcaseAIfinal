import os
import csv
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PRED_DIR = os.path.join(ROOT, 'runs', 'segment', 'predict')
IMAGE = 'WIN_20260610_09_33_53_Pro_watchcase_predict.png'
PIXEL_CSV = os.path.join(PRED_DIR, 'pixels', 'WIN_20260610_09_33_53_Pro_watchcase_pixels.csv')

def load_centroid_from_csv(csv_path):
    xs = []
    ys = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        # support column names: x,y or col,row
        for r in reader:
            if 'x' in r and 'y' in r and r['x'] and r['y']:
                xs.append(int(float(r['x'])))
                ys.append(int(float(r['y'])))
            elif 'col' in r and 'row' in r and r['col'] and r['row']:
                xs.append(int(float(r['col'])))
                ys.append(int(float(r['row'])))
    if not xs:
        return None
    return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))

def main():
    img_path = os.path.join(PRED_DIR, IMAGE)
    if not os.path.exists(img_path):
        print('Image not found:', img_path)
        return
    if not os.path.exists(PIXEL_CSV):
        print('Pixel CSV not found:', PIXEL_CSV)
        return

    centroid = load_centroid_from_csv(PIXEL_CSV)
    print('Centroid (mask):', centroid)

    # show image (optional)
    try:
        im = Image.open(img_path)
        im.show()
    except Exception:
        pass

if __name__ == '__main__':
    main()
