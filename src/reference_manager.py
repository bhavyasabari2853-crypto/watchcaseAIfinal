import os
import sys
import cv2
import numpy as np

# Single Path Manager
file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(file_dir)

def data_path(relative_path):
    if getattr(sys, "frozen", False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = project_root
    return os.path.join(base_path, relative_path)

class ReferenceManager:
    def __init__(self, ref_dir=None):
        self.ref_dir = ref_dir if ref_dir else data_path("references")
        os.makedirs(self.ref_dir, exist_ok=True)
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.references = {}
        self.active_name = None
        self._load_all()

    def _load_all(self):
        self.references.clear()
        if not os.path.exists(self.ref_dir):
            return
        for fname in os.listdir(self.ref_dir):
            path = os.path.join(self.ref_dir, fname)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png'):
                continue
            name = os.path.splitext(fname)[0]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            kp, des = self.orb.detectAndCompute(img, None)
            self.references[name] = (kp, des)
            if self.active_name is None:
                self.active_name = name
        print(f"Loaded {len(self.references)} reference(s) from '{self.ref_dir}'")

    def activate(self, name):
        name = name.strip()
        found = False
        for ext in ('.jpg', '.jpeg', '.png'):
            path = os.path.join(self.ref_dir, f"{name}{ext}")
            if os.path.isfile(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    kp, des = self.orb.detectAndCompute(img, None)
                    self.references[name] = (kp, des)
                    found = True
                    break
        if name in self.references:
            self.active_name = name
            print(f"Active reference set to: {name}")
        elif found:
            self.active_name = name
            print(f"Active reference set to: {name}")
        else:
            print(f"Warning: '{name}' not found in references")

    def get_active_name(self):
        return self.active_name

    def add(self, image_path, name):
        name = name.strip()
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png'):
            ext = '.jpg'
        dest = os.path.join(self.ref_dir, f"{name}{ext}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        cv2.imwrite(dest, img)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        self.references[name] = (kp, des)
        self.active_name = name
        print(f"Added and activated reference: {dest}")
        return dest

    def match(self, query_gray, min_matches=5):
        if self.active_name is None or self.active_name not in self.references:
            return None, 0.0
        if query_gray is None or query_gray.size == 0:
            return None, 0.0
        kp_q, des_q = self.orb.detectAndCompute(query_gray, None)
        if des_q is None or len(kp_q) < 2:
            return None, 0.0

        kp_ref, des_ref = self.references[self.active_name]
        if des_ref is None or len(kp_ref) < 2:
            return None, 0.0

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = matcher.knnMatch(des_q, des_ref, k=2)
        good = []
        for pair in knn:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        if len(good) >= min_matches:
            score = len(good) / max(len(kp_q), len(kp_ref))
            return self.active_name, round(score, 3)
        return None, 0.0
