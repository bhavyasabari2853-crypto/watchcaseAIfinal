import os
import cv2
import numpy as np


class ReferenceManager:
    def __init__(self, ref_dir="references"):
        self.ref_dir = ref_dir
        os.makedirs(ref_dir, exist_ok=True)
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.references = {}
        self.active_name = None
        self._load_all()

    def _load_all(self):
        self.references.clear()
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
        print(f"Loaded {len(self.references)} reference(s) from '{self.ref_dir}'")

    def activate(self, name):
        path = os.path.join(self.ref_dir, f"{name}.jpg")
        if not os.path.isfile(path):
            path = os.path.join(self.ref_dir, f"{name}.jpeg")
        if not os.path.isfile(path):
            path = os.path.join(self.ref_dir, f"{name}.png")
        if os.path.isfile(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = self.orb.detectAndCompute(img, None)
                self.references[name] = (kp, des)
        if name in self.references:
            self.active_name = name
            print(f"Active reference set to: {name}")
        else:
            print(f"Warning: '{name}' not found in references")

    def get_active_name(self):
        return self.active_name

    def add(self, image_path, name):
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
