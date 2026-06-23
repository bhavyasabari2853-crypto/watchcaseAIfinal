import cv2
import numpy as np

def get_angle(contour):
    """
    Calculate object orientation angle using PCA.
    Returns angle in degrees (0-180).
    """

    if contour is None:
        return 0.0

    contour = np.array(contour)

    if len(contour) < 5:
        return 0.0

    pts = contour.reshape(-1, 2).astype(np.float32)

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(
        pts,
        mean=np.empty((0))
    )

    angle = np.degrees(
        np.arctan2(
            eigenvectors[0, 1],
            eigenvectors[0, 0]
        )
    )

    if angle < 0:
        angle += 180

    return round(float(angle), 2)