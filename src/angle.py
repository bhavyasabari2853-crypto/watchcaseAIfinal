import cv2
import numpy as np

def get_angle(contour):
    """
    Calculate object centroid and orientation angle using PCA.
    Returns (centroid_x, centroid_y, angle) in degrees.
    Works for any shape.
    """
    if contour is None:
        return 0.0, 0.0, 0.0

    pts = np.array(contour).reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return 0.0, 0.0, 0.0

    # Calculate centroid (mean of points) as a fallback
    mean = np.mean(pts, axis=0)
    centroid_x = float(mean[0])
    centroid_y = float(mean[1])

    try:
        # cv2.PCACompute expects data as a floating point matrix of shape (N, 2)
        # mean=None tells it to compute the mean automatically
        mean_pca, eigenvectors = cv2.PCACompute(pts, mean=None)
        
        # Angle from the first eigenvector (direction of maximum variance)
        angle = np.degrees(np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0]))
        if angle < 0:
            angle += 180.0
            
        # Use PCA mean as centroid for higher precision
        if mean_pca is not None and mean_pca.size >= 2:
            centroid_x = float(mean_pca[0, 0])
            centroid_y = float(mean_pca[0, 1])
    except Exception as e:
        print(f"PCA computation error, falling back to moments: {e}")
        # Fallback using moments
        M = cv2.moments(pts)
        if M["m00"] != 0:
            centroid_x = M["m10"] / M["m00"]
            centroid_y = M["m01"] / M["m00"]
        angle = 0.0

    return round(centroid_x, 2), round(centroid_y, 2), round(angle, 2)