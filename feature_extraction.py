import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

BINS_HSV = [8, 2, 2]
RANGES_HSV = [0, 180, 0, 256, 0, 256]
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0]
GLCM_NIVEAUX = 64
GLCM_PROPRIETES = ["contrast", "homogeneity", "energy", "correlation"]
LBP_POINTS = 8
LBP_RAYON = 1
LBP_METHODE = "uniform"
LBP_N_BINS = LBP_POINTS + 2


def extraire_stats_rgb(image_rgb):
    features = []
    for canal in range(3):
        pixels = image_rgb[:, :, canal].astype(np.float64)
        features.append(np.mean(pixels))
        features.append(np.std(pixels))
    return np.array(features, dtype=np.float64)


def extraire_hist_hsv(image_rgb):
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [image_hsv], [0, 1, 2], None, BINS_HSV, RANGES_HSV
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float64)


def extraire_glcm(image_rgb):
    image_gris = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    image_quant = (image_gris / 256 * GLCM_NIVEAUX).astype(np.uint8)
    glcm = graycomatrix(
        image_quant,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_NIVEAUX,
        symmetric=True,
        normed=True,
    )
    features = []
    for prop in GLCM_PROPRIETES:
        valeurs = graycoprops(glcm, prop)
        features.extend(valeurs.flatten())
    return np.array(features, dtype=np.float64)


def extraire_lbp(image_rgb):
    image_gris = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(
        image_gris, LBP_POINTS, LBP_RAYON, method=LBP_METHODE
    )
    hist, _ = np.histogram(
        lbp, bins=LBP_N_BINS, range=(0, LBP_N_BINS), density=True
    )
    return hist.astype(np.float64)


def extraire_moments_hu(image_rgb):
     image_gris = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
     moments = cv2.moments(image_gris)
     hu = cv2.HuMoments(moments).flatten()
     epsilon = 1e-12
     hu_log = -np.sign(hu) * np.log10(np.abs(hu) + epsilon)
     hu_log = np.where(np.isfinite(hu_log), hu_log, 0.0)
     return hu_log.astype(np.float64)

def extraire_caracteristiques(image_rgb):
    f_rgb = extraire_stats_rgb(image_rgb)
    f_hsv = extraire_hist_hsv(image_rgb)
    f_glcm = extraire_glcm(image_rgb)
    f_lbp = extraire_lbp(image_rgb)
    f_hu = extraire_moments_hu(image_rgb)
    vecteur = np.concatenate([f_rgb, f_hsv, f_glcm, f_lbp, f_hu])
    if vecteur.shape != (59,):
        raise ValueError(
            f"Expected 59 features, got {vecteur.shape[0]}."
        )
    if not np.all(np.isfinite(vecteur)):
        raise ValueError("Feature extraction produced non-finite values.")
    return vecteur
