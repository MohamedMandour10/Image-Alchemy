import numpy as np
import cv2


def get_histograms(image):
    # Separate the R, G, and B channels
    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    # Calculate the histograms for R, G, and B channels
    hist_R, bins_R = np.histogram(R.flatten(), bins=256, range=(0, 256))
    hist_G, bins_G = np.histogram(G.flatten(), bins=256, range=(0, 256))
    hist_B, bins_B = np.histogram(B.flatten(), bins=256, range=(0, 256))

    # Calculate the cumulative distribution functions (CDFs) for R, G, and B channels
    cdf_R = hist_R.cumsum()
    cdf_G = hist_G.cumsum()
    cdf_B = hist_B.cumsum()

    return {"R": [hist_R, cdf_R], "G": [hist_G, cdf_G], "B": [hist_B, cdf_B]}
