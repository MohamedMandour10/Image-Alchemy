import numpy as np
import cv2


class thresholding:
    def __init__(self, original_img):
        self.gray = cv2.convertScaleAbs(original_img)
        self.threshold = 120
        self.block_size = 11

    def global_thresholding(self):
        return np.where(self.gray > self.threshold, 255, 0).astype(np.uint8)

    def local_thresholding(self):
        # Create meshgrids for block indices with adjusted step size
        rows, cols = np.meshgrid(np.arange(0, self.gray.shape[0] - self.block_size + 1, self.block_size),
                                 np.arange(0, self.gray.shape[1] - self.block_size + 1, self.block_size),
                                 indexing='ij')

        # Extract blocks and compute local thresholds
        blocks = np.array([self.gray[r:r + self.block_size, c:c + self.block_size] for r, c in zip(rows.flatten(), cols.flatten())])
        threshold_values = np.mean(blocks, axis=(1, 2))

        # Apply local thresholding using vectorized operations
        local_threshold = np.zeros_like(self.gray)
        for (r, c), block, threshold_value in zip(zip(rows.flatten(), cols.flatten()), blocks, threshold_values):
            local_threshold[r:r + self.block_size, c:c + self.block_size] = np.where(block > threshold_value, 255, 0)

        return local_threshold.astype(np.uint8)
