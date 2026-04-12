import numpy as np
import cv2

def brovey_transform(pan, ms):
    ms = ms.astype(np.float32)
    pan = pan.astype(np.float32)

    sum_rgb = ms[:, :, 0] + ms[:, :, 1] + ms[:, :, 2]
    sum_rgb[sum_rgb == 0] = 1e-6

    out = np.zeros_like(ms)
    for i in range(3):
        out[:, :, i] = (ms[:, :, i] / sum_rgb) * pan

    return np.clip(out, 0, 255).astype(np.uint8)

# Read images
pan = cv2.imread('pan_image.png', cv2.IMREAD_GRAYSCALE)
ms = cv2.imread('lrms_image.png')

# 🔥 IMPORTANT FIX
ms = cv2.resize(ms, (pan.shape[1], pan.shape[0]), interpolation=cv2.INTER_CUBIC)

# Apply transform
pansharpened = brovey_transform(pan, ms)

# Show
cv2.imshow('Pansharpened Image', pansharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save
cv2.imwrite('brovey_output.jpg', pansharpened)