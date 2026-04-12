import numpy as np
import cv2

def ihs_transform(pan, ms):
    """
    pan: 2D numpy array (H, W)
    ms:  3D numpy array (H, W, 3) -> BGR (OpenCV format)

    Returns: pansharpened image (BGR)
    """
    # Convert to float
    ms = ms.astype(np.float32)
    pan = pan.astype(np.float32)

    # Normalize to [0,1]
    ms_norm = ms / 255.0
    pan_norm = pan / 255.0

    # Convert BGR -> HSV (OpenCV uses BGR!)
    hsv = cv2.cvtColor(ms_norm, cv2.COLOR_BGR2HSV)

    # Replace intensity (Value channel) with PAN
    hsv[:, :, 2] = pan_norm

    # Convert back HSV -> BGR
    fused = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return np.clip(fused * 255, 0, 255).astype(np.uint8)
# Read images
pan = cv2.imread('pan_image.png', cv2.IMREAD_GRAYSCALE)
ms = cv2.imread('lrms_image.png')

# 🔥 Resize MS to PAN resolution (same as before)
ms = cv2.resize(ms, (pan.shape[1], pan.shape[0]), interpolation=cv2.INTER_CUBIC)

# Apply IHS
ihs_img = ihs_transform(pan, ms)

# Show
cv2.imshow('IHS Pansharpened', ihs_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save
cv2.imwrite('ihs_output.jpg', ihs_img)