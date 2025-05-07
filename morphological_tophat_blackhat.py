# morphological_tophat_blackhat.py

"""
Apply top-hat and black-hat morphological operations to a grayscale image
and display the results side by side.
"""

import cv2
import numpy as np
import argparse

def main(image_path, kernel_size=(15,15), shape=cv2.MORPH_RECT):
    # 1) Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # 2) Build structuring element
    kx, ky = kernel_size
    # ensure odd dimensions ≥1
    kx = max(1, (kx//2)*2 + 1)
    ky = max(1, (ky//2)*2 + 1)
    kernel = cv2.getStructuringElement(shape, (kx, ky))

    # 3) Compute top-hat (original − opening)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    # 4) Compute black-hat (closing − original)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    # 5) Stack and annotate for display
    def to_bgr(gray):
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    orig_bgr    = to_bgr(img)
    tophat_bgr  = to_bgr(tophat)
    black_bgr   = to_bgr(blackhat)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig_bgr,   'Original',  (10,30), font, 1, (255,255,255), 2)
    cv2.putText(tophat_bgr, 'Top-Hat',   (10,30), font, 1, (0,255,0),    2)
    cv2.putText(black_bgr,  'Black-Hat', (10,30), font, 1, (0,255,255),  2)

    combined = np.hstack([orig_bgr, tophat_bgr, black_bgr])

    # 6) Display
    cv2.imshow(f"Top-Hat & Black-Hat (kernel={kx}x{ky})", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate morphological top-hat and black-hat transforms"
    )
    parser.add_argument(
        "image", help="Path to input image (grayscale)"
    )
    parser.add_argument(
        "--kernel", "-k",
        nargs=2, type=int, metavar=('KX','KY'),
        default=[15,15],
        help="Kernel size (width height), e.g. -k 15 15"
    )
    parser.add_argument(
        "--shape", "-s",
        choices=['rect','ellipse','cross'],
        default='rect',
        help="Structuring element shape"
    )
    args = parser.parse_args()

    shape_map = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS
    }
    main(args.image, tuple(args.kernel), shape_map[args.shape])
