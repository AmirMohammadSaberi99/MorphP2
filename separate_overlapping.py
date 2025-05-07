# separate_overlapping.py

"""
Separate overlapping objects (e.g. coins or cells) using
morphology + distance transform + watershed.
"""

import cv2
import numpy as np

def separate_objects(image_path):
    # 1) Load & pre-process
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarize (invert so objects = white)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 2) Remove noise with opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3) Determine sure background by dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4) Determine sure foreground via distance transform + threshold
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist,
                               0.5 * dist.max(),
                               255,
                               cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # 5) Find unknown region (neither bg nor fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6) Marker labelling
    num_markers, markers = cv2.connectedComponents(sure_fg)
    # ensure background is 1 not 0
    markers = markers + 1
    # mark unknown as zero
    markers[unknown == 255] = 0

    # 7) Apply watershed
    markers = cv2.watershed(img, markers)
    # mark boundaries (where markers == -1)
    img[markers == -1] = [0, 0, 255]

    # 8) Optional: color each segment randomly
    output = np.zeros_like(img)
    for marker_id in range(2, num_markers+2):
        mask = (markers == marker_id)
        color = list(np.random.randint(0,255,3))
        output[mask] = color

    # 9) Display results
    cv2.imshow("Original", img)
    cv2.imshow("Separated Objects", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Separate overlapping objects using morphology + watershed"
    )
    parser.add_argument("image", help="Path to input image (e.g. coins.jpg)")
    args = parser.parse_args()
    separate_objects(args.image)
