# connected_components_morph.py

"""
Count connected components before and after morphological processing.

This script:
1. Loads a grayscale image and thresholds it to binary.
2. Counts and prints the number of connected components.
3. Applies a morphological operation (opening or closing).
4. Counts and prints the number of connected components again.
5. Displays the original and processed binary images side by side,
   annotated with the component counts.
"""

import cv2
import numpy as np
import argparse

# Map operation names to OpenCV flags
MORPH_OPS = {
    'open': cv2.MORPH_OPEN,
    'close': cv2.MORPH_CLOSE
}

def count_components(bin_img):
    """
    Counts connected components in a binary image.
    Returns:
        num_labels (int): number of labels, including background.
        stats (ndarray): component statistics.
        centroids (ndarray): component centroids.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    return num_labels, stats, centroids

def apply_morphology(bin_img, op, kx, ky, iterations=1):
    """
    Applies a morphological operation (open or close) to a binary image.
    Args:
        bin_img    : Input binary image (0 or 255).
        op         : 'open' or 'close'.
        kx, ky     : Kernel width and height.
        iterations : Number of times to apply the operation.
    Returns:
        result     : Binary image after morphology.
    """
    kx = max(1, (kx//2)*2 + 1)
    ky = max(1, (ky//2)*2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    result = cv2.morphologyEx(bin_img, MORPH_OPS[op], kernel, iterations=iterations)
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Count connected components before and after morphological processing"
    )
    parser.add_argument("image", help="Path to input grayscale or binary image")
    parser.add_argument(
        "--op", "-o",
        choices=list(MORPH_OPS.keys()),
        default="open",
        help="Morphological operation to apply (default: open)"
    )
    parser.add_argument(
        "--kernel", "-k",
        nargs=2, type=int, metavar=('KX','KY'),
        default=[3,3],
        help="Structuring element size (width height), e.g. -k 5 5"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int, default=1,
        help="Number of times to apply the morphological operation"
    )
    args = parser.parse_args()

    # 1) Load and threshold to binary
    img_gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")
    _, bin_orig = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # 2) Count original components
    n_orig, stats_orig, _ = count_components(bin_orig)
    print(f"Original connected components (including background): {n_orig}")

    # 3) Apply morphology
    bin_proc = apply_morphology(
        bin_orig,
        args.op,
        args.kernel[0],
        args.kernel[1],
        iterations=args.iterations
    )

    # 4) Count processed components
    n_proc, stats_proc, _ = count_components(bin_proc)
    print(f"After {args.op} (kernel={args.kernel[0]}x{args.kernel[1]}, "
          f"iter={args.iterations}) connected components: {n_proc}")

    # 5) Visualize side by side
    bgr_orig = cv2.cvtColor(bin_orig, cv2.COLOR_GRAY2BGR)
    bgr_proc = cv2.cvtColor(bin_proc, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bgr_orig,  f"Orig CC: {n_orig}",  (10,30), font, 1, (255,255,255), 2)
    label = f"{args.op.title()} -> CC: {n_proc}"
    cv2.putText(bgr_proc,  label,               (10,30), font, 1, (0,255,0),    2)
    combined = np.hstack([bgr_orig, bgr_proc])

    cv2.imshow("Connected Components Before vs After", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
