# custom_kernel_morphology.py

"""
Apply morphological operations using custom-shaped kernels:
- Rectangular
- Elliptical
- Cross-shaped
"""

import cv2
import numpy as np
import argparse

# Map string names to OpenCV structuring element shapes
KERNEL_SHAPES = {
    'rect': cv2.MORPH_RECT,
    'ellipse': cv2.MORPH_ELLIPSE,
    'cross': cv2.MORPH_CROSS
}

# Map string names to OpenCV morphology operations
MORPH_OPS = {
    'erode': cv2.MORPH_ERODE,
    'dilate': cv2.MORPH_DILATE,
    'open': cv2.MORPH_OPEN,
    'close': cv2.MORPH_CLOSE,
    'gradient': cv2.MORPH_GRADIENT,
    'tophat': cv2.MORPH_TOPHAT,
    'blackhat': cv2.MORPH_BLACKHAT
}

def apply_morphology(img_bin, op_name, shape_name, kx, ky, iterations=1):
    """
    Apply the specified morphological operation with a custom-shaped kernel.

    Args:
        img_bin    : Input binary image (0 or 255).
        op_name    : One of MORPH_OPS keys.
        shape_name : One of KERNEL_SHAPES keys.
        kx, ky     : Kernel width and height.
        iterations : Number of times to apply the operation.

    Returns:
        result_img : The processed binary image.
    """
    op = MORPH_OPS[op_name]
    shape = KERNEL_SHAPES[shape_name]
    # Ensure odd kernel dimensions >= 1
    kx = max(1, (kx // 2) * 2 + 1)
    ky = max(1, (ky // 2) * 2 + 1)
    kernel = cv2.getStructuringElement(shape, (kx, ky))
    result = cv2.morphologyEx(img_bin, op, kernel, iterations=iterations)
    return result, kernel

def main():
    parser = argparse.ArgumentParser(
        description="Apply morphological ops with custom-shaped kernels"
    )
    parser.add_argument(
        "image", help="Path to input image (grayscale or binary)"
    )
    parser.add_argument(
        "--op", "-o",
        choices=list(MORPH_OPS.keys()),
        default="erode",
        help="Morphology operation to perform"
    )
    parser.add_argument(
        "--shape", "-s",
        choices=list(KERNEL_SHAPES.keys()),
        default="rect",
        help="Kernel shape: rect, ellipse, or cross"
    )
    parser.add_argument(
        "--kernel", "-k",
        nargs=2, type=int, metavar=('KX','KY'),
        default=[5,5],
        help="Kernel size (width height), must be positive integers"
    )
    parser.add_argument(
        "--iter", "-n",
        type=int, default=1,
        help="Number of iterations"
    )
    args = parser.parse_args()

    # Load and threshold image to binary
    img_gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot load image: {args.image}")
    _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    # Apply morphology
    result, kernel = apply_morphology(
        img_bin,
        args.op,
        args.shape,
        args.kernel[0],
        args.kernel[1],
        args.iter
    )

    # Convert to BGR for annotation
    orig_bgr = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
    res_bgr  = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # Annotate
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        orig_bgr, "Original", (10, 30), font, 1, (255, 255, 255), 2
    )
    label = f"{args.op.title()} | {args.shape} kernel {args.kernel[0]}x{args.kernel[1]} | iter={args.iter}"
    cv2.putText(
        res_bgr, label, (10, 30), font, 0.8, (0, 255, 0), 2
    )

    # Stack and display
    combined = np.hstack([orig_bgr, res_bgr])
    cv2.imshow("Morphology with Custom Kernel", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
