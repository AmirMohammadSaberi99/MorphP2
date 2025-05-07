# boundary_gradient.py

"""
Extract the boundaries of shapes in a binary image using the morphological gradient.
The morphological gradient is defined as the difference between the dilation and erosion
of the image, which highlights object edges.
"""

import cv2
import numpy as np
import argparse

def compute_gradient(img_bin, kernel_size=(3,3), iterations=1, shape=cv2.MORPH_RECT):
    """
    Compute the morphological gradient of a binary image.
    
    Args:
        img_bin       : Input binary image (0 or 255).
        kernel_size   : Tuple (kx, ky) for the structuring element size.
        iterations    : Number of times to apply dilation and erosion.
        shape         : cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, or cv2.MORPH_CROSS.
    Returns:
        gradient_img  : Image showing boundaries (uint8).
        kernel        : The structuring element used.
    """
    kx, ky = kernel_size
    # Ensure odd dimensions ≥ 1
    kx = max(1, (kx // 2) * 2 + 1)
    ky = max(1, (ky // 2) * 2 + 1)
    kernel = cv2.getStructuringElement(shape, (kx, ky))
    
    # Compute morphological gradient
    gradient = cv2.morphologyEx(img_bin, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    return gradient, kernel

def main(image_path, kx, ky, iterations, shape_name):
    # 1) Load image in grayscale
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    # 2) Threshold to binary
    _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # 3) Choose kernel shape
    shapes = {
        'rect': cv2.MORPH_RECT,
        'ellipse': cv2.MORPH_ELLIPSE,
        'cross': cv2.MORPH_CROSS
    }
    shape = shapes.get(shape_name, cv2.MORPH_RECT)
    
    # 4) Compute gradient
    gradient, kernel = compute_gradient(
        img_bin,
        kernel_size=(kx, ky),
        iterations=iterations,
        shape=shape
    )
    
    # 5) Visualize
    # Convert binary and gradient to BGR for labeling
    orig_bgr     = cv2.cvtColor(img_bin,   cv2.COLOR_GRAY2BGR)
    gradient_bgr = cv2.cvtColor(gradient,  cv2.COLOR_GRAY2BGR)
    
    # Annotate
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(orig_bgr,     "Binary Input", (10,30), font, 1, (255,255,255), 2)
    label = f"Gradient | {shape_name} {kx}x{ky} | iter={iterations}"
    cv2.putText(gradient_bgr, label,        (10,30), font, 0.8, (0,255,0),    2)
    
    # Stack side by side
    combined = np.hstack([orig_bgr, gradient_bgr])
    
    cv2.imshow("Morphological Gradient Boundary Extraction", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract boundaries of shapes using morphological gradient"
    )
    parser.add_argument(
        "image",
        help="Path to input image (grayscale or binary)"
    )
    parser.add_argument(
        "--kernel", "-k",
        nargs=2, type=int, metavar=('KX','KY'),
        default=[3,3],
        help="Structuring element size (odd integers), e.g. -k 5 5"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int, default=1,
        help="Number of iterations of dilation & erosion"
    )
    parser.add_argument(
        "--shape", "-s",
        choices=['rect','ellipse','cross'],
        default='rect',
        help="Shape of the structuring element"
    )
    args = parser.parse_args()
    
    main(
        args.image,
        args.kernel[0],
        args.kernel[1],
        args.iterations,
        args.shape
    )
