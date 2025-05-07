# Advanced Morphological Operations Toolkit

This repository contains Python scripts demonstrating advanced morphological image processing techniques using OpenCV. Five key operations are provided:

## Files

```
├── custom_kernel_morphology.py   # 1. Apply morphology with custom-shaped kernels (rect, ellipse, cross)
├── boundary_gradient.py          # 2. Extract shape boundaries using morphological gradient
├── morphological_tophat_blackhat.py # 3. Perform top-hat and black-hat transforms
├── connected_components_morph.py # 4. Count connected components before/after morphology
└── separate_overlapping.py        # 5. Separate overlapping objects via morphology + watershed
```

## Requirements

* Python 3.6 or higher
* OpenCV

Install dependencies:

```bash
pip install opencv-python
```

For scripts leveraging watershed, NumPy is also required (installed alongside OpenCV).

---

## Script Descriptions & Usage

### 1. `custom_kernel_morphology.py`

**Custom-shaped kernels**: Demonstrates applying any morphological operation (erode, dilate, open, close, gradient, tophat, blackhat) using rectangular, elliptical, or cross-shaped structuring elements.

```bash
python custom_kernel_morphology.py <image> --op erode --shape ellipse --kernel 7 3 --iter 2
```

* `--op`: one of `erode`, `dilate`, `open`, `close`, `gradient`, `tophat`, `blackhat`
* `--shape`: `rect`, `ellipse`, or `cross`
* `--kernel KX KY`: structuring element size
* `--iter N`: iterations

### 2. `boundary_gradient.py`

**Morphological Gradient**: Computes the difference between dilation and erosion to highlight object boundaries in a binary image.

```bash
python boundary_gradient.py <image> --kernel 5 5 --iterations 1 --shape rect
```

* Extracts boundaries side by side with original.

### 3. `morphological_tophat_blackhat.py`

**Top-Hat & Black-Hat**: Extracts small bright (top-hat) and dark (black-hat) features relative to local background using opening and closing.

```bash
python morphological_tophat_blackhat.py <image> --kernel 15 15 --shape rect
```

* Displays original, top-hat, and black-hat results side by side.

### 4. `connected_components_morph.py`

**Component Counting**: Counts connected components before and after a morphological transform (opening or closing), illustrating noise removal or region merging effects.

```bash
python connected_components_morph.py <image> --op open --kernel 3 3 --iterations 1
```

* Prints component counts and shows before/after images.

### 5. `separate_overlapping.py`

**Overlapping Object Separation**: Uses opening, distance transform, and watershed to separate touching or overlapping objects (e.g., coins, cells).

```bash
python separate_overlapping.py <image>
```

* Displays original with watershed boundaries and a color-coded separated output.

---

## Notes

* All scripts threshold input to binary internally if needed.
* Tweak kernel sizes and iteration counts to suit your image characteristics.
* `separate_overlapping.py` relies on a clear distance transform peak for each object; results may vary w
