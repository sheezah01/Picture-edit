# Image Processing Library

This is a Python library for performing common image processing operations such as brightness adjustment, contrast adjustment, blurring, and image combination.

## Features

- **Brightness Adjustment**: Adjust the brightness of an image by scaling the pixel intensities.
- **Contrast Adjustment**: Adjust the contrast of an image relative to a specified midpoint.
- **Blurring**: Apply a box filter blur to an image with proper edge handling.
- **Kernel Application**: Apply a custom convolution kernel to an image with improved edge handling.
- **Image Combination**: Combine two images using various methods (root sum of squares, addition, multiplication, maximum).
- **Utility Functions**: Create Gaussian and edge detection kernels.

## Installation

To use this library, you'll need to have the following dependencies installed:

- NumPy

You can install the dependencies using pip:

```
pip install numpy
```

## Usage

Here's an example of how to use the library:

```python
from image import Image
from image_processing import brighten, adjust_contrast, blur, apply_kernel, combine_images

# Load an image
image = Image(filename='example.jpg')

# Brighten the image
brightened_image = brighten(image, factor=1.5)

# Adjust the contrast
contrast_adjusted_image = adjust_contrast(image, factor=2.0, mid=0.5)

# Apply a Gaussian blur
blurred_image = blur(image, kernel_size=5)

# Apply a custom kernel
edge_kernel = create_edge_kernel('sobel_x')
edged_image = apply_kernel(image, edge_kernel)

# Combine two images
image1 = Image(filename='image1.jpg')
image2 = Image(filename='image2.jpg')
combined_image = combine_images(image1, image2, method='root_sum_square')
```

## API Reference

### Functions

#### `brighten(image: Image, factor: float) -> Image`
Brighten or darken the image by scaling each pixel's intensity.

#### `adjust_contrast(image: Image, factor: float, mid: float = 0.5) -> Image`
Adjust contrast relative to a midpoint using vectorized operations.

#### `blur(image: Image, kernel_size: int) -> Image`
Blur image using an optimized box filter approach with proper edge handling.

#### `apply_kernel(image: Image, kernel: np.ndarray, padding_mode: str = 'edge') -> Image`
Apply a convolution kernel to the image with improved edge handling.

#### `combine_images(image1: Image, image2: Image, method: str = 'root_sum_square') -> Image`
Combine two images using various methods.

#### `create_gaussian_kernel(size: int, sigma: float) -> np.ndarray`
Create a Gaussian kernel for blurring.

#### `create_edge_kernel(direction: str) -> np.ndarray`
Create kernels for edge detection.

## Contributing

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).