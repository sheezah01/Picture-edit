import numpy as np
from typing import Union, Tuple
from image import Image

def validate_image(func):
    """Decorator to validate image input"""
    def wrapper(image: Image, *args, **kwargs):
        if not isinstance(image, Image):
            raise TypeError("Input must be an Image object")
        if image.array.size == 0:
            raise ValueError("Image array is empty")
        return func(image, *args, **kwargs)
    return wrapper

@validate_image
def brighten(image: Image, factor: float) -> Image:
    """Brighten or darken the image by scaling each pixel's intensity."""
    if factor < 0:
        raise ValueError("Factor must be non-negative")
    new_image_array = np.clip(image.array * factor, 0, 1)
    return Image(array=new_image_array)

@validate_image
def adjust_contrast(image: Image, factor: float, mid: float = 0.5) -> Image:
    """Adjust contrast relative to a midpoint using vectorized operations."""
    if not 0 <= mid <= 1:
        raise ValueError("Midpoint must be between 0 and 1")
    new_image_array = np.clip((image.array - mid) * factor + mid, 0, 1)
    return Image(array=new_image_array)

@validate_image
def blur(image: Image, kernel_size: int) -> Image:
    """Blur image using an optimized box filter approach with proper edge handling."""
    if kernel_size % 2 != 1 or kernel_size < 3:
        raise ValueError("Kernel size must be an odd integer >= 3")
    x, y, c = image.array.shape
    new_im = Image(x_pixels=x, y_pixels=y, num_channels=c)
    pad = kernel_size // 2
    padded = np.pad(image.array, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    for x in range(x):
        for y in range(y):
            new_im.array[x, y] = np.mean(padded[x:x+kernel_size, y:y+kernel_size], axis=(0, 1))
    return new_im

@validate_image
def apply_kernel(image: Image, kernel: np.ndarray, padding_mode: str = 'edge') -> Image:
    """Apply a convolution kernel to the image with improved edge handling."""
    if kernel.shape[0] != kernel.shape[1] or kernel.shape[0] % 2 != 1:
        raise ValueError("Kernel must be a square with odd size")
    x, y, c = image.array.shape
    k = kernel.shape[0] // 2
    padded = np.pad(image.array, ((k, k), (k, k), (0, 0)), mode=padding_mode)
    new_im = Image(x_pixels=x, y_pixels=y, num_channels=c)
    for x in range(x):
        for y in range(y):
            new_im.array[x, y] = np.sum(padded[x:x+kernel.shape[0], y:y+kernel.shape[1]] * kernel, axis=(0, 1))
    return new_im

@validate_image
def combine_images(image1: Image, image2: Image, method: str = 'root_sum_square') -> Image:
    """Combine two images using various methods."""
    if image1.array.shape != image2.array.shape:
        raise ValueError("Images must have the same dimensions")
    methods = {
        'root_sum_square': lambda x, y: np.sqrt(x**2 + y**2),
        'add': lambda x, y: x + y,
        'multiply': lambda x, y: x * y,
        'maximum': np.maximum
    }
    if method not in methods:
        raise ValueError(f"Invalid method. Choose from: {list(methods.keys())}")
    combined_array = np.clip(methods[method](image1.array, image2.array), 0, 1)
    return Image(array=combined_array)

def create_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a Gaussian kernel for blurring."""
    if size % 2 != 1:
        raise ValueError("Kernel size must be odd")
    k = (size - 1) // 2
    x, y = np.meshgrid(np.linspace(-k, k, size), np.linspace(-k, k, size))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def create_edge_kernel(direction: str) -> np.ndarray:
    """Create kernels for edge detection."""
    kernels = {
        'sobel_x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'sobel_y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'prewitt_x': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        'prewitt_y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        'laplacian': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    }
    if direction not in kernels:
        raise ValueError(f"Invalid direction. Choose from: {list(kernels.keys())}")
    return kernels[direction]