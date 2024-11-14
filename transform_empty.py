import numpy as np
from scipy import signal

class Image:
    def __init__(self, filename=None, x_pixels=None, y_pixels=None, num_channels=3):
        if filename:
            self.array = np.array(Image.open(filename))
        else:
            self.array = np.zeros((x_pixels, y_pixels, num_channels))
    
    def write_image(self, filename):
        Image.save(self.array, filename)

def brighten(image, factor):
    return image * factor

def adjust_contrast(image, factor, mid):
    return (image - mid) * factor + mid

def blur(image, kernel_size):
    assert kernel_size % 2 == 1
    k = kernel_size // 2
    kernel = np.exp(-(np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))[0]**2 + 
                     np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))[1]**2) / (2 * 1.0**2))
    kernel /= kernel.sum()
    return apply_kernel(image, kernel)

def apply_kernel(image, kernel):
    x, y, c = image.array.shape
    result = np.zeros_like(image.array)
    for channel in range(c):
        result[:,:,channel] = signal.convolve2d(image.array[:,:,channel], kernel, 'same')
    return Image(x_pixels=x, y_pixels=y, num_channels=c, array=result)

def combine_images(image1, image2):
    x, y, c = image1.array.shape
    result = np.zeros((x, y, c))
    for channel in range(c):
        result[:,:,channel] = np.sqrt(image1.array[:,:,channel]**2 + image2.array[:,:,channel]**2)
    return Image(x_pixels=x, y_pixels=y, num_channels=c, array=result)

def rotate_image(image, angle_degrees):
    x, y, c = image.array.shape
    center_x, center_y = x // 2, y // 2
    angle_radians = np.radians(angle_degrees)
    result = np.zeros_like(image.array)
    
    for x_ in range(x):
        for y_ in range(y):
            xr = x_ - center_x
            yr = y_ - center_y
            new_x = int(xr * np.cos(angle_radians) - yr * np.sin(angle_radians) + center_x)
            new_y = int(xr * np.sin(angle_radians) + yr * np.cos(angle_radians) + center_y)
            if 0 <= new_x < x and 0 <= new_y < y:
                result[x_,y_] = image.array[new_x,new_y]
    
    return Image(x_pixels=x, y_pixels=y, num_channels=c, array=result)

def apply_sepia(image):
    x, y, c = image.array.shape
    result = np.zeros_like(image.array)
    sepia = np.array([[0.393, 0.769, 0.189], 
                      [0.349, 0.686, 0.168], 
                      [0.272, 0.534, 0.131]])
    
    for x_ in range(x):
        for y_ in range(y):
            result[x_,y_] = np.clip(image.array[x_,y_] @ sepia.T, 0, 1)
    
    return Image(x_pixels=x, y_pixels=y, num_channels=c, array=result)

def add_vignette(image, strength=0.5):
    x, y, c = image.array.shape
    cx, cy = x//2, y//2
    result = np.zeros_like(image.array)
    
    x_coords, y_coords = np.meshgrid(np.arange(y), np.arange(x))
    distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    max_distance = np.sqrt(cx**2 + cy**2)
    vignette = 1 - (distances / max_distance * strength)
    vignette = np.clip(vignette, 0, 1)
    
    for channel in range(c):
        result[:,:,channel] = image.array[:,:,channel] * vignette
    
    return Image(x_pixels=x, y_pixels=y, num_channels=c, array=result)

def apply_gaussian_blur(image, kernel_size=3, sigma=1.0):
    return blur(image, kernel_size)

def adjust_saturation(image, factor):
    x, y, c = image.array.shape
    result = np.zeros_like(image.array)
    
    max_vals = np.max(image.array, axis=2)
    min_vals = np.min(image.array, axis=2)
    delta = max_vals - min_vals
    
    for channel in range(c):
        result[:,:,channel] = ((image.array[:,:,channel] - min_vals) * factor) + min_vals
    
    return Image(x_pixels=x, y_pixels=y, num_channels=c, array=result)

def create_histogram_equalized(image):
    x, y, c = image.array.shape
    result = np.zeros_like(image.array)
    
    for channel in range(c):
        hist, bins = np.histogram(image.array[:,:,channel].flatten(), bins=256, range=(0,1))
        cdf = hist.cumsum()
        cdf_norm = cdf / cdf.max()
        result[:,:,channel] = np.interp(image.array[:,:,channel].flatten(), bins[:-1], cdf_norm).reshape((x, y))
    
    return Image(x_pixels=x, y_pixels=y, num_channels=c, array=result)