import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter(image, radius, sigma):
    size = 2 * radius + 1
    x, y = np.meshgrid(np.linspace(-radius, radius, size), np.linspace(-radius, radius, size))
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return cv2.filter2D(image, -1, kernel)

# Görüntü dosyasını oku ve doğru okunup okunmadığını kontrol et
image = cv2.imread("lenna.png")
if image is None:
    print("Image file couldn't be read.")
else:
    blurred_image = gaussian_filter(image, radius=9, sigma=50)
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax2.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    ax2.set_title("Gaussian Blur Filter")
    plt.show()