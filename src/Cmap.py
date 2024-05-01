import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img(pth):
    img = cv2.imread(pth)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img / 255.0

def rgb_to_xyz(rgb_img):
    xyz = np.zeros_like(rgb_img)
    xyz[:, :, 0] = 0.412453 * rgb_img[:, :, 0] + 0.35758 * rgb_img[:, :, 1] + 0.180423 * rgb_img[:, :, 2]
    xyz[:, :, 1] = 0.212671 * rgb_img[:, :, 0] + 0.71516 * rgb_img[:, :, 1] + 0.072169 * rgb_img[:, :, 2]
    xyz[:, :, 2] = 0.019334 * rgb_img[:, :, 0] + 0.119193 * rgb_img[:, :, 1] + 0.950227 * rgb_img[:, :, 2]
    return xyz

def xyz_to_luv(xyz):
    xn = 0.312713
    yn = 0.329016
    Yn = 1.0  # D65 white point for Y normalization
    
    X, Y, Z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    u = np.divide(4 * X, (X + 15 * Y + 3 * Z), out=np.zeros_like(X), where=(X + 15 * Y + 3 * Z) != 0)
    v = np.divide(9 * Y, (X + 15 * Y + 3 * Z), out=np.zeros_like(Y), where=(X + 15 * Y + 3 * Z) != 0)

    # Compute L
    L = np.where(Y > 0.008856, 116 * (Y ** (1/3)) - 16, 903.3 * Y)
    
    # Compute u', v'
    un = 4 * xn / (-2 * xn + 12 * yn + 3)
    vn = 9 * yn / (-2 * xn + 12 * yn + 3)
    up = 13 * L * (u - un)
    vp = 13 * L * (v - vn)
    
    return np.stack([L, up, vp], axis=-1)

def rgb_to_luv(rgb_image):
    xyz_image = rgb_to_xyz(rgb_image)
    luv_image = xyz_to_luv(xyz_image)
    return luv_image

def plot_rgb_luv_comparison(rgb_image):
    plt.figure(figsize=(12, 4))
    
    # Plot original RGB
    plt.subplot(131)
    plt.imshow(rgb_image)
    plt.title("Original RGB")
    plt.axis('off')

    # Convert and plot OpenCV LUV
    luv_image_cv2 = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2Luv)
    plt.subplot(132)
    plt.imshow(luv_image_cv2)
    plt.title("OpenCV LUV")
    plt.axis('off')

    # Convert and plot custom LUV, normalized for visualization
    luv_image_custom = rgb_to_luv(rgb_image)
    plt.subplot(133)
    plt.imshow(luv_image_custom / np.max(luv_image_custom))  
    plt.title("Custom LUV")
    plt.axis('off')

    plt.show()
    
#TODO LINK THIS
def convert(pth):
    img = load_img(pth)
    luv_image = rgb_to_luv(img)
    return luv_image


# if __name__ == '__main__':
#     rgb_image = load_img('images/bay.jpg')
#     plot_rgb_luv_comparison(rgb_image)

luv_img = convert("images/bay.jpg")
print(luv_img)


