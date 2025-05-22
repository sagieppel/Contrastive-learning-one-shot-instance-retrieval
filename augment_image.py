import cv2
import numpy as np
import random
########################################################################################################################################################3
def random_lens_distortion(image, k1_range=(-0.4, 0.0), k2_range=(0.0, 0.1), k3_range=(0.0, 0.05)):
    """
    Apply randomized lens distortion for data augmentation.

    Parameters:
    - image: Input image (NumPy array).
    - k1_range, k2_range, k3_range: Tuples defining the range to sample distortion coefficients.

    Returns:
    - Distorted image (NumPy array).
    """

    h, w = image.shape[:2]

    # Random distortion coefficients
    k1 = random.uniform(*k1_range)  # Usually negative for barrel distortion
    k2 = random.uniform(*k2_range)
    k3 = random.uniform(*k3_range)

    # Camera intrinsic matrix (focal lengths and optical center)
    K = np.array([[w, 0, w / 2],
                  [0, w, h / 2],
                  [0, 0, 1]], dtype=np.float32)

    # Radial distortion + zero tangential distortion (p1, p2)
    dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float32)

    # Compute optimal new camera matrix (without cropping)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1)

    # Remap image
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)
    distorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    return distorted
#######################################################################################################33
import cv2
import numpy as np
import random


def photometric_augmentations(image):
    """
    Apply randomized photometric augmentations to an image.

    Parameters:
    - image: Input image (NumPy array, BGR format as read by OpenCV)

    Returns:
    - Augmented image (NumPy array)
    """

    augmented = image.copy()

    # 1. Random Brightness
    if random.random() < 0.7:
        factor = random.uniform(0.7, 1.3)
        augmented = np.clip(augmented * factor, 0, 255).astype(np.uint8)

    # 2. Random Contrast
    if random.random() < 0.7:
        mean = np.mean(augmented)
        factor = random.uniform(0.75, 1.25)
        augmented = np.clip((augmented - mean) * factor + mean, 0, 255).astype(np.uint8)

    # 3. Random Saturation and Hue (convert to HSV)
    if random.random() < 0.6:
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Saturation
        hsv[:, :, 1] *= random.uniform(0.7, 1.3)

        # Hue shift
        hsv[:, :, 0] += random.uniform(-10, 10)
        hsv = np.clip(hsv, 0, 255)

        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4. Gamma Correction
    if random.random() < 0.5:
        gamma = random.uniform(0.7, 1.5)
        inv_gamma = 1.0 / gamma
        table = (np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)])
                 .astype("uint8"))
        augmented = cv2.LUT(augmented, table)

    # 5. Gaussian Blur
    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)

    # # 6. Gaussian Noise
    # if random.random() < 0.5:
    #     noise = np.random.normal(0, 10, augmented.shape).astype(np.uint8)
    #     augmented = cv2.add(augmented, noise)

    return augmented


##################################################################################################
def crop_and_resize(image, mask, minsize):

    maxsz=image.shape[0]
    # Find bounding box of the mask
    ys, xs = np.where(mask > 0)

    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)

    sz = np.max([y_max-y_min,x_max-x_min,minsize])
    sz=np.random.randint(sz,maxsz)
    x0 = np.random.randint(np.max([x_min - (sz - (x_max - x_min)),0]), x_min+1)
    y0 = np.random.randint(np.max([y_min - (sz - (y_max - y_min)),0]), y_min+1)
    # Crop
    mask=mask.astype(np.uint8)
    img_crop = image[y0:y0+sz, x0:x0+sz]
    mask_crop = mask[y0:y0+sz, x0:x0+sz]

    # Resize back to 512x512
    img_resized = cv2.resize(img_crop, (maxsz, maxsz), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_crop, (maxsz, maxsz), interpolation=cv2.INTER_NEAREST)

    return img_resized, mask_resized
###########################################################################################################

#######################################################################################################################################################
def augment_image(image,mask=None,
                  apply_gaussian=True,
                  apply_decolor=True,
                  apply_noise=True,
                  apply_intensity=True,
                  blur_ksize=(5, 5),
                  noise_std=25,min_size=256,high_augmentation=True):
    """
    Augments the input image with optional Gaussian blur, decoloring, and Gaussian noise.

    Parameters:
        image (np.ndarray): Input image as (H, W, 3) NumPy array, dtype uint8.
        apply_gaussian (bool): Whether to apply Gaussian blur.
        apply_decolor (bool): Whether to apply decoloring (grayscale conversion).
        apply_noise (bool): Whether to add Gaussian noise.
        blur_ksize (tuple): Kernel size for Gaussian blur.
        noise_std (float): Standard deviation of the Gaussian noise.
        decolor_prob (float): Probability to convert image to grayscale.

    Returns:
        np.ndarray: Augmented image.
    """

    aug_image = image.copy()
    if np.random.rand()<0.5:
        aug_image = np.rot90(aug_image)
        if mask is not None: mask = np.rot90(mask)
    # Apply Gaussian blur
    if apply_gaussian:
        aug_image = cv2.GaussianBlur(aug_image, blur_ksize, 0)

    # Random grayscale conversion
    if apply_decolor:
        gray = cv2.cvtColor(aug_image, cv2.COLOR_BGR2GRAY)
        aug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Add Gaussian noise
    if apply_noise:
        noise = np.random.normal(0, noise_std, aug_image.shape).astype(np.float32)
        aug_image = aug_image.astype(np.float32) + noise
        aug_image = np.clip(aug_image, 0, 255).astype(np.uint8)
    if apply_intensity:
        aug_image=aug_image.astype(np.float32)*(0.65+0.65*np.random.rand())
        aug_image = np.clip(aug_image, 0, 255).astype(np.uint8)
    if mask is not None and np.random.rand()<0.5:
           aug_image, mask = crop_and_resize(aug_image, mask, min_size)
          ########### cv2.imshow("",aug_image2);cv2.waitKey()
    if high_augmentation:
        aug_image=random_lens_distortion(aug_image)
        aug_image=photometric_augmentations(aug_image)
    return aug_image, mask
#################################################################################################################################################################

