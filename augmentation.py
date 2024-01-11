import numpy as np
import cv2
from scipy.ndimage import shift
from process_image_set import compute_edges, extract_roi
import matplotlib.pyplot as plt

def select_images(imagearray):
    length = len(imagearray)
    s_idx = np.random.permutation(length)[0:length//4]
    return np.array(imagearray)[s_idx.astype(int)]

# extracts patches with full height
def crop_image(imageSet, wcrop, slide=25):
    # extract overlapping crops from image
    image = imageSet.imageNormalised
    h, w = image.shape
    # crop image width, keep the height
    w_crop = w - wcrop
    # Vein probabilities
    # VP = imageSet.veinprobs
    patch_image = []
    for columnRight in range(w_crop, w, slide):
        columnLeft = columnRight - w_crop
        patchImg = image[:, columnLeft:columnRight]
        # patchMs = VP[:, columnLeft:columnRight]
        patch_image.append(patchImg)
    patch_image.append(image)
    return patch_image

def flip_image(imagepairs):
    flips = []
    for pair in imagepairs:
        flipImage = cv2.flip(pair, 0)
        # flipVP = cv2.flip(pair[1], 0)
        flips.append(flipImage)
    imagepairs.extend(flips)
    return imagepairs