import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm
import numba
from edgeUtils import*

def gradient_magnitudes(grayscale_img):
    """
    Simple computation the magnitute of pixel gradients.
    Returns gradient magnitude energy map of image.
    Sub-optimal energy map for seam carving
    """
    _, _, Im = partial_derivatives(grayscale_img)

    return Im

@numba.jit
def least_energy_seam(gradient_magnitude_img):
    """
    Finding the seam can be seen as a "minimum path sum" problem.
    Finds the seam with the lowest weighted path from top to bottom of image.
    Optimized with dynamic programming principles for fast computation.
    It is highly recomended to use numba for speed increase.
    """
    grad_mag = gradient_magnitude_img.copy()
    curr_sums = grad_mag.copy()                                                 #energy map of image (contains the sum of the shortest path up to each pixel)
    min_col_indices = np.zeros((curr_sums.shape[0],curr_sums.shape[1]))         #Column indices to lowest valued neighbor-element in row above
    for row in range(1, curr_sums.shape[0]):
        for col in range(curr_sums.shape[1]):
            if col == 0:   #handle column edge case
                candidates = curr_sums[row-1, col:col+2]                         #List of the two possible elements in row above
                best_candidate = np.amin(candidates)                             #Best candidate is the lowest valued element
                index = int(np.where(candidates == best_candidate)[0][0]) + col  #Index for best candidate in row above
            else:
                candidates = curr_sums[row-1, col-1:col+2]                        #List of the three possible elements in row above
                best_candidate = np.amin(candidates)                              #Best candidate is the lowest valued element
                index = int(np.where(candidates == best_candidate)[0][0]) + col-1 #Index for best candidate in row above
                
            curr_sums[row, col] += best_candidate 
            min_col_indices[row, col] = index

    complete_energy_map = curr_sums                                               #Just syntax sugar
    return complete_energy_map , min_col_indices

@numba.jit
def get_min_seam(energy_map, path_indices):
    """
    Returns 1D array of column indices for the optimal seam. 
    The seam array indices corresponds to the row intex.
    """ 
    energy = energy_map.copy()
    j = np.argmin(energy[-1])
    seam = np.zeros(path_indices.shape[0])

    for i in range(path_indices.shape[0]-1, -1, -1):
        j = int(path_indices[i, j])
        seam[i] = j
    return seam

def draw_seam(img, seam, seam_width = 1, img_type = 'RGB'):
    """
    Returns the input image with the seam drawn in the image.
    """
    if img_type == 'RGB':
        seam_image = img.copy()
        for row in range(seam.shape[0]):
            seam_image[row, int(seam[row]):int(seam[row]+seam_width)] = np.array([1,0,0])
    elif img_type == 'BGR':
        seam_image = img.copy()
        for row in range(seam.shape[0]):
            seam_image[row, int(seam[row]):int(seam[row]+seam_width)] = np.array([1,0,0])
    else:
        seam_image = img.copy()
        for row in range(seam.shape[0]):
            seam_image[row, int(seam[row]):int(seam[row]+seam_width)] = 1
    
    return seam_image


def remove_seam(RGB, seam):
    """
    Inputs an image and a 1D array of column indices of a seam.
    Removes the seam from the image and reshapes the it.
    Mask is of shape RGB.shape and holds boolean values. All
    elements in a row is set to True except the ones with column
    index seam[row], which is set to False.
    """
    mask = np.arange(RGB.shape[1]) != np.array(seam)[:, None]
    without_seam = RGB[mask]
    without_seam = without_seam.reshape(RGB.shape[0], RGB.shape[1] -1, 3)
    return without_seam


def carve_image_width(RGB, remove_width):
    """
    Removes remove_width number of vertical seams in the image.
    """
    resized_RGB = RGB.copy()
    for width in tqdm(range(remove_width)):
        gray = rgb_to_gray(resized_RGB)
        grad_mag = gradient_magnitudes(gray)
        energy_map, paths = least_energy_seam(grad_mag)
        seam = get_min_seam(energy_map, paths)
        resized_RGB = remove_seam(resized_RGB, seam)
    
    return resized_RGB

def carve_image_height(RGB, remove_height):
    """
    Removes remove_height number of horizontal seams in the image. 
    """
    RGB = np.rot90(RGB, k = 1)
    resized_RGB = carve_image_width(RGB, remove_height)

    return np.rot90(resized_RGB, k = -1)
    