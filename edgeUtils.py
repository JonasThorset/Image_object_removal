import numpy as np


#Transform an RGB image to a grayscale image
def rgb_to_gray(img):
    return img[:, :, 0]*0.299 + img[:, :, 1]*0.587 +img[:, :, 2]*0.114

#Normalize an rgb image
def normalize_RGB(img):
    img_norm = np.zeros(img.shape)
    epsilon = 1/255.0 #Avoid singularities
    img_norm[:, :, 0] = img[:, :, 0]/(img[:, :, 0] + img[:, :, 1] +img[:, :, 2]+epsilon)
    img_norm[:, :, 1] = img[:, :, 1]/(img[:, :, 0] + img[:, :, 1] +img[:, :, 2]+epsilon)
    img_norm[:, :, 2] = img[:, :, 2]/(img[:, :, 0] + img[:, :, 1] +img[:, :, 2]+epsilon)
    
    return img_norm


def convolution_2D(img, kernel, padding = 0, strides = 1):

    #Cross-correlation and convolution is very similar except
    #that convolution uses the flipped version of the kernel.
    kernel = np.flipud(kernel) #Flip vertically (rotation around x)
    kernel = np.fliplr(kernel) #Flip vertically (rotation around y)

    #The output image size is dependent on the image size, kernel size,
    #padding size and stride size.
    x_out_size = int(((img.shape[0] - kernel.shape[0] + 2*padding)/strides) + 1)
    y_out_size = int(((img.shape[1] - kernel.shape[1] + 2*padding)/strides) + 1)
    img_out = np.zeros((x_out_size, y_out_size))

    #Padding the input image
    if padding != 0:
        padded_img = np.zeros((img.shape[0] + 2*padding, img.shape[1] + 2*padding)) #The image dimension increases with 2*padding since we add rows/columns on each sides of the image.
        padded_img[int(padding): int(-padding),int(padding): int(-padding)] = img

    else:
        padded_img = img


    #Finally the convolution part(Convolving step in the try/except only)
    for col in range(img.shape[1]):
        if col > img.shape[1] - kernel.shape[1]: #Check if kernel is outside image
            break
        if col % strides == 0:
            for row in range(img.shape[0]):
                if row > img.shape[0] - kernel.shape[0]:
                    break
                try:
                    #Only convolve when the kernel has moved the specified number of strides
                    if row % strides == 0:
                        img_out[row, col] = (kernel * padded_img[row: row + kernel.shape[0], col: col + kernel.shape[1]]).sum()
                except:
                    break

    return img_out


#Calculates the partial derivatives and magnitude of the image gradient

def partial_derivatives(img):
    #Partial derivatives in x and y direction
    img_x = np.zeros_like(img)
    img_y = np.zeros_like(img)
    CD_kernel = np.array([0.5, 0, -0.5])
    for row in range(img.shape[0]): img_x[row, :] = np.convolve(img[row, :], CD_kernel, mode = 'same')
    for col in range(img.shape[1]): img_y[:, col] = np.convolve(img[:, col], CD_kernel, mode = 'same')
    
    #Magnitude of the image gradient
    img_magnitude = np.sqrt(img_x**2 + img_y**2)

    return img_x, img_y, img_magnitude

    
def gaussian_smoothing(img, sigma):
    #Creating a 1D gaussian first
    h = int(np.ceil(sigma))
    x = np.linspace(-h, h, 2*h +1) #Need 2*h+1 elements since we have to include x = 0
    g = 1./np.sqrt(2*np.pi*sigma**2)*np.exp(-x**2/(2*sigma**2))

    #Since the gaussian is separable we can use the 1D version on a 2D image if we convolve both over the rows, and then the columns
    blurred_img = np.zeros_like(img)
    for row in range(img.shape[0]): blurred_img[row, :] = np.convolve(img[row, :], g, mode = 'same')
    for col in range(img.shape[1]): blurred_img[:, col] = np.convolve(blurred_img[:, col], g, mode = 'same')

    return blurred_img

#Finds and returns the pixel coordinates and orientation of pixels that have gradient magnitude higher than threshold
def extract_edges(img, threshold):
    Ix, Iy, Im = partial_derivatives(img)

    y, x = np.nonzero(Im > threshold)
    theta = np.arctan2(Iy[y, x], Ix[y, x])

    return x, y, theta



            