import numpy as np 
import matplotlib.pyplot as plt
from Tools import*

img_path = 'Images/kunst.jpg'

#Setting parameters
threshold = 0.1
sigma = 1

RGB = plt.imread(img_path)
RGB = RGB/255.0 #Normalizing
gray = rgb_to_gray(RGB)
blurred_img = gray#gaussian_smoothing(gray, sigma)
Ix, Iy, Im = partial_derivatives(blurred_img)

#Edge pixel coordinates and orientation
x, y, theta = extract_edges(blurred_img, threshold)


fig, axes = plt.subplots(1,5,figsize=[15,4], sharey='row')
plt.set_cmap('gray')
axes[0].imshow(blurred_img)
axes[1].imshow(Ix)
axes[2].imshow(Iy)
axes[3].imshow(Im)
edges = axes[4].scatter(x, y, s=1, c=theta, cmap='hsv')
fig.colorbar(edges, ax=axes[4], orientation='horizontal', label='$\\theta$ (radians)')
for a in axes:
    a.set_xlim([0, RGB.shape[0]])
    a.set_ylim([RGB.shape[0], 0])
    a.set_aspect('equal')
axes[0].set_title('Blurred input')
axes[1].set_title('Gradient in x')
axes[2].set_title('Gradient in y')
axes[3].set_title('Gradient magnitude')
axes[4].set_title('Extracted edges')
plt.tight_layout()
# plt.savefig('out_edges.png') # Uncomment to save figure in working directory
plt.show()