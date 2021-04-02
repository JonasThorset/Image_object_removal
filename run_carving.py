import numpy as np 
import matplotlib.pyplot as plt
from carvingTools import*

img_path = 'Images/city.jpeg'
RGB = plt.imread(img_path)
RGB = RGB/255.0             #Normalizing



reduced_image = carve_image_height(RGB, 100)

#Plotting stuff
plt.figure(figsize = (3*14,3*3))
plt.subplot(121)
plt.imshow(RGB)
plt.subplot(122)
plt.imshow(reduced_image)

plt.show()
