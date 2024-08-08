from PIL import Image
import numpy as np

#img = Image.open('ldr_sky_sphere.jpg')
img = Image.open('ldr_sky_sphere_mask.png')
img_np = np.array(img)
print(img_np.shape)
print(np.unique(img_np))

