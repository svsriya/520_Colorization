######################## Created by Sriya Vudata (sv520) #########################

# this file contains functions used to analyze the results of each agent

import numpy as np

# given the original image and the new image, calculates the differences
# between the original pixel's color and the new pixel's color
def pix_diff(og_img, new_img):
    diff = np.subtract(og_img, new_img)
    vect_sum = np.sum(diff, axis=(0,1))
    return np.true_divide(vect_sum, len(new_img)*len(new_img[0]))
