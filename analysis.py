######################## Created by Sriya Vudata (sv520) #########################

# this file contains functions used to analyze the results of each agent

import numpy as np

# given the original image and the new image, calculates the differences
# between the original pixel's color and the new pixel's color
def pix_diff(og_img, new_img):

    vect_sum = 0.0
    for r in range(len(new_img)):
        for c in range(len(new_img[0])):
            og = og_img[r][c]
            nw = new_img[r][c]
            diff = (np.subtract(og, nw))
            vect_sum += np.dot(diff,diff)
    return float(vect_sum/(len(new_img)*len(new_img[0])))
