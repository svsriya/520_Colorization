######################## Created by Sriya Vudata (sv520) #########################

########### Import Statements ############
import math
import numpy as np
import time
from matplotlib import image
from matplotlib import pyplot
from random import randint
import analysis as an
from improved import improved_agent
from improved import grayscale

############################### Basic Agent ####################################

# returns euclidean distance between pixels
def dist(p1, p2):
    return np.linalg.norm(p1-p2)

# returns whether a given pixel is in list
def contains_p(list, p):
    for p2 in list:
        if np.array_equal(p,p2):
            return True
    return False

# sorts the pixels in left_img into clusters
# returns pixel_dict (the mapping of pixel to representative color) and cluster lists
def sort_clusters(left_img, centers, pixel_dict, k):
    clust_dict = {i:[] for i in range(k)}

    for r in range(len(left_img)):
        for c in range(len(left_img[0])):
            # (cluster num, distance)
            min = (0, math.inf)
            # finds the center that the pixel is closest to
            for i in range(len(centers)):
                new_dist = dist(centers[i], left_img[r][c])
                if new_dist < min[1]:
                    min = (i, new_dist)
            # set the cluster number for the pixel
            pixel_dict[(r,c)] = min[0]
            # add the pixel to the cluster
            clust_dict[min[0]].append(left_img[r][c])
    return pixel_dict, clust_dict

# given cluster, returns a new center
def new_center(cluster):
    # average the pixels in the cluster
    vect_sum = np.sum(cluster, axis=0)
    center = np.true_divide(vect_sum, len(cluster))
    return center

# returns the representative colors from the data
def kmeans(k, left_img):
    width = len(left_img)
    length = len(left_img[0])
    centers = []
    # keeps mapping of each pixel's index and what cluster its part of
    pixel_dict = {}

    # randomly select k centers
    while len(centers) != k:
        r = randint(0,width-1)
        c = randint(0, length-1)
        if not contains_p(centers, left_img[r][c]):
            centers.append(left_img[r][c])

    diff = 100
    j = 0
    # sorts pixels into clusters and recalibrates the centers
    while diff != 0:
        j+=1
        # separate pixels into clusters
        pixel_dict, clust_dict = sort_clusters(left_img, centers, pixel_dict, k)
        # for each cluster, compute a new center by averaging the points in the cluster
        old = centers.copy()
        centers.clear()
        for i in range(k):
            clr = clust_dict[i]
            c_prime = new_center(clr)
            centers.append(c_prime)
        diff = np.absolute(np.subtract(old, centers))
        diff = np.sum(diff)
    print("Iterations for convergence: " + str(j))

    # return the k centers as the representative colors and the clusters
    return centers, pixel_dict

# tr is left half of image
# ts is right half of image
# n is number of nearest neighbors to find
# pixel_dict contains info on the representative colors for training pixels
# num_colors is the number of representative colors
def nearest_neighbor(tr, ts, n, pixel_dict, num_colors):
    # neighbors: stores ((row,column), dist)
    nbr = []
    r_dict = {}

    # iterate through each of the pixels in the testing data
    for r1 in range(len(ts)):
        for c1 in range(len(ts[0])):
            # if edge pixel, set it as default color
            if r1 == 0 or r1 == len(ts)-1 or c1 == 0 or c1 == len(ts[0])-1:
                r_dict[(r1,c1)] = 0
                continue
            # create the 3x3 graypatch for the testing pixel
            gray1 = ts[np.ix_(list(range(r1-1,r1+2)),list(range(c1-1,c1+2)))]
            # iterate through pixels in the training data to find nearest neighbors
            for r2 in range(1,len(tr)-1, 3):
                for c2 in range(1,len(tr[0])-1, 3):
                    # create 9x9 graypath for the training pixel
                    gray2 = tr[np.ix_(list(range(r2-1,r2+2)),list(range(c2-1,c2+2)))]
                    diff = dist(gray1, gray2)
                    # check if dist is smaller and add to 6 neighbors
                    if len(nbr) < n:
                        nbr.append( ((r2,c2), diff) )
                    else:
                        # find min
                        min = nbr[0]
                        for i in range(n):
                            if nbr[i][1] < min[1]:
                                min = nbr[i]
                        # update nearest neighbors
                        if min[1] > diff:
                            nbr.remove(min)
                            nbr.append(((r2,c2), diff))
            # choose the representative color with the most "votes"
            vote = [0]*num_colors
            for ((r,c), _ ) in nbr:
                vote[pixel_dict[(r,c)]] += 1
            # if theres a tie, choose the most similar neighbor
            if len(vote) != len(set(vote)):
                min = nbr[0]
                for i in range(n):
                    if nbr[i][1] < min[1]:
                        min = nbr[i]
                r_dict[(r1,c1)] = pixel_dict[min[0]]
            else:
                r_dict[(r1,c1)] = np.argmax(vote)
            nbr.clear()
    return r_dict

# recolors the given image with the representative colors, using the color
# specified in dict for each pixel
def recolor(img, rep_colors, dict):
    # iterate through the pixels and recolor based on rep color
    temp = [[0 for i in range(len(img[0]))] for j in range(len(img))]
    for r in range(len(img)):
        for c in range(len(img[0])):
            color = rep_colors[dict[(r,c)]]
            temp[r][c] = np.array(color)
    return np.array(temp)

# basic agent for colorization
def basic_agent(img, k, n):
    print("Running basic agent...")
    f, axes = pyplot.subplots(1,3)
    # display the original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    # display the grayscale image
    gray_img = grayscale(img)
    axes[1].imshow(gray_img, cmap='gray')
    axes[1].set_title('Grayscaled Image')

    # TRAINING
    ## run k-means algo on left half of image to find the k best representative colors
    left_img, right_img = np.hsplit(img, 2)
    rep_colors, pixel_dict = kmeans( k, left_img)

    ## recolor the left half of the picture
    left_recolor = recolor(left_img, rep_colors, pixel_dict)
    left_recolor = left_recolor.astype('uint8')

    # TESTING
    ## for each 3x3 grayscale patch, find the six most similar patches in training data
    ## pick a representative color out of those six
    # n is number of nearest neighbors
    gray_l = grayscale(left_img)
    gray_r = grayscale(right_img)
    r_dict = nearest_neighbor(gray_l, gray_r, n, pixel_dict, len(rep_colors))

    ## color the middle pixel of the testing patch
    right_recolor = recolor(right_img, rep_colors, r_dict)
    right_recolor = right_recolor.astype('uint8')

    new_img = np.concatenate((left_recolor, right_recolor), axis=1)

    # display the array of pixels as an image
    axes[2].imshow(new_img)
    axes[2].set_title('Basic Agent (' + str(k) + ' colors, ' + str(n) + ' neighbors)')
    return new_img

################################# Main #####################################

if __name__ == '__main__':
    # loads image as a pixel array
    pic_path = input("Enter the path to the picture: ")
    img = image.imread(pic_path)
    cmd = input("Enter a letter corresponding to a command:\n(B)asic Agent\n(I)mproved Agent\n(C)ompare both\n(E)xit\n")

    while cmd is not "E":
        if cmd is "B":
            ex = input("Vary representative colors? ")
            if ex is "y":
                n = int(input("Enter ending point from 1: "))
                k = 5
                for i in range(1,n):
                    print("n = " + str(i))
                    b_img = basic_agent(img, k, i)
                    print("Average difference: " + str(an.pix_diff(img, b_img)))
            else:
                k = int(input("Enter the number of representative colors: "))
                n = int(input("Enter the number of nearest neighbors: "))
                start = time.perf_counter()
                b_img = basic_agent(img, k, n)
                end = time.perf_counter()
                print(f"Elapsed Time: {end - start:0.2f} seconds")
                print("Average difference: " + str(an.pix_diff(img, b_img)))
            pyplot.show()
        if cmd is "I":
            a = float(input("Enter a step size (alpha): "))
            e = int(input("Enter the number of times to go through training data (epoch): "))
            start = time.perf_counter()
            improved_agent(img, a, e)
            end = time.perf_counter()
            print(f"Elapsed Time: {end - start:0.2f} seconds")
            #print("Average difference: " + str(an.pix_diff(img, b_img)))
            pyplot.show()
        cmd = input("\nEnter a letter corresponding to a command:\n(B)asic Agent\n(I)mproved Agent\n(E)xit\n")
