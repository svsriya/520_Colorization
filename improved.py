######################## Created by Sriya Vudata (sv520) #########################

########### Import Statements ############
from random import uniform
from random import randint
import numpy as np
import math
from matplotlib import pyplot
from neuralnet import Node, Layer
##################### Common Functions (between agents) ########################

# converts the given image vector into a grayscale image
def grayscale(img):
    return np.dot(img[...,:3], [0.21, 0.72, 0.07])

############################# Improved Agent ##################################

# ## used in linear regression, which ended up not working and was thus not used
# def grad_descent(data, actual, epoch, alpha, in_dim, out_dim):
#     # initialize weight vector close to the zero vector (1x3)
#     w = np.array([uniform(0,0.01) for _ in range(in_dim[1])])
#     w[0] = w[0] * 5
#     # run the training through the data set epoch number of times
#     for _ in range(epoch):
#         mid = int(in_dim[0]/2)
#         # choose random index
#         r = randint(mid, len(data)-mid-1)
#         c = randint(mid, len(data[0])-mid-1)
#         print("(r,c) = " + str((r,c)))
#         print("w = " + str(w))
#         # get the X matrix of gray values
#         X = data[np.ix_(list(range(r-mid,r+mid+1)),list(range(c-mid,c+mid+1)))]
#         # compute the predicted color for l_img[r][c]
#         pred = np.matmul(w, X)
#         act = actual[r][c]
#         print("pred = " + str(pred))
#         print("act = " + str(act))
#         # compute the loss
#         mse = ((pred-act)**2).mean(axis=0)
#         print("MSE: " + str(mse))
#         # compute the partial difference
#         partial = 0
#         for i in range(len(pred)):
#             column = np.sum(X[:,i])
#             partial += (column*(pred[i]-act[i]))
#         partial *= (2/float(len(pred)))
#         #print("Partial = " + str(partial))
#         # compute new weight vector
#         w = w-(alpha*partial)
#     return w

## below is a neural network implementation using logistic regression

# gets weight vector and input data
def sigmoid(w, x):
    # dot sum the vectors
    # first add the bias term
    sum = w[-1]
    for i in range(len(x)):
        sum += (w[i]*x[i])
    # compute the sigmoid value
    sig = 1.0 + math.exp(-sum)
    return (1.0/sig)

# forward propogates the neural network
# nn: the neural network
# x: the given input data
def fwd_prop(nn, x):
    prev_out = x
    # iterate through each layer
    for l in nn:
        ins = []
        #print(prev_out)
        # iterate through each node in the layer and compute out
        for node in l.nodes:
            node.out = sigmoid( node.w, prev_out )
            ins.append(node.out)
        #ins.append(1)
        # set the previous layer's output as the current layer's input
        prev_out = ins
    # should return output of 1x3 vector
    return prev_out

# derivative function of sigmoid, used in backpropogation
def sig_der(out):
    return (1.0-out)*out

# backpropogation for training
# nn: the neural network
# y: actual output color
def back_prop(nn, y):
    # work our way backwards in the network
    for i in reversed(range(len(nn))):
        layer = nn[i]
        diff = []
        # if at the output layer, compute the first loss by iterating through nodes
        if i == len(nn)-1:
            for j in range(len(layer.nodes)):
                diff.append((layer.nodes[j].out-y[j]))
        # otherwise, use the derivative of the layer in front for each node
        else:
            for j in range(len(layer.nodes)):
                err = 0.0
                front_layer = nn[i+1]
                for node in front_layer.nodes:
                    err += (node.w[j]*node.derivative)
                diff.append(err)
        # for each node in layer, multiple accumulated error by sigma prime
        for j in range(len(layer.nodes)):
            node = layer.nodes[j]
            node.derivative = diff[j]*sig_der(node.out)

# updates the weights for each layer
# nn: the neural network
# x: input vector
# alpha: step size
def update_w(nn, x, alpha):
    for i in range(len(nn)):
        layer = nn[i]
        # update the bias separately
        # the first layer's inputs is the input vector
        ins = []
        if i == 0:
            ins = x[:-1]
        # otherwise the input to the layer is previous layer's output
        else:
            ins = [node.out for node in nn[i-1].nodes]
        # update each node in current layer
        for node in layer.nodes:
            for j in range(len(ins)):
                node.w[j] -= (alpha*node.derivative*ins[j])
            # update bias
            node.w[-1] -= (alpha*node.derivative)

# improved agent for colorization (currently using neural network)
def improved_agent(img, alpha, epoch):
    print("Running improved agent...")
    f, axes = pyplot.subplots(1,3)
    # display the original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    # display the grayscale image
    gray_img = grayscale(img)
    axes[1].imshow(gray_img, cmap='gray')
    axes[1].set_title('Grayscaled Image')
    # split the gray and original image into 2
    l_img, r_img = np.hsplit(img, 2)
    l_gr, r_gr = np.hsplit(gray_img, 2)

    # initialize the neural network as a list of layers
    # first hidden layer and output layer have 3 nodes for 3 color values
    # input: 1x9 vector
    # output: 1x3 vector
    nn = [None]*2
    nn[0] = Layer(n=50,inn=9)
    nn[1] = Layer(n=3,inn=50)
    # nn[2] = Layer(n=3, inn=10)
    for l in nn:
        for i in range(l.num_nodes):
            l.nodes[i] = Node(l.num_ins)

    ## TRAINING - using SGD
    for e in range(epoch):
        # choose random data point
        r = randint(1, len(l_gr)-2)
        c = randint(1, len(l_gr[0])-2)
        # 1x9 vector
        x_mat = l_gr[np.ix_(list(range(r-1,r+2)),list(range(c-1,c+2)))]
        x_flat = x_mat.flatten()
        x = np.multiply(x_flat,float(1/255))
        #x = np.append(x,1)
        #print("x: " + str(x))
        # actual color
        actual = np.multiply(l_img[r][c],float(1/255))
        # get the predicted color from forward propogation
        pred_color = fwd_prop(nn, x)
        diff = (pred_color-actual)
        err = 0.0
        for i in range(len(diff)):
            err += (diff[i]**2)
        err /= 3.0
        # update the weight vectors
        back_prop(nn, actual)
        update_w(nn, x, alpha)
        if e % 1000 == 0:
            print("step " + str(e))
            print("x: " + str(x))
            print("actual: " + str(actual) )
            print("pred: " + str(pred_color) )
            print("error: " + str(err))


    # TESTING - use the model to recolor image
    new_img = []
    for r in range(len(r_gr)):
        new_img.append([])
        for c in range(len(r_gr[0])):
            if r == 0 or r == len(r_gr)-1 or c == 0 or c == len(r_gr[0])-1:
                new_img[r].append([0,0,0])
                continue
            x_mat = r_gr[np.ix_(list(range(r-1,r+2)),list(range(c-1,c+2)))]
            x_flat = x_mat.flatten()
            x = np.multiply(x_flat,float(1/255))
            #x = np.append(x,1)
            # compute the predicted color for l_img[r][c]
            color = fwd_prop(nn, x)
            # set the color
            new_img[r].append((np.multiply(color,225)))
    new_img = np.array(new_img).astype('uint8')
    axes[2].imshow(new_img)
    axes[2].set_title('Improved Agent')
    return new_img
