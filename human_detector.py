import math
import os
import glob
import cv2
import numpy as np


# Conversion to greyscale

def greyconversion(img):

    rows, columns, arr = np.shape(img)
    greyimg = np.zeros((rows, columns), dtype=int)

    for i in range(rows):
        for j in range(columns):
            greyimg[i, j] = round((0.299 * img[i, j, 2]) + (0.587 * img[i, j, 1]) + (0.114 * img[i, j, 0]))

    return greyimg


# Gradient operation

def gradient_opt(gimg):

    kernelx = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)])
    kernely = np.array([(1, 2, 1), (0, 0, 0), (-1, -2, -1)])

    gimgn = gimg.shape[0] - 1
    gimgm = gimg.shape[1] - 1
    gximg = np.zeros_like(gimg, dtype=float)
    gyimg = np.zeros_like(gimg, dtype=float)
    gmimg = np.zeros_like(gimg)
    gaimg = np.zeros_like(gimg, dtype=float)

    # Convolution for x-gradient

    for i in range(1, gimgn):
        for j in range(1, gimgm):
            gximg[i, j] = ((kernelx * gimg[i - 1:i + 2, j - 1:j + 2]).sum()) / 4 #Normalised

    # Convolution for y-gradient

    for i in range(1, gimgn):
        for j in range(1, gimgm):
            gyimg[i, j] = ((kernely * gimg[i - 1:i + 2, j - 1:j + 2]).sum()) / 4 #Normalised

    # Gradient magnitude calcultaion

    for i in range(1, gimgn):
        for j in range(1, gimgm):
            gmimg[i, j] = np.round(math.sqrt((gximg[i, j] * gximg[i, j]) + (gyimg[i, j] * gyimg[i, j])))

    # Gradient angle calculation

    for i in range(1, gimgn):
        for j in range(1, gimgm):
            if gximg[i, j] != 0:
                gaimg[i, j] = np.degrees(np.arctan((gyimg[i, j] / gximg[i, j])))
                if gaimg[i, j] < 0:
                    gaimg[i, j] += 360
            else:
                if gyimg[i, j] < 0:
                    gaimg[i, j] = 270
                elif gyimg[i, j] > 0:
                    gaimg[i, j] = 90
                else:
                    gaimg[i, j] = 0


    return gmimg, gaimg


# HOG feature

def hog_feature(gmimg, gaimg):

    gmimgn = gmimg.shape[0]
    gmimgm = gmimg.shape[1]
    hog = []
    for i in range(0, gmimgn, 8):
        hogrow = []
        for j in range(0, gmimgm, 8):
            bin = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for x in range(8):
                for y in range(8):
                    k = gaimg[i + x, j + y]
                    m = gmimg[i + x, j + y]
                    if 170 <= k < 350:
                        k -= 180
                    elif 350 <= k <= 360:
                        k -= 360
                    if -10 <= k < 0:
                        bin [8] += ((0 - k) / 20) * m
                        bin [0] += ((20 + k) / 20) * m
                    elif 160 <= k < 170:
                        bin[8] += ((180 - k) / 20) * m
                        bin[0] += ((k - 160) / 20) * m
                    elif 0 < k < 20:
                        bin[0] += ((20 - k) / 20) * m
                        bin[1] += ((k - 0) / 20) * m
                    elif 20 < k < 40:
                        bin[1] += ((40 - k) / 20) * m
                        bin[2] += ((k - 20) / 20) * m
                    elif 40 < k < 60:
                        bin[2] += ((60 - k) / 20) * m
                        bin[3] += ((k - 40) / 20) * m
                    elif 60 < k < 80:
                        bin[3] += ((80 - k) / 20) * m
                        bin[4] += ((k - 60) / 20) * m
                    elif 80 < k < 100:
                        bin[4] += ((100 - k) / 20) * m
                        bin[5] += ((k - 80) / 20) * m
                    elif 100 < k < 120:
                        bin[5] += ((120 - k) / 20) * m
                        bin[6] += ((k - 100) / 20) * m
                    elif 120 < k < 140:
                        bin[6] += ((140 - k) / 20) * m
                        bin[7] += ((k - 120) / 20) * m
                    elif 140 < k < 160:
                        bin[7] += ((160 - k) / 20) * m
                        bin[8] += ((k - 140) / 20) * m
                    elif k == 20:
                        bin[1] += k
                    elif k == 40:
                        bin[2] += k
                    elif k == 60:
                        bin[3] += k
                    elif k == 80:
                        bin[4] += k
                    elif k == 100:
                        bin[5] += k
                    elif k == 120:
                        bin[6] += k
                    elif k == 140:
                        bin[7] += k
                    elif k == 160:
                        bin[8] += k
            hogrow.append(bin)
        hog.append(hogrow)

    icells = gmimgn // 8
    jcells = gmimgm // 8
    bhog = []
    block = []
    for i in range(icells - 1):
        bhogrow = []
        for j in range(jcells - 1):
            block = hog[i][j] + hog[i][j + 1] + hog[i + 1][j] + hog[i + 1][j + 1]
            res = sum(map(lambda i: i * i, block))
            res = math.sqrt(res)
            if res != 0:
                block[:] = [x / res for x in block]     #Noramalised
            bhogrow.append(block)
        bhog.append(bhogrow)
    hog_vector = []
    for i in range(len(bhog)):
        for j in bhog[i]:
            hog_vector += j
    return hog_vector


# LBP feature:

def lbp_feature(gimg):

    gimgn = gimg.shape[0] - 1
    gimgm = gimg.shape[1] - 1
    lbpimg = np.full_like(gimg, 5)

    for i in range(1, gimgn):
        for j in range(1, gimgm):
            str = ""
            if gimg[i - 1, j - 1] > gimg[i, j]:
                str += "1"
            else:
                str += "0"
            if gimg[i - 1, j] > gimg[i, j]:
                str += "1"
            else:
                str += "0"
            if gimg[i - 1, j + 1] > gimg[i, j]:
                str += "1"
            else:
                str += "0"
            if gimg[i, j + 1] > gimg[i, j]:
                str += "1"
            else:
                str += "0"
            if gimg[i + 1, j + 1] > gimg[i, j]:
                str += "1"
            else:
                str += "0"
            if gimg[i + 1, j] > gimg[i, j]:
                str += "1"
            else:
                str += "0"
            if gimg[i + 1, j - 1] > gimg[i, j]:
                str += "1"
            else:
                str += "0"
            if gimg[i, j - 1] > gimg[i, j]:
                str += "1"
            else:
                str += "0"

            lbpimg[i, j] = int(str, 2)

    lbp = []
    binlist = [0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60, 62, 63, 64, 96, 112, 120, 124,
               126, 127, 128, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240,
               241, 243, 247, 248, 249, 251, 252, 253, 254, 255, 'x']
    for i in range(0, gimgn + 1, 16):
        block = []
        for j in range(0, gimgm + 1, 16):
            bindict = dict.fromkeys(binlist, 0)
            for x in range(16):
                for y in range(16):
                    z = lbpimg[i + x, j + y]
                    if z in binlist:
                        bindict.update({z: bindict.get(z) + 1})
                    else:
                        bindict.update({'x': bindict.get('x') + 1})
            block.append(list(bindict.values()))
        lbp.append(block)

    lbp_vector = []
    for i in range(len(lbp)):
        for j in lbp[i]:
            lbp_vector += j
    lbp_vector[:] = [x / 256 for x in lbp_vector]    #Normalised

    return lbp_vector

#Feature extraction:

def feature_extract(imgpath, choice):

    img = cv2.imread(imgpath, 1)
    greyimg = greyconversion(img)
    gmimg, gaimg = gradient_opt(greyimg)
    hog_vector = hog_feature(gmimg, gaimg)
    if choice == 1:
        lbp_vector = lbp_feature(greyimg)
        feature_vector = hog_vector + lbp_vector
    else:
        feature_vector = hog_vector

    return feature_vector

#Relu function:

def relu(x):

    n, m = x.shape
    y = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            y[i,j] = max(0,x[i,j])
    return y

#Sigmod function:

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

#Neural negtwork:

def network(inputlist,output,neurons,alpha,epoch):

    n, m = inputlist.shape
    a = n
    b = m
    x, y = output.shape
    c = y

    # Random initializing of weights
    w1 = np.random.randn(b, neurons)
    w1 = np.multiply(w1, math.sqrt(2 / int(b + neurons)))

    b1 = np.random.randn(neurons)
    b1 = np.multiply(b1, math.sqrt(2 / int(neurons)))

    # declared the weights for layer 2 and normalize it according to the dimensions
    w2 = np.random.randn(neurons, c)
    w2 = np.multiply(w2, math.sqrt(1 / int(neurons + c)))

    b2 = np.random.randn(c)
    b2 = np.multiply(b2, math.sqrt(1 / int(c)))

    for n in range(epoch):
        sq_err_sum = np.zeros((1, c))
        for k in range(a):
            ak = inputlist[k, :].reshape([1, -1])    #x=ak z=aj y= ai
            aj = relu((ak.dot(w1) + b1))
            ai = sigmoid(aj.dot(w2) + b2)
            err = (output[k, 0] - ai)
            sq_err_sum += 0.5 * np.square(err)

            # Delta for ouput layer
            deti = (-1 * err) * (1 - ai) * ai
            detw2 = aj.T.dot(deti)
            detb2 = np.sum(deti, axis=0)

            #Relu differential
            
            d = np.zeros_like(aj)
            for j in range(neurons):

                if (aj[0][j] > 0):
                    d[0][j] = 1
                else:
                    d[0][j] = 0

            # Delta for hidden layer
            detj = deti.dot(w2.T) * d
            detw1 = ak.T.dot(detj)
            detb1 = np.sum(detj, axis=0)

            # Updating weights
            w2 -= alpha * detw2
            b2 -= alpha * detb2
            w1 -= alpha * detw1
            b1 -= alpha * detb1

        # Stop training if error less than 0.01
        if (np.mean(sq_err_sum) / a < 0.01):
            break


    return w1, b1, w2, b2

#Training network:

def train_network(imgpathpos, imgpathneg, neurons, choice):

    posimglist = glob.glob(imgpathpos + "/*")
    negimglist = glob.glob(imgpathneg + "/*")
    alpha = 0.2
    inputlist = []
    output = [[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]]
    epoch = 200

    for i in range(len(posimglist)):
        inputlist.append(feature_extract(posimglist[i], choice))
        inputlist.append(feature_extract(negimglist[i], choice))

    inputlist = np.array(inputlist)
    output = np.array(output)
    w1, b1, w2, b2 = network(inputlist,output,neurons,alpha,epoch)

    return w1,b1,w2,b2

#Testing network:

def test_network(imgtestpathpos, imgtestpathneg, w1, b1, w2, b2, choice):

    posimglist = glob.glob(imgtestpathpos + "/*")
    negimglist = glob.glob(imgtestpathneg + "/*")
    inputlist = []

    for i in range(len(posimglist)):
        inputlist.append(feature_extract(posimglist[i], choice))
    for i in range(len(negimglist)):
        inputlist.append(feature_extract(negimglist[i], choice))

    # Printing normalised magnitude images

    for i in range(len(negimglist)):
        img = cv2.imread(negimglist[i], 1)
        greyimg = greyconversion(img)
        gmimg, gaimg = gradient_opt(greyimg)
        name = "img"+str(i+len(posimglist))+".bmp"
        cv2.imwrite(name,gmimg)

    for i in range(len(posimglist)):
        img = cv2.imread(posimglist[i], 1)
        greyimg = greyconversion(img)
        gmimg, gaimg = gradient_opt(greyimg)
        name = "img"+str(i)+".bmp"
        cv2.imwrite(name,gmimg)

    inputlist = np.array(inputlist)
    p = []
    n,m = inputlist.shape
    for k in range(n):
        ak = inputlist[k, :].reshape([1, -1])
        aj = relu((ak.dot(w1) + b1))
        ai = sigmoid(aj.dot(w2) + b2)
        p.append(ai[0][0])
    print(p)
    for i in p:
        if i <= 0.4:
            print("No human")
        elif 0.4 < i < 0.6:
            print("Boderline")
        else:
            print("Human")

#Main fucntion:

if __name__ == "__main__":

    imgpathpos = input("Enter the path for the folder of positive images(contains humans) for training")
    imgpathneg = input("Enter the path for the folder of negative images(does not contain human) for training")
    imgtestpathpos = input("Enter the path to the positive test image:")
    imgtestpathneg = input("Enter the path to the negative test image:")
    neurons = int(input("Enter number of neurons in hidden layer"))
    choice = int(input("Enter 1 to include lbp feature"))
    if not (os.path.exists(imgpathneg)) or not (os.path.exists(imgpathpos) or not (os.path.exists(imgtestpathpos) or not (os.path.exists(imgtestpathneg)))):
        print("Image path is incorrect")
    else:
        w1,b1, w2, b2 = train_network(imgpathpos, imgpathneg, neurons, choice)
        test_network(imgtestpathpos,imgtestpathneg, w1, b1,w2, b2, choice)
    print("Program terminated")
