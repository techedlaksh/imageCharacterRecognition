# Import the modules
import cv2
from sklearn.externals import joblib
import numpy as np
import sys, getopt
from keras.models import load_model

def main(argv):
    inp_pic = "timages/"
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:  
        print 'test.py -i <inputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inp_pic += arg

    # Load the Keras CNN trained model
    model = load_model('tmodels/cDigit1.h5')

    # Original image
    im = cv2.imread(inp_pic)
    cv2.imshow("Original Image", im)
    cv2.waitKey()

    ################# OLD Algorithm with Fixed Threshold #####################
    # Convert to grayscale and apply Gaussian filtering
    # im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    # Threshold the image
    # ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("Thres",im_th)
    ###########################################################################


    ################# NEW Algorithm with Adaptive Threshold #########################################
    # Read image in grayscale mode
    img = cv2.imread(inp_pic,0)
    
    # Median Blur and Gaussian Blur to remove Noise
    img = cv2.medianBlur(img,3)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive Threshold for handling lightning
    im_th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,5)
    # cv2.imshow("Threshold Image",im_th)
    kernel = np.ones((1,1),np.uint8)
    im_th = cv2.dilate(im_th,kernel,iterations = 4)
    cv2.imshow("After", im_th)
    ##################################################################################################


    # Find contours in the image
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, predict using cnn model
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Input for CNN Model
        roi = roi[np.newaxis,np.newaxis,:,:]

        # Input for Feed Forward Model
        # roi = roi.flatten()
        # roi = roi[np.newaxis]
        nbr = model.predict_classes(roi,verbose=0)
        cv2.putText(im, str(int(nbr[0])), (int(rect[0]), int(rect[1])),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    cv2.imshow("Resulting Image with Predicted numbers", im)
    cv2.waitKey()

if __name__ == "__main__":
   main(sys.argv[1:])