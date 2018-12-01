import skimage.io as io
import skimage.color as colors
import skimage.filters as filters
import skimage.feature as feature
import matplotlib.pyplot as graphs
import numpy as np
import cv2

#IMPORTANT NOTE: 
#all points in my code are expressed as tuples of format (y, x)
#all rectangles, expect those returned by build-in openCV functions, are defined as tuple of 2 points (P1,P2) 
#where P1 is the top-left point and P2 is the right-bottom point

#Preprocessing Step:
#Inputs:
#I: Raw Image
#Outputs:
#Binary Image containing all text compact

def Preprocessing(I):
    #Load image and covert it to grayscale
    I = colors.rgb2gray(I)

    #IAM Specific: Crop hand written part from the image
    I = I[730:2790,:]
    (h,w) = np.shape(I)

    #Convert to binary image using adaptive thersholding technique (Otsu Method)
    thershold = filters.threshold_otsu(I) 
    Ibw = np.zeros((h,w), dtype=np.uint8)
    Ibw[I >= thershold] = 1


    #fix scanning problems
    Ibw[:,:50] = 1
    Ibw[:,w - 50:] = 1


    ''' LINE ROI DETECTION BEGIN '''

    #Detect horizontal ROI using horizontal projection
    cumulativegraph_horizontal = np.zeros((h))
    for i in range(1,h):
        cumulativegraph_horizontal[i] = sum(1 - Ibw[i,:])


    #Detect vertical ROI using horizontal projection
    cumulativegraph_vertical = np.zeros((w))
    for i in range(1,w):
        cumulativegraph_vertical[i] = sum(1 - Ibw[:,i])


    #Uncomment if you want to see how horizontal and vertical projection look like

    #graphs.plot(np.r_[0:h],cumulativegraph_horizontal)
    #graphs.show()
    #graphs.plot(np.r_[0:w],cumulativegraph_vertical)
    #graphs.show()


    #Overall ROI is the intersection of both horizontal ROI and vertical ROI that gives value above certain (small for HQ scanned images) thershold
    rectangles = []
    horizon_inside = False
    vertical_inside = False
    temp_points = []

    for i in range(1,h):
        if cumulativegraph_horizontal[i] > 0.001 * w:
            horizon_inside = False
            for j in range(1,w):
                if cumulativegraph_vertical[j] > 0.001 * h:
                    #append top left point of current rectangle
                    if (not horizon_inside) and (not vertical_inside): 
                        current_rect = ((i,j),(i,j))  #take care: point is (y,x)
                        horizon_inside = True
                   
                else:
                    #update x of bottom right point of current rectangle
                    if horizon_inside and (not vertical_inside):
                        horizon_inside = False
                        current_rect = (current_rect[0],(current_rect[1][0],j))
                        temp_points.append(current_rect)
        
            vertical_inside = True
        else:
            vertical_inside = False
            #update y of bottom right point of each rectangle
            for rectangle in temp_points:   

                #compress rectangle to exactly fit in horizonatal direction
                start_x = rectangle[0][1]
                end_x = rectangle[1][1]

                line_extracted = 1 - Ibw[rectangle[0][0]:i, start_x:end_x]
                line_sigma = line_extracted.sum(axis = 0)
            
                #if rectangle is non empty append it
                if line_sigma.sum() != 0:
                    start_x = start_x + (line_sigma != 0).argmax()
                    end_x = end_x - (line_sigma[::-1] != 0).argmax()
                    rectangle = ((rectangle[0][0],start_x), (i, end_x))
                    rectangles.append(rectangle)

            temp_points.clear()
    ''' LINE ROI DETECTION END '''

    #uncomment if you want to see detected  regions of interest

    #I2 = np.array(Ibw * 255, dtype = np.uint8) 
    #for rectangle in rectangles:
    #    top_left = rectangle[0]
    #    bottom_right = rectangle[1]
    #    I2[top_left[0]:top_left[0] + 2, top_left[1]:bottom_right[1]] = 0
    #    I2[bottom_right[0]:bottom_right[0] + 2, top_left[1]:bottom_right[1]] = 0
    #    I2[top_left[0]:bottom_right[0],top_left[1]:top_left[1] + 2] = 0
    #    I2[top_left[0]:bottom_right[0],bottom_right[1]:bottom_right[1] + 2] = 0
    #io.imshow(I2)
    #io.show()


    ''' CONNECTED COMPOMENTS PER LINE (ROI) BEGIN + COMPACT IMAGE'''
    #Using same procedure (not fully) explained in the paper 
    I_Compact = np.zeros((2*h,2*w), dtype=np.bool) # How to know the exact image size ?? also wrong detected/or merged ROI sizes can cause a problem here..

    start_y = 100 

    width_per_line=0

    for line in rectangles:
        #Work on negative image extracted line by line (we know where are lines from the previous part)
        Line_extracted = 1 - Ibw[line[0][0]:line[1][0], line[0][1]:line[1][1]]
        #Find connected components using build-in OpenCV function
        contours = cv2.findContours(Line_extracted * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

        if width_per_line == 0:
            width_per_line = 0.5 * line[1][1] - line[0][1]
            start_x = 50
            h_total = 0
            n = 0
    
        #Convert Binary (negative) image to boolean so we can do logical operations on it
        Line_extracted = (Line_extracted!=0)

        for cnt in contours[::-1]:
            (x, y, ww, hh) = cv2.boundingRect(cnt)
        
            #Condition to discard small dots, ticks, etc.
            if ww * hh > 100:
                #Find the center of mass
                mu = cv2.moments(Ibw[y:y + hh, x:x + ww])
                center = ((mu['m01'] / mu['m00']), (mu['m10'] / mu['m00']))


                begin_y = start_y - int(np.ceil(center[0]))

                #Elementwise OR on negative image
                I_Compact[begin_y:begin_y + hh, start_x:start_x + ww] = np.logical_or(Line_extracted[y:y + hh, x:x + ww], I_Compact[begin_y:begin_y + hh, start_x:start_x + ww])
            
                start_x = start_x + ww
                h_total = h_total + hh
                n = n + 1
    
                if start_x>=width_per_line:
                    start_y = start_y + int(7*h_total / (8*n))
                    start_x = 50
                    h_total = 0
                    n = 0
       


    #Remove empty space
    start_x = (I_Compact.sum(axis=0) != 0).argmax()
    start_y = (I_Compact.sum(axis=1) != 0).argmax()
    end_x = 2*w - (I_Compact.sum(axis=0)[::-1] != 0).argmax()
    end_y = 2*h - (I_Compact.sum(axis=1)[::-1] != 0).argmax()
    I_Compact = I_Compact[start_y:end_y, start_x:end_x] 

    #Convert image to back normal binary image
    I_Compact = 1 - np.uint8(I_Compact)

    ''' CONNECTED COMPOMENTS END '''


    I2 = np.array(I_Compact * 255, dtype = np.uint8) 
    io.imsave('output.jpg',I2)



