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
#[line_spacing = 1/2]: spacing between each line multiplied by average height of the line
#Outputs:
#List of 2D greyscale Blocks

def Preprocessing(I, line_spacing=1/2):
    #Load image and covert it to grayscale
    I = colors.rgb2gray(I)

    #IAM Specific: Crop hand written part from the image
    I = I[730:2690,:]
    (h,w) = np.shape(I)

    #Convert to binary image using adaptive thersholding technique (Otsu Method)
    thershold = filters.threshold_otsu(I) 
    Ibw = np.zeros((h,w), dtype=np.uint8)
    Ibw[I >= thershold] = 1
    #I[I >= thershold] = 255

    #fix scanning problems
    Ibw[:,:50] = 1
    Ibw[:,w - 50:] = 1



    ''' CONNECTED COMPOMENTS + Texture Blocks'''
    #Using same procedure (not fully) explained in the paper 
    texture_blocks = []

    #save widths so we will able to get the minimum 
    widths = []

    #Find connected components using build-in OpenCV function
    contours = cv2.findContours((1-Ibw) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    start_y = 100 
    start_x = 50
    h_total = 0
    n = 0

    #Convert Binary image to negative image so we can do logical operations on it
    negative_I = 255 - I

    #Estimate width and height of I_Compact
    sum_a = 0 #sum of widths of all connected components
   

    for a in [cv2.boundingRect(cnt)[2]*cv2.boundingRect(cnt)[3] for cnt in contours]:
        if a >  100:
            sum_a += a
  
    estimate_length = int(np.sqrt(sum_a))
    
    I_Compact = np.zeros((3*estimate_length,3*estimate_length), dtype=np.uint8) 

    for cnt in contours[::-1]:
        (x, y, ww, hh) = cv2.boundingRect(cnt)
        
        #Condition to discard small dots, ticks, etc.
        if ww * hh > 100:
            #Find the center of mass
            mu = cv2.moments(I[y:y + hh, x:x + ww])
            center = ((mu['m01'] / mu['m00']), (mu['m10'] / mu['m00']))


            begin_y = start_y - int(np.ceil(center[0]))

            #Elementwise OR on negative image
            I_Compact[begin_y:begin_y + hh, start_x:start_x + ww] = np.maximum(negative_I[y:y + hh, x:x + ww], I_Compact[begin_y:begin_y + hh, start_x:start_x + ww])
            
            start_x = start_x + ww
            h_total = h_total + hh
            n = n + 1
    
            #Start new line in the compact image
            if start_x >= estimate_length:
                widths.append(start_x)
                start_y = start_y + int(line_spacing * h_total / n)
                start_x = 50
                h_total = 0
                n = 0
    
    
    end_x = np.min(widths)
    end_y = start_y

    I_Compact = I_Compact[100:end_y, 50:end_x].copy()

    ''' CONNECTED COMPOMENTS END '''

    I_Compact = 255 - np.uint8(I_Compact)
    return I_Compact



#Modeule test 
if __name__ == "__main__":
    I_Compact = Preprocessing(io.imread("a05-089.png"))
    io.imsave("output.jpg",I_Compact)
