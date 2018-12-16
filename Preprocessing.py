import skimage.io as io
import skimage.color as colors
import skimage.filters as filters
import skimage.feature as feature
import matplotlib.pyplot as graphs
import numpy as np
import cv2
import FLAGS
import os
import keeper
import datetime
import time
# IMPORTANT NOTE:
# all points in my code are expressed as tuples of format (y, x)
# all rectangles, expect those returned by build-in openCV functions, are defined as tuple of 2 points (P1,P2)
# where P1 is the top-left point and P2 is the right-bottom point

blocks_dict = {}
if FLAGS.CACHE_BLOCKS:
    blocks = keeper.get_tensor_list_dict_from_disk(FLAGS.TEXTURE_BLOCKS_LOG_PATH)


def store_texture_blocks_of_list(form_filenames_list, log_filename=None):
    if log_filename is None:
        log_filename = 'texture_blocks' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '.') + '.txt'

    keeper.store_tensor_list(form_filenames_list, get_texture_blocks, log_filename)


def store_all_texture_blocks(log_filename=None):
    forms_filenames = [n for n in os.listdir(FLAGS.DEFAULT_DATASET_PATH)]
    store_texture_blocks_of_list(forms_filenames, log_filename)


def get_texture_blocks(form_filename, dataset_directory=FLAGS.DEFAULT_DATASET_PATH):
    if form_filename in blocks_dict:
        return blocks_dict[form_filename]
    else:
        form_image = io.imread(os.path.join(dataset_directory, form_filename))
        return Preprocessing(form_image)


# Preprocessing Step:
# Inputs:
# I: Raw Image
# [line_spacing = 1/2]: spacing between each line multiplied by average height of the line
# [block_size = (128, 256)]: Block size (height, width).
# [IAM_dataset = True]: set this boolean to True if we are using IAM dataset, False otherwise 
# Outputs:
# List of 2D binary Blocks

def Preprocessing(I, line_spacing=1/2, block_size=(128, 256), IAM_dataset = True):
    # Load image and covert it to grayscale
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    
    #Fix Contrast
    if np.max(I)<=1:
        I = np.array(255*I,dtype = np.uint8)

    #IAM Specific: Crop hand written part from the image
    if IAM_dataset:
        I = I[900:2800,:]
    (h, w) = np.shape(I)

    # Convert to binary image using adaptive thersholding technique (Otsu Method)
    th,Ibw = cv2.threshold(I,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #I[I >= th] = 255

    
    # fix scanning problems
    if IAM_dataset:
        Ibw[:, :50] = 255
        Ibw[:, w - 50:] = 255
      
    ''' CONNECTED COMPOMENTS + Texture Blocks'''
    # Using same procedure (not fully) explained in the paper
    texture_blocks = []

    # save widths so we will able to get the minimum
    widths = []

    # Find connected components using build-in OpenCV function
    contours = cv2.findContours(255 - Ibw , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    margin_y = 200
    margin_x = 50

    start_y = margin_y
    start_x = margin_x
    h_total = 0
    n = 0

    I_Compact = 255 - np.zeros((h, w), dtype=np.uint8)

    for cnt in contours[::-1]:
        (x, y, ww, hh) = cv2.boundingRect(cnt)

        # Condition to discard small dots, ticks, etc.
        if ww * hh > 100:
            # Find the center of mass
            mu = cv2.moments(I[y:y + hh, x:x + ww])
            center = ((mu['m01'] / mu['m00']), (mu['m10'] / mu['m00']))

            begin_y = start_y - int(np.ceil(center[0]))

            # Elementwise OR on negative image
            I_Compact[begin_y:begin_y + hh, start_x:start_x + ww] = np.minimum(I[y:y + hh, x:x + ww],
                                                                               I_Compact[begin_y:begin_y + hh,
                                                                               start_x:start_x + ww])


            start_x = start_x + ww
            h_total = h_total + hh
            n = n + 1

            # Start new line in the compact image
            if start_x >= block_size[1] + margin_x:
                widths.append(start_x)
                start_y = start_y + int(line_spacing * h_total / n)
                start_x = margin_x
                h_total = 0
                n = 0
                if start_y >= block_size[0] + margin_y:
                    end_x = np.min(widths)
                    end_y = start_y
                    start_y = margin_y
                    I_Compact2 = I_Compact[margin_y:min(block_size[0] + margin_y, end_y),
                                 margin_x:min(block_size[1] + margin_x, end_x)].copy()
                    texture_blocks.append(I_Compact2)
                    I_Compact.fill(255)

    ''' CONNECTED COMPOMENTS END '''

    return texture_blocks

# Modeule test
if __name__ == "__main__":
    start = time.process_time()
    blocks = Preprocessing(io.imread(r"data\01\test.PNG"))
    blocks = Preprocessing(io.imread(r"data\01\1\1.PNG"))
    blocks = Preprocessing(io.imread(r"data\01\1\2.PNG"))
    blocks = Preprocessing(io.imread(r"data\01\2\1.PNG"))
    blocks = Preprocessing(io.imread(r"data\01\2\2.PNG"))
    blocks = Preprocessing(io.imread(r"data\01\3\1.PNG"))
    blocks = Preprocessing(io.imread(r"data\01\3\2.PNG"))
    end = time.process_time()
    print((end-start))
    #for block in blocks:
    #    io.imshow(block)
    #    io.show()