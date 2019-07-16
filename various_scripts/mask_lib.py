import os
import numpy as np
import cv2


def is_mask(folder, value):
    """
    Check if every mask in a folder is 0 255 format
    :param folder: folder to examine
    :param value: expected value, either 1 for binary or 255 for 0-255 valued
    """
    print("Examining", folder)
    list_mask_path = os.listdir(folder)
    is_binary = True
    for mask_path in list_mask_path:
        mask = cv2.imread(folder + mask_path, cv2.IMREAD_UNCHANGED)
        mask_shape = mask.shape
        if len(mask_shape) > 2:
            is_binary = False
            print("This mask has " + str(mask_shape[2]) + " channels !", mask_path)
        if len(mask_shape) == 2:
            distinct_val = np.unique(mask)
            if (distinct_val != np.asarray(([0, value]), dtype="uint8")).any():
                is_binary = False
                print("This mask has " + str(distinct_val) + " values !", mask_path)
    if is_binary:
        print("Folder " + folder + " contains 0" + str(value) + " mask only")
    else:
        print("Folder " + folder + " does not contains 0" + str(value) + " masks only")
    return is_binary


def uint8_to_binary(folder, out_folder):
    """
    Convert a folder of mask in 0 255 format to binary format
    :param folder: folder to examine
    :param out_folder: folder 
    """

    try:
        os.mkdir(out_folder)
    except:
        pass
    if is_mask(folder, 255):
        list_mask_path = os.listdir(folder)
        for mask_path in list_mask_path:
            mask = cv2.imread(folder + mask_path, cv2.IMREAD_UNCHANGED)
            ret, thresh = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            cv2.imwrite(out_folder + mask_path, thresh)
    print("Conversion is done")


def binary_to_uint8(folder, out_folder):
    """
    Convert a folder of mask in 0 1 format to 0 255 format
    :param folder: folder to examine
    :param out_folder: folder 
    """

    try:
        os.mkdir(out_folder)
    except:
        pass
    if is_mask(folder, 1):
        list_mask_path = os.listdir(folder)
        for mask_path in list_mask_path:
            mask = cv2.imread(folder + mask_path, cv2.IMREAD_UNCHANGED)
            ret, thresh = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(out_folder + mask_path, thresh)
    print("Conversion is done")


def copy_paste(im,mask,fake_im,path=False):
    """
    Copy content from within the mask of image fake_im and 
    stitch it to the complementary content of image im
    
    :param im     : original image
    :param mask   : binary mask of where to alter the original image
    :param fake_im: fake image thatP
    :param path   : boolean, when true it indicates the first three
    parameters are path and not images.
    
    Images and mask should have the same size.
    """
    if path:
        im      = cv2.imread(im)
        mask    = cv2.imread(mask)
        fake_im = cv2.imread(fake_im)
        
    new_im = im*(1-mask) + fake_im*mask
    return(new_im)

