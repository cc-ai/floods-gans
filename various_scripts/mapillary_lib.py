# Sasha code
from PIL import Image, ImageChops
from matplotlib.pyplot import imshow
import numpy as np
import glob
import tqdm as tq
import os 

def percent_building(im_path,label=17):
    """
    From an image of label, return the percent of label building
    in mapillary default value is 17 for the buildings
    
    :param im_path: path to the label image
    :param label  : label we consider as building
    """
    im = Image.open(im_path)
    percent_cover = 0
    for tup in im.getcolors():
        if tup[1] == label:
            percent_cover = (tup[0]/im.size[0])/im.size[1]
    return(percent_cover)


def im_w_building(path_dir,label=17):
    """
    Return a list of the path to the images and percentage of building
    in it
    
    :param path_dir: path to the directory to analyse
    :param label   : label we consider as building
    
    """
    list_img = os.listdir(path_dir)
    list_pb  = []
    for im_index in tq.tqdm(range(len(list_img))):
        im_path = list_img[im_index]
        list_pb.append(percent_building(path_dir+im_path,label))
    return(list_img,list_pb)

def threshold_building(list_img,list_pb,threshold=0.15):
    """
    Return the list of images where the % of building is over the 
    threshold
    :param threshold: varies between 0 and 1 is the minimum % of label
    building we want in an image
    :param list_img : output of im_w_building list path of images
    :param list_pb  : output of im_w_building list of % of building
    in the image
    """
    over_thresh = np.where(np.array(list_pb)>threshold_percent)
    return(np.array(list_img)[over_thresh])
    
# Example 
# path_dir = '/network/tmp1/ccai/data/mapillary/training/labels/'
# list_img,list_pb = im_w_building(path_dir)
# over_thresh = threshold_building(list_img,list_pb)
