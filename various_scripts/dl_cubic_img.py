#!/usr/bin/env python
import sys
from PIL import Image
from math import pi, sin, cos, tan, atan2, hypot, floor
from numpy import clip
import imageio
import os
import requests
import numpy as np
from urllib.request import urlopen
import cv2

current_path = os.getcwd()
try: os.mkdir(current_path + "/Downloaded")
except:pass
try: os.mkdir(current_path + "/output/")
except:pass

def get_json_from_lat(lat,lng):
    """
    Download in json format information about panorama taken at a position in
    google streetview
    
    :param lat: latitute
    :param lng: longitude
    """
    
    URL = "https://maps.google.com/cbk?output=xml&ll="\
        + str(lat) + "," + str(lng) + "8&dm=1"
    print(URL)
    
    response = requests.get(URL)
    depth_map_xml=response.content
    str_depth = str(depth_map_xml)
    index_ = str_depth.find('<depth_map>')
    
    if index_> 10:
        index_end = str_depth.find('</depth_map>')
        s = str_depth[index_+len('<depth_map>'):index_end]
        index_pano_id = str_depth.find('pano_id=')+1
        index_end_pano = str_depth.find('imagery_type=')-2
        pano_id = str_depth[index_pano_id+len('pano_id='):index_end_pano]
    else:
        print('there is no data for this location')
        s=0
    return(s,pano_id)



def downloadImage(url,image_name):
    """
    Download and save image given its url in ./Downloaded/{image_name}.jpg
    :param url:         url of the image
    :param image_name:  filename to save the image
    """  

    try:
        print("Downloading %s" % (url))
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imageio.imwrite(current_path + "/Downloaded/" +image_name+'.jpg', image)
        return(image)
    except Exception as error:
        print(error)
        
def dl_pano(pano_id,zoom_level,case):
  
    """
    Low level function downloading a panoramic image depending the case of
    panoramic images 512x256 or 416x208
    
    :param pano_id:      panorama identifier
    :param zoom_level:   size of the zoom (0-5)
    :param case:         boolean encoding the storage format 512 or 416
    
    
    """
    TILE_HEIGHT = 512
    TILE_WIDTH  = 512
    
    if case == 1:
        ZOOM0_WIDTH = 416
        ZOOM0_HEIGHT = 208
    else:
        ZOOM0_WIDTH = 512
        ZOOM0_HEIGHT = 256


    #Dimensions of the image at a particular zoom_level

    width = ZOOM0_WIDTH * 2**zoom_level
    height = ZOOM0_HEIGHT * 2**zoom_level

    #Create Texture
    pano_image= np.zeros(width*height*3)

    #Download each individual tile and compose them into the big texture
    for tile_y in range(int(np.ceil(height/TILE_HEIGHT))):
        for tile_x in range(int(np.ceil(width /TILE_WIDTH))):
          
            url   = "http://cbk0.google.com/cbk?output=tile&panoid="+ \
                     str(pano_id)+'&zoom='+str(zoom_level)+'&x='+str(tile_x)+ \
                    '&y='+str(tile_y)
            
            tile  = downloadImage(url,str(tile_y)+str(tile_x))

            tile_width = tile.shape[0]
            tile_height = tile.shape[1]
            tile=tile.reshape((tile_width*tile_height*3))
            if (tile_width != TILE_WIDTH |tile_height != TILE_HEIGHT):
                print("Downloaded tile had unexpected dimensions")

            for y in range(tile_height):
                for x in range(tile_width):
                    global_x = tile_x * tile_width + x
                    global_y = tile_y * tile_height + y

                    if (global_x < width and global_y < height):
                        pano_image[(global_y * width + global_x)*3 + 0] = \
                        tile[(y * tile_width + x)*3 + 0]
                        pano_image[(global_y * width + global_x)*3 + 1] = \
                        tile[(y * tile_width + x)*3 + 1]
                        pano_image[(global_y * width + global_x)*3 + 2] = \
                        tile[(y * tile_width + x)*3 + 2]
        pano_image=pano_image.astype(int)
        
    return(pano_image.reshape((height,width,3)))


def dl_panorama(pano_id,zoom_level):
    """
    Download the panorama in both identified case of panoramic images 512x256 
    and 416x208
    :param pano_id:    panorama identifier
    :param zoom_level: size of the zoom (0-5)
    """

    img = dl_pano(pano_id,0,0)
    if((img[220:250,0,:] == 0).all()):
      
        # Regular case where there is overlapping
        img_to_return = dl_pano(pano_id,zoom_level,1)
        img_to_return = img_to_return.astype(np.uint8)
        cv2.resize(img_to_return, dsize=(512*2**zoom_level,256*2**zoom_level),\
                   interpolation=cv2.INTER_CUBIC)
    else:
      
        img_to_return=dl_pano(pano_id,zoom_level,0)
        
    return(img_to_return)



def outImgToXYZ(i, j, faceIdx, faceSize):
    """
    Get x,y,z coordinates from output image pixels coordinates
    :param i:        pixel coordinate
    :param j:        pixel coordinate
    :param faceIdx:  face number
    :param faceSize: edge length
    """
    a = 2.0 * float(i) / faceSize
    b = 2.0 * float(j) / faceSize

    if faceIdx == 0: # back
        (x,y,z) = (-1.0, 1.0 - a, 1.0 - b)
    elif faceIdx == 1: # left
        (x,y,z) = (a - 1.0, -1.0, 1.0 - b)
    elif faceIdx == 2: # front
        (x,y,z) = (1.0, a - 1.0, 1.0 - b)
    elif faceIdx == 3: # right
        (x,y,z) = (1.0 - a, 1.0, 1.0 - b)
    elif faceIdx == 4: # top
        (x,y,z) = (b - 1.0, a - 1.0, 1.0)
    elif faceIdx == 5: # bottom
        (x,y,z) = (1.0 - b, a - 1.0, -1.0)

    return (x, y, z)


def convertFace(imgIn, imgOut, faceIdx):
    """
    Convert using an inverse transformation

    :param imgIn:    input image
    :param imgOut:   output image
    :param faceIdx:  face number
    """
    inSize = imgIn.size
    outSize = imgOut.size
    inPix = imgIn.load()
    outPix = imgOut.load()
    faceSize = outSize[0]

    for xOut in range(faceSize):
        for yOut in range(faceSize):
            (x,y,z) = outImgToXYZ(xOut, yOut, faceIdx, faceSize)
            theta = atan2(y,x) # range -pi to pi
            r = hypot(x,y)
            phi = atan2(z,r) # range -pi/2 to pi/2

            # source img coords
            uf = 0.5 * inSize[0] * (theta + pi) / pi
            vf = 0.5 * inSize[0] * (pi/2 - phi) / pi

            # Use bilinear interpolation between the four surrounding pixels
            ui = floor(uf)  # coord of pixel to bottom left
            vi = floor(vf)
            u2 = ui+1       # coords of pixel to top right
            v2 = vi+1
            mu = uf-ui      # fraction of way across pixel
            nu = vf-vi

            # Pixel values of four corners
            A = inPix[ui % inSize[0], int(clip(vi, 0, inSize[1]-1))]
            B = inPix[u2 % inSize[0], int(clip(vi, 0, inSize[1]-1))]
            C = inPix[ui % inSize[0], int(clip(v2, 0, inSize[1]-1))]
            D = inPix[u2 % inSize[0], int(clip(v2, 0, inSize[1]-1))]

            # interpolate
            (r,g,b) = (
              A[0]*(1-mu)*(1-nu) + B[0]*(mu)*(1-nu) + C[0]*(1-mu)*nu+D[0]*mu*nu,
              A[1]*(1-mu)*(1-nu) + B[1]*(mu)*(1-nu) + C[1]*(1-mu)*nu+D[1]*mu*nu,
              A[2]*(1-mu)*(1-nu) + B[2]*(mu)*(1-nu) + C[2]*(1-mu)*nu+D[2]*mu*nu )

            outPix[xOut, yOut] = (int(round(r)), int(round(g)), int(round(b)))


def dl_cubic_img(lat,lng,dir_output):
    """
    Download 4 lateral faces of the cube corresponding to the panoramic
    images taken at lat,lng.
    
    :param lat:        latitute
    :param lng:        longitude
    :param dir_output: directory to save images
    """
    s,pano_id = get_json_from_lat(lat,lng)
    img = dl_panorama(pano_id,2)
    imageio.imsave(dir_output+'seg1.jpg',img)
    imgIn = Image.open(dir_output+'seg1.jpg')
    inSize = imgIn.size
    faceSize = inSize[0] // 4

    FACE_NAMES = {
        0: 'back',
        1: 'left',
        2: 'front',
        3: 'right',
    }

    for face in range(4):
        imgOut = Image.new("RGB", (faceSize, faceSize), "black")
        convertFace(imgIn, imgOut, face)
        imgOut.save(dir_output+str(pano_id) + FACE_NAMES[face] + ".png")
    os.remove(dir_output+'seg1.jpg')

# Example of cmd might be 
# dl_cubic_img(45.5119339,-73.5682286,'./output/')