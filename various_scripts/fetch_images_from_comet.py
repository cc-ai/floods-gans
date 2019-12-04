import os
import imageio
import requests
from PIL import Image, ImageDraw, ImageFont
import tqdm as tq
import errno

# Define the Font
font = ImageFont.truetype('Roboto-Bold.ttf', size=100)

# Experiment Key
EXPERIMENT_KEY=''

# COMET api key
COMET_REST_API_KEY=''

# REST API call
r = requests.get('https://www.comet.ml/api/rest/v1/experiment/images?experimentKey='+EXPERIMENT_KEY,\
     headers={'Authorization': COMET_REST_API_KEY})

# Get the JSON response
json_content = r.json()
gif_images=[]

# Counter=0
# Filter the response so that we extract path and step to download the images
for j in tq.tqdm(range(len(json_content['images']))):
    im_info = json_content['images'][j]
    if im_info['figName']=='gen_a2b_train_current.jpg':
        step = im_info['step']
        image_url = im_info['imagePath']
        # Download the image
        img_data = requests.get(image_url).content
        with open('image_name.jpg', 'wb') as handler:
            handler.write(img_data)
        # create Image object with the input image
        image = Image.open('image_name.jpg')
        # initialise the drawing context with
        # the image object as background
        draw = ImageDraw.Draw(image)
        # starting position of the message
        (x, y) = (90, 600)
        message = str(step)
        color = 'rgb(249, 20, 20)' # kind of red color
        # draw the message on the background
        draw.text((x, y), message, font=font, fill=color)
        # save edited image
        image.save('image_name.jpg')
        gif_images.append(imageio.imread('image_name.jpg'))        
try:
    os.mkdir('./gif/')
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

# Write the GIF
imageio.mimwrite('./gif/movie_spade.gif', gif_images, fps ='1')






