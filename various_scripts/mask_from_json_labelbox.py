import tqdm as tq
import json
import requests
import os


fname= 'export-2019-06-12T18_33_17.792Z.json'
x = json.load(open(fname))


# Download the masks 
for k in tq.tqdm(range(len(x))):
    p = x[k]
    if p['Label']!='Skip':
        masks=p['Masks']
        name =p['External ID']
        filename, file_extension = os.path.splitext(name)
        for m,l in masks.items():
            r = requests.get(l, allow_redirects=False)
            open('./masks/'+filename+'_0.png', 'wb').write(r.content)

# Download the imgs
for k in tq.tqdm(range(len(x))):
    p = x[k]
    if p['Label']!='Skip':
        labeled_data=p['Labeled Data']
        name =p['External ID']
        r = requests.get(labeled_data, allow_redirects=False)
        open('./imgs/'+name, 'wb').write(r.content)


# Convert the image in png format
list_img = os.listdir('./imgs/')
for img_path in tq.tqdm(range(len(list_img))):
    filename, file_extension = os.path.splitext(list_img[img_path])
    img_  = imageio.imread('./imgs/'+list_img[img_path])
    imageio.imsave('./imgs/'+filename+'.png',img_)
