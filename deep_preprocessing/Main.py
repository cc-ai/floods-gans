import os
import deep_preprocessing.TF as TF2
import tqdm as tq 

# Directory input and output path 
input_dir = './input_folder/'
output_dir = './output_folder/'

# Extension of the image to transform
file_extension ='.jpg'

# List all the file with the correct extension
filelist=os.listdir(input_dir)
for file in filelist[:]:
    if not(file.endswith(file_extension)):
        filelist.remove(file)

print("There is " +str(len(filelist)) + " element in the folder")

# Inference 1 unit at a time
for file in tq.tqdm(filelist):
  file_in_name = TF2.getInputPhoto(input_dir+file)
  TF2.processImg(file_in_name,output_dir+file)