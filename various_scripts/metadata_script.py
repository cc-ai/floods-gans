import json
import os
import sys
import os.path
log_dir = sys.argv[1]
img_dir = sys.argv[2]
for filename in os.listdir(log_dir):
	if filename.endswith(".json"):
		data1 = []
		with open(os.path.join(log_dir,filename)) as json_file:  
			data = json.load(json_file)
			core_file = os.path.splitext(filename)[0]
			for item in range(len(data)):
				exists = os.path.isfile(os.path.join(img_dir,os.path.join(core_file,data[item]['image_filename'])))
				if exists:
					data1.append(data[item])
		with open(os.path.join(log_dir,"mod_"+filename), 'w') as outfile:  
			json.dump(data1, outfile)
