import random
QUESTION_TEMP_LIST=[
"Can you segment a place to generate {class_name} in this image?\nPlease respond with a bounding box in the form of [xmin, ymin, xmax, ymax].",
"Please segment a space for the generation of {class_name} in this image.\nPlease respond with a bounding box in the form of [xmin, ymin, xmax, ymax].",
"What is the best place to generate {class_name} in this image?\nPlease respond with a bounding box in the form of [xmin, ymin, xmax, ymax].",
"Where can I put {class_name} in this image?\nPlease respond with a bounding box in the form of [xmin, ymin, xmax, ymax]."
]
ANSWER_TEMP="[{}, {}, {}, {}]"

import json
import os
from PIL import Image


with open("VisualGenome/VrR-VG_info.json", "r") as json_file:
    data = json.load(json_file)

info_list = []
with open("VisualGenome/split.json", "r") as json_file:
    splits = json.load(json_file)
image_list = []
for i, split in splits.items():
    if split == "Pass":
        image_list.append(i)

print(len(image_list))
random.shuffle(image_list)
for i, image in enumerate(image_list):
    width, height = Image.open(os.path.join("VisualGenome/PowerPaint", image)).size

    image_index, mask_index = os.path.splitext(image)[0].split("_")
    class_name = data[image_index + ".jpg"][mask_index]["sentence"]
    bbox = data[image_index + ".jpg"][mask_index]["subject"]["bbox"]
    question_temp = random.choice(QUESTION_TEMP_LIST)

    id = str(i).zfill(12)
    human = "<image>\n" + question_temp.format(class_name=class_name)
    gpt = ANSWER_TEMP.format(round(bbox[0]/width, 2),
                             round(bbox[1]/height, 2),
                             round(bbox[2]/width, 2),
                             round(bbox[3]/height, 2),
                             )
    
    info = {"id": id,
            "image": image,
            "conversations": [
                {
                    "from": "human",
                    "value": human
                },
                {
                    "from": "gpt",
                    "value": gpt
                }
            ]}
    info_list.append(info)
    
    if i % 2000 == 0:
        print("Loading info:", i)
        
with open("playground/gqa_inpaint.json", "w") as json_file:
            json.dump(info_list, json_file, indent=4)
    