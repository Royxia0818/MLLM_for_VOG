import random
QUESTION_TEMP_LIST=[
"Can you segment out {class_name} in this image?",
"Please segment out {class_name} in this image.",
"Can you locate {class_name} in this image?",
"Where is {class_name} in this image?"
]
# QUESTION_TEMP_APPENDIX="\n{} is in [{}, {}, {}, {}].\nPlease respond with a bounding box in the form of [xmin, ymin, xmax, ymax]."
# QUESTION_TEMP_LIST=[
# "Can you segment a place to generate {class_name} in this image?\nPlease respond with a bounding box in the form of [xmin, ymin, xmax, ymax].",
# "Please segment a space for the generation of {class_name} in this image.\nPlease respond with a bounding box in the form of [xmin, ymin, xmax, ymax].",
# "What is the best place to generate {class_name} in this image?\nPlease respond with a bounding box in the form of [xmin, ymin, xmax, ymax].",
# "Where can I put {class_name} in this image?\nPlease respond with a bounding box in the form of [xmin, ymin, xmax, ymax]."
# ]
# ANSWER_TEMP="[{}, {}, {}, {}]"
ANSWER_TEMP="It is here."

import json
import os
from PIL import Image


with open("VisualGenome/VrR-VG_info.json", "r") as json_file:
    data = json.load(json_file)


def load_list(file):
    result = []
    with open(file, "r") as json_file:
        info = json.load(json_file)
    result = list(info.keys())
    
    return result

image_list_1 = load_list("VisualGenome/Split/test_demo.json")
# print("train:", len(image_list_2), "test:", len(image_list_1))

def json_write(input, output_file):
    info_list = []
    for i, image in enumerate(input):
        width, height = Image.open(os.path.join("VisualGenome/PowerPaint", image)).size

        image_index, mask_index = os.path.splitext(image)[0].split("_")
        class_name = data[image_index + ".jpg"][mask_index]["sentence"]
        
        # bbox_sub = data[image_index + ".jpg"][mask_index]["subject"]["bbox"]
        # name_sub = data[image_index + ".jpg"][mask_index]["subject"]["name"]
        # bbox_ob = data[image_index+".jpg"][mask_index]["object"]["bbox"]
        # name_ob = data[image_index + ".jpg"][mask_index]["object"]["name"]

        question_temp = random.choice(QUESTION_TEMP_LIST)

        id = str(i).zfill(12)
        human = "<image>\n" + question_temp.format(class_name=class_name)
        
        gpt = ANSWER_TEMP
        
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
        
    with open(output_file, "w") as json_file:
        json.dump(info_list, json_file, indent=4)
            
json_write(image_list_1, "playground/VG/demo.json")