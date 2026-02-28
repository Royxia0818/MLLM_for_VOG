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

image_list = os.listdir("VisualGenome/PixelHacker")
random.shuffle(image_list)

def split_list(input, test_size=1000):
    part1 = input[:test_size]
    part2 = input[test_size:]
    return part1, part2

image_list_1, image_list_2 = split_list(image_list)
print(len(image_list_1), len(image_list_2))

def json_write(input, output_file):
    info_list = []
    for i, image in enumerate(input):
        width, height = Image.open(os.path.join("VisualGenome/PixelHacker", image)).size

        image_index, mask_index = os.path.splitext(image)[0].split("_")
        class_name = data[image_index + ".jpg"][mask_index]["sentence"]
        
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
            

json_write(image_list_1, "playground/PixelHacker/test.json")
json_write(image_list_2, "playground/PixelHacker/train.json")