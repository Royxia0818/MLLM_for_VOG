import random
QUESTION_TEMP_LIST=[
"Can you segment a place to generate {class_name} in this image?",
"Please segment a space for the generation of {class_name} in this image.",
"What is the best place to generate {class_name} in this image?",
"Where can I put {class_name} in this image?"
]
QUESTION_TEMP_APPENDIX="\nPlease respond with a bounding box in the form of [xmin, ymin, xmax, ymax]."

ANSWER_TEMP="{object_name} is here, so {subject_name} should be placed here."

import json
import os
from PIL import Image


with open("Vir/Select.json", "r") as json_file:
    data = json.load(json_file)

image_list = os.listdir("Vir/PowerPaint")
print(len(image_list))
random.shuffle(image_list)

test_list = image_list[:50]
train_list = image_list[50:]
print("test", len(test_list), "train", len(train_list))


def json_write(input, output_file):
    info_list = []
    for i, image in enumerate(input):
        width, height = Image.open(os.path.join("Vir/PowerPaint", image)).size

        class_name = data[image]["sentence"]
        
        bbox_sub = data[image]["subject"]["bbox"]
        name_sub = data[image]["subject"]["name"]
        bbox_ob = data[image]["object"]["bbox"]
        name_ob = data[image]["object"]["name"]

        question_temp = random.choice(QUESTION_TEMP_LIST)

        id = str(i).zfill(12)
        human = "<image>\n" + question_temp.format(class_name=class_name) + \
                QUESTION_TEMP_APPENDIX
        
        gpt = ANSWER_TEMP.format(subject_name=name_sub,
                                 object_name=name_ob
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
        
    with open(output_file, "w") as json_file:
        json.dump(info_list, json_file, indent=4)
            
json_write(test_list, "playground/Vir/test.json")
json_write(train_list, "playground/Vir/train.json")