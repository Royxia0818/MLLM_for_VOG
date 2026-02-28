import torch
from args import *
from transformers import HfArgumentParser
from llava import conversation as conversation_lib
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from llava.model import *
from build import *
from dataset import *
import cv2
from torchvision.ops import box_iou
import json
import numpy as np

torch.set_printoptions(threshold=10)  # 超过 10 个元素就省略中间部分
batch_size = 4
resume_path = "checkpoint/llava_vog/llava-v3_3"

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(json_file="llava_config.json")

tokenizer, model = build_model_tokenizerv3(model_args, data_args, training_args)
model.load_checkpoint(resume_path)
model.to(torch.bfloat16).cuda()
model.freeze_module()
model.describe()

train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                            data_path="playground/PixelHacker/train.json",
                            data_args=data_args,
                            )

collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,     # 或你想要的大小
                              collate_fn=collator,
                              shuffle=False,
                              drop_last=False)


def demo(model, train_dataloader, 
         save: bool = True, output_path: str = None, output_mask_path: str = None):
    with open("VisualGenome/Split/VrR-VG_info.json", "r") as json_file:
        data = json.load(json_file)

    ref_iou_list = []
    vir_iou_list = []
    loop = iter(train_dataloader)
    for i, batch in enumerate(loop):
        input = {
            "input_ids": batch["input_ids"].to(torch.int64).to("cuda"),
            "labels": batch["labels"].to(torch.int64).to("cuda"),
            "attention_mask": batch["attention_mask"].to(torch.bool).to("cuda"),
            "images": batch["images"].to(torch.bfloat16).to("cuda"),
            "gt_ref_mask": batch["ref_masks"].to(torch.bfloat16).to("cuda"),
            "gt_vir_mask": batch["vir_masks"].to(torch.bfloat16).to("cuda"),
            "gt_ref_box": batch["ref_boxes"].to(torch.bfloat16).to("cuda"),
            "gt_vir_box": batch["vir_boxes"].to(torch.bfloat16).to("cuda"),
        }
        
        image_names = batch["img_names"]
        
        gt_ref_boxes = batch["ref_boxes"].cuda()        
        gt_vir_boxes = batch["vir_boxes"].cuda()
        
        
        with torch.no_grad():
            output = model(**input)
        
        ref_boxes = output["ref_box"]
        vir_boxes = output["vir_box"]
        
        ref_ious = box_iou(ref_boxes, gt_ref_boxes)
        ref_ious = ref_ious.diag().tolist()
        ref_iou_list += ref_ious

        vir_ious = box_iou(vir_boxes, gt_vir_boxes)
        vir_ious = vir_ious.diag().tolist()
        vir_iou_list += vir_ious
        
        if save:
            if not (output_path and output_mask_path):
                raise ValueError("output_paths are not input")
            for j in range(vir_boxes.shape[0]):
                image_name = image_names[j]

                id1, id2 = os.path.splitext(image_name)[0].split("_")
                gt_vir_box = gt_vir_boxes[j]
                
                vir_box = vir_boxes[j]
                            
                # sentence = data[id1+".jpg"][id2]["sentence"]

                image_path = os.path.join("VisualGenome/PixelHacker", image_name)
                image_np = cv2.imread(image_path)
                height, width, channel = image_np.shape       
                if height / width != 0.75:
                    continue
                
                def box_int(box: torch.Tensor,
                            width: int, height: int) -> list:
                    box = box.tolist()
                    box = [box[0]*width, box[1]*height, box[2]*width, box[3]*height]
                    box = [int(p) for p in box]
                    return box

                def draw_box(input_img: np.ndarray,
                            box: list,
                            color: tuple = (255, 0, 0)):
                    cv2.rectangle(input_img, box[0:2], box[2:4], color, thickness=2)
                
                
                vir_box = box_int(vir_box, width, height)
                gt_vir_box = box_int(gt_vir_box, width, height)
                vir_img = image_np.copy()
                draw_box(vir_img, vir_box, color=(147, 212, 108))
                draw_box(vir_img, gt_vir_box, color=(243, 203, 0))
                
                img_save_path = Path(output_path)/image_name
                cv2.imwrite(img_save_path, vir_img)                

                mask_save_path = Path(output_mask_path)/image_name
                mask_img = np.zeros((image_np.shape[0],image_np.shape[1]), dtype=np.uint8)
                mask_img[vir_box[1]:vir_box[3], vir_box[0]:vir_box[2]] = 255
                cv2.imwrite(mask_save_path, mask_img)
        
        if i == 250:
            break
                
    return vir_iou_list, ref_iou_list
            

def refresh_folder(path: str):
    import shutil
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    print(f"Old Files removed in {path}")


output_path = "demo_output"
output_mask_path = "mask"
refresh_folder(output_path)
refresh_folder(output_mask_path)

from datetime import datetime
start = datetime.now()
vir_iou_list, ref_iou_list = demo(model, train_dataloader, 
                                  True, output_path, output_mask_path)
duration = datetime.now() - start
print(str(duration/250))

def plot_iou_distribution(iou_list, title, output_file="plt.png"):
    iou_array = np.array(iou_list)
    bins = np.arange(0, 1.1, 0.1)
    counts, _ = np.histogram(iou_array, bins=bins)
    frequencies = counts / len(iou_array)

    plt.figure(figsize=(8, 5))
    plt.bar([f"{round(bins[i+1],1)}" for i in range(len(bins)-1)], frequencies, width=0.8, color='skyblue', edgecolor='black')
    plt.xlabel("IoU")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.clf()
    
    iou_stat = [sum(frequencies[-i:]) for i in range(1, 6)]
    return iou_stat
    
vir_iou_stat = plot_iou_distribution(vir_iou_list, "vir iou", "vir.png")
ref_iou_stat = plot_iou_distribution(ref_iou_list, "ref iou", "ref.png")
print(vir_iou_stat)
print(ref_iou_stat)