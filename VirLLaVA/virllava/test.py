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

torch.set_printoptions(threshold=10)  # 超过 10 个元素就省略中间部分
batch_size = 8
resume_path = "checkpoint/llava-v1"
accumulation = 4

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(json_file="llava_config.json")

tokenizer, model = build_model_tokenizer(model_args, data_args, training_args)

model.load_checkpoint(resume_path)
model.to(torch.bfloat16).cuda()
model.freeze_module()
model.trainable_params()

train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                            data_path="playground/VG/test.json",
                            data_args=data_args,
                            )

collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,     # 或你想要的大小
                              collate_fn=collator,
                              shuffle=False,
                              drop_last=True)



def test(model, dataloader):
    model.eval()
    
    losses = []
    loop = tqdm(iter(dataloader))

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
        
        with torch.no_grad():
            output = model(**input)
        
        loss = output["loss"]

        losses.append(loss)

        if (i+1) % 100 == 0:
            print(f"Iter {i}, Losses: {sum(losses)/len(losses)}")
        
        if i == 500:
            break

    return losses

losses = test(model, train_dataloader)
print("Loss:", sum(losses)/len(losses))
