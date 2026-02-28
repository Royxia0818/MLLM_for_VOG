import torch
from args import *
from transformers import HfArgumentParser
from llava import conversation as conversation_lib
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from llava.model import *
from build import *
from dataset import *
import logging

torch.set_printoptions(threshold=10)  # 超过 10 个元素就省略中间部分

batch_size = 1
num_epochs = 15
save_path = "checkpoint/llava-v3_3"
resume_path = "checkpoint/llava-v3_2"
resume_path=None
accumulation = 4

Path(save_path).mkdir(exist_ok=True)
logging.basicConfig(format="[%(asctime)s] %(message)s",
                    datefmt="%I:%M:%S",
                    level=logging.INFO,
                    filename=Path(save_path)/"save.log",
                    filemode="w")

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(json_file="llava_config.json")

tokenizer, model = build_model_tokenizerv3(model_args, data_args, training_args)

model.load_checkpoint(resume_path)
model.to(torch.bfloat16).cuda()
model.describe()


train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                            data_path=data_args.data_path,
                            data_args=data_args,
                            )

collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,     # 或你想要的大小
                              collate_fn=collator,
                              shuffle=True,
                              drop_last=True)


optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)


def train_one_epoch(model, dataloader, optimizer):
    model.train()
    
    losses = []
    loop = tqdm(iter(dataloader))
    accumulated_loss = 0.0
    accum_count = 0

    for i, batch in enumerate(loop):
        input = {
            "input_ids": batch["input_ids"].to(torch.int64).to("cuda"),
            "labels": batch["labels"].to(torch.int64).to("cuda"),
            "attention_mask": batch["attention_mask"].to(torch.bool).to("cuda"),
            "images": batch["images"].to(torch.bfloat16).to("cuda"),
            # "gt_ref_mask": batch["ref_masks"].to(torch.bfloat16).to("cuda"),
            # "gt_vir_mask": batch["vir_masks"].to(torch.bfloat16).to("cuda"),
            "gt_ref_box": batch["ref_boxes"].to(torch.bfloat16).to("cuda"),
            "gt_vir_box": batch["vir_boxes"].to(torch.bfloat16).to("cuda"),
        }
        
        output = model(**input)
        
        ref_loss = output["ref_loss"]
        vir_loss = output["vir_loss"]
        loss = 0.8 * ref_loss + 1.0 * vir_loss
        loss.backward()

        accumulated_loss += loss.item()
        accum_count += 1

        if (i + 1) % accumulation == 0 or (i + 1) == len(loop):
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = accumulated_loss / accum_count
            losses.append(avg_loss)  # 只在更新时记录一次平均 loss
            accumulated_loss = 0.0
            accum_count = 0
        
        if (i+1) % 10000 == 0:
            # print(f"Iter {i}, Losses: {sum(losses)/len(losses)}")
            logging.info(f"Iter {i}, Losses: {sum(losses)/len(losses)}")

    return losses

train_losses = []
for e in range(num_epochs):
    train_loss = train_one_epoch(model, train_dataloader, optimizer)
    
    print(sum(train_loss)/len(train_loss))
    logging.info(sum(train_loss)/len(train_loss))
    train_losses.append(sum(train_loss)/len(train_loss))
    
    model.save_checkpoint(save_path)
    print("Model saved to", save_path)
    logging.info(f"[Epoch: {e}] Model saved to {save_path}")

print(train_losses)
logging.info(f"train losses: {train_losses}")