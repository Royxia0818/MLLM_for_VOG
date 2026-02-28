import torch
import torch.nn as nn
import torchvision.models as models
import os
from pathlib import Path
from safetensors.torch import load_file


from typing import Dict, Optional, Sequence, List

import torch
import torch.nn.functional as F

import transformers
import tokenizers

from llava.model import *

from peft import PeftModel
from peft import LoraConfig, get_peft_model

from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
import torch.nn.init as init


class mseloss(nn.Module):
    def __init__(self):
        super(mseloss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_boxes, target_boxes):
        return self.criterion(pred_boxes, target_boxes)
    
class BoxDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.GELU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
        
        self.mlp.apply(init_weights)

    def forward(self, x):
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x
    
    
class MetaLLaVA(nn.Module):
    def __init__(self,
                 model_args, data_args, training_args,
                 token_len=32):
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        self.llava_model = LlavaLlamaForCausalLM.from_pretrained(
                'liuhaotian/llava-v1.5-7b',
                cache_dir=None,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
        self.llava_model.config.output_hidden_states = True
        
        self.token_len = token_len
        self.learnable_embedding = nn.Embedding(self.token_len, 4096)
        self.register_buffer("learnable_ids", torch.arange(self.token_len))
        
        self.box_decoder = BoxDecoder()
        self.loss_compute = mseloss()
        
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        gt_ref_box: Optional[torch.FloatTensor] = None,
        gt_vir_box: Optional[torch.FloatTensor] = None,
        **kwargs
        ):
        
        batch_size = input_ids.shape[0]
        learnable_ids = self.learnable_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
        learnable_tokens = self.learnable_embedding(learnable_ids)

        input = {"input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images,
            "learnable_tokens": learnable_tokens,
            "token_len": self.token_len}
        
        output, learnable_token_mask = self.llava_model(**input)
        last_hidden_states = output["hidden_states"][-1]
        
        output_learnable_tokens = []
        for i, last_hidden_state in enumerate(last_hidden_states):
            output_learnable_token = last_hidden_state[learnable_token_mask[i]]
            output_learnable_tokens.append(output_learnable_token)
        output_learnable_tokens = torch.stack(output_learnable_tokens, dim=0)
        
        vir_box = self.box_decoder(output_learnable_tokens)
        box_loss = self.loss_compute(vir_box, gt_vir_box)
        
        output = {"last_hidden_states": last_hidden_states,
                  "vir_box": vir_box,
                  "loss": box_loss}
        return output
        
    
    def save_checkpoint(self, save_path): 
        os.makedirs(save_path, exist_ok=True)
        self.llava_model.save_pretrained(save_path)
        torch.save(self.llava_model.get_model().mm_projector.state_dict(), os.path.join(save_path, "mm_projector.bin"))
        
        torch.save(self.box_decoder.state_dict(), os.path.join(save_path, "box_decoder.pth"))
        torch.save(self.learnable_embedding.state_dict(), os.path.join(save_path, "learnable_embedding.pth"))

        
    def load_checkpoint(self, resume=None):
        def find_all_linear_names(model):
            cls = torch.nn.Linear
            lora_module_names = set()
            multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
            for name, module in model.named_modules():
                if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                    continue
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if 'lm_head' in lora_module_names: # needed for 16-bit
                lora_module_names.remove('lm_head')
            return list(lora_module_names)
        
        lora_config = LoraConfig(
            r=self.training_args.lora_r,
            lora_alpha=self.training_args.lora_alpha,
            target_modules=find_all_linear_names(self.llava_model),
            lora_dropout=self.training_args.lora_dropout,
            bias=self.training_args.lora_bias,
            task_type="CAUSAL_LM",
        )        
        
        if resume:
            self.llava_model = PeftModel.from_pretrained(self.llava_model, resume)
            for name, param in self.llava_model.named_parameters():
                if "lora" in name.lower() or "adapter" in name.lower():
                    param.requires_grad = True
            # 加载 mm_projector 权重
            mm_proj_path = os.path.join(resume, "mm_projector.bin")
            self.llava_model.get_model().mm_projector.load_state_dict(torch.load(mm_proj_path, map_location='cpu'))
            
            box_decoder_path = os.path.join(resume, "box_decoder.pth")
            box_decoder_state_dict = torch.load(box_decoder_path, map_location="cpu")
            self.box_decoder.load_state_dict(box_decoder_state_dict)
            
            learnable_embedding_path = os.path.join(resume, "learnable_embedding.pth")
            learnable_embedding_path = torch.load(learnable_embedding_path, map_location="cpu")
            self.learnable_embedding.load_state_dict(learnable_embedding_path)
            
            print(f"Model resumed from {resume}")
            
        else:            
            self.llava_model = get_peft_model(self.llava_model, lora_config)
            self.llava_model.to(torch.bfloat16)
            print("Resuming from None")
                
                
    def freeze_module(self):
        for param in self.llava_model.parameters():
            param.requires_grad = False

        
    def trainable_params(self):
        def trainable_info(module):
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(type(module).__name__)
            print(f"Trainable: {trainable_params:,} Total parameters: {total_params:,}")
        
        trainable_info(self)
        trainable_info(self.box_decoder)



class MetaLLaVAv3(nn.Module):
    def __init__(self,
                 model_args, data_args, training_args,
                 token_len=32):
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        
        self.llava_model = LlavaLlamaForCausalLM.from_pretrained(
                'liuhaotian/llava-v1.5-7b',
                cache_dir=None,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16
            )
        self.llava_model.config.output_hidden_states = True
        
        # import inspect
        # print(inspect.getfile(self.llava_model.__class__))
        
        self.token_len = token_len
        self.ref_learnable_embedding = nn.Embedding(self.token_len, 4096)
        self.vir_learnable_embedding = nn.Embedding(self.token_len, 4096)
        self.register_buffer("learnable_ids", torch.arange(self.token_len))
        
        self.ref_box_decoder = BoxDecoder()
        self.vir_box_decoder = BoxDecoder()
        self.loss_compute = mseloss()
        
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        gt_ref_box: Optional[torch.FloatTensor] = None,
        gt_vir_box: Optional[torch.FloatTensor] = None,
        **kwargs
        ):
        
        batch_size = input_ids.shape[0]
        learnable_ids = self.learnable_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
        ref_learnable_tokens = self.ref_learnable_embedding(learnable_ids)
        vir_learnable_tokens = self.vir_learnable_embedding(learnable_ids)
        learnable_tokens = torch.cat([ref_learnable_tokens, vir_learnable_tokens], dim=1)

        input = {"input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "images": images,
            "learnable_tokens": learnable_tokens,
            "token_len": self.token_len * 2}
        
        output, learnable_token_mask = self.llava_model(**input)
        last_hidden_states = output["hidden_states"][-1]

        ref_output_learnable_tokens = []
        vir_output_learnable_tokens = []
        for i, last_hidden_state in enumerate(last_hidden_states):
            output_learnable_token = last_hidden_state[learnable_token_mask[i]]

            ref_output_learnable_token = output_learnable_token[:32, :]
            vir_output_learnable_token = output_learnable_token[32:, :]

            ref_output_learnable_tokens.append(ref_output_learnable_token)
            vir_output_learnable_tokens.append(vir_output_learnable_token)

        ref_output_learnable_tokens = torch.stack(ref_output_learnable_tokens, dim=0)
        vir_output_learnable_tokens = torch.stack(vir_output_learnable_tokens, dim=0)
                
        ref_box = self.ref_box_decoder(ref_output_learnable_tokens)        
        ref_box_loss = self.loss_compute(ref_box, gt_ref_box)
        
        vir_box = self.vir_box_decoder(vir_output_learnable_tokens)        
        vir_box_loss = self.loss_compute(vir_box, gt_vir_box)
        
        output = {"last_hidden_states": last_hidden_states,
                  "ref_box": ref_box,
                  "vir_box": vir_box,
                  "ref_loss": ref_box_loss,
                  "vir_loss": vir_box_loss}
        return output
        
    
    def save_checkpoint(self, save_path): 
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)

        self.llava_model.save_pretrained(save_path)
        torch.save(self.llava_model.get_model().mm_projector.state_dict(), save_path/"mm_projector.bin")
        
        torch.save(self.ref_learnable_embedding.state_dict(), save_path/"ref_le.pth")
        torch.save(self.ref_box_decoder.state_dict(), save_path/"ref_box_decoder.pth")

        torch.save(self.vir_learnable_embedding.state_dict(), save_path/"vir_le.pth")
        torch.save(self.vir_box_decoder.state_dict(), save_path/"vir_box_decoder.pth")

        
    def load_checkpoint(self, resume=None):
        def find_all_linear_names(model):
            cls = torch.nn.Linear
            lora_module_names = set()
            multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
            for name, module in model.named_modules():
                if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                    continue
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])

            if 'lm_head' in lora_module_names: # needed for 16-bit
                lora_module_names.remove('lm_head')
            return list(lora_module_names)
        
        lora_config = LoraConfig(
            r=self.training_args.lora_r,
            lora_alpha=self.training_args.lora_alpha,
            target_modules=find_all_linear_names(self.llava_model),
            lora_dropout=self.training_args.lora_dropout,
            bias=self.training_args.lora_bias,
            task_type="CAUSAL_LM",
        )        
        
        if resume:
            resume_path = Path(resume)            
            self.llava_model = PeftModel.from_pretrained(self.llava_model, resume_path)
            for name, param in self.llava_model.named_parameters():
                if "lora" in name.lower() or "adapter" in name.lower():
                    param.requires_grad = True
            
            def load_module(module: nn.Module, path: Path):
                state_dict = torch.load(path, map_location="cpu")
                module.load_state_dict(state_dict)
                
            load_module(self.llava_model.get_model().mm_projector, resume_path/"mm_projector.bin")

            load_module(self.ref_box_decoder, resume_path/"ref_box_decoder.pth")
            load_module(self.ref_learnable_embedding, resume_path/"ref_le.pth")

            load_module(self.vir_box_decoder, resume_path/"vir_box_decoder.pth")
            load_module(self.vir_learnable_embedding, resume_path/"vir_le.pth")
                        
            print(f"Model resumed from {resume}")
            
        else:            
            self.llava_model = get_peft_model(self.llava_model, lora_config)
            self.llava_model.to(torch.bfloat16)
            print("Resuming from None")
                
                
    def freeze_module(self):
        for param in self.llava_model.parameters():
            param.requires_grad = False

        
    def describe(self):
        def trainable_info(module):
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(type(module).__name__)
            print(f"Trainable: {trainable_params:,} Total parameters: {total_params:,}")
        
        trainable_info(self)
        trainable_info(self.ref_box_decoder)
        trainable_info(self.vir_box_decoder)


