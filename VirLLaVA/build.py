import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler
from torch.utils.data import Dataset

import transformers
from model import *



def build_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            'liuhaotian/llava-v1.5-7b',
            cache_dir=None,
            model_max_length=2048,
            padding_side="right",
            use_fast=False,
        )
    tokenizer.pad_token = tokenizer.unk_token
    return tokenizer



def build_model_tokenizer(model_args, data_args, training_args):
    tokenizer = build_tokenizer()
    model = MetaLLaVA(model_args, data_args, training_args, token_len=32)
    
    
    model.llava_model.get_model().initialize_vision_modules(
        model_args=model.model_args,
        fsdp=model.training_args.fsdp
    )
    
    vision_tower = model.llava_model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=model.training_args.device)

    model.data_args.image_processor = vision_tower.image_processor
    model.data_args.is_multimodal = True

    model.llava_model.config.image_aspect_ratio = model.data_args.image_aspect_ratio
    model.llava_model.config.tokenizer_padding_side = tokenizer.padding_side
    model.llava_model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.llava_model.config.tune_mm_mlp_adapter = model.training_args.tune_mm_mlp_adapter = model.model_args.tune_mm_mlp_adapter
    if model.model_args.tune_mm_mlp_adapter:
        model.llava_model.requires_grad_(False)
        for p in model.llava_model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.llava_model.config.freeze_mm_mlp_adapter = model.training_args.freeze_mm_mlp_adapter
    if model.training_args.freeze_mm_mlp_adapter:
        for p in model.llava_model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.llava_model.config.mm_use_im_start_end = model.data_args.mm_use_im_start_end = model.model_args.mm_use_im_start_end
    model.llava_model.config.mm_projector_lr = model.training_args.mm_projector_lr
    model.llava_model.config.mm_use_im_patch_token = model.model_args.mm_use_im_patch_token
    model.llava_model.initialize_vision_tokenizer(model.model_args, tokenizer=tokenizer)
    
    return tokenizer, model




def build_model_tokenizerv2(model_args, data_args, training_args):
    tokenizer = build_tokenizer()
    model = MetaLLaVA(model_args, data_args, training_args, token_len=64)
    
    
    model.llava_model.get_model().initialize_vision_modules(
        model_args=model.model_args,
        fsdp=model.training_args.fsdp
    )
    
    vision_tower = model.llava_model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=model.training_args.device)

    model.data_args.image_processor = vision_tower.image_processor
    model.data_args.is_multimodal = True

    model.llava_model.config.image_aspect_ratio = model.data_args.image_aspect_ratio
    model.llava_model.config.tokenizer_padding_side = tokenizer.padding_side
    model.llava_model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.llava_model.config.tune_mm_mlp_adapter = model.training_args.tune_mm_mlp_adapter = model.model_args.tune_mm_mlp_adapter
    if model.model_args.tune_mm_mlp_adapter:
        model.llava_model.requires_grad_(False)
        for p in model.llava_model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.llava_model.config.freeze_mm_mlp_adapter = model.training_args.freeze_mm_mlp_adapter
    if model.training_args.freeze_mm_mlp_adapter:
        for p in model.llava_model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.llava_model.config.mm_use_im_start_end = model.data_args.mm_use_im_start_end = model.model_args.mm_use_im_start_end
    model.llava_model.config.mm_projector_lr = model.training_args.mm_projector_lr
    model.llava_model.config.mm_use_im_patch_token = model.model_args.mm_use_im_patch_token
    model.llava_model.initialize_vision_tokenizer(model.model_args, tokenizer=tokenizer)
    
    return tokenizer, model



def build_model_tokenizerv3(model_args, data_args, training_args):
    tokenizer = build_tokenizer()
    model = MetaLLaVAv3(model_args, data_args, training_args, token_len=32)
    
    model.llava_model.get_model().initialize_vision_modules(
        model_args=model.model_args,
        fsdp=model.training_args.fsdp
    )
    
    vision_tower = model.llava_model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=model.training_args.device)

    model.data_args.image_processor = vision_tower.image_processor
    model.data_args.is_multimodal = True

    model.llava_model.config.image_aspect_ratio = model.data_args.image_aspect_ratio
    model.llava_model.config.tokenizer_padding_side = tokenizer.padding_side
    model.llava_model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.llava_model.config.tune_mm_mlp_adapter = model.training_args.tune_mm_mlp_adapter = model.model_args.tune_mm_mlp_adapter
    if model.model_args.tune_mm_mlp_adapter:
        model.llava_model.requires_grad_(False)
        for p in model.llava_model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.llava_model.config.freeze_mm_mlp_adapter = model.training_args.freeze_mm_mlp_adapter
    if model.training_args.freeze_mm_mlp_adapter:
        for p in model.llava_model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.llava_model.config.mm_use_im_start_end = model.data_args.mm_use_im_start_end = model.model_args.mm_use_im_start_end
    model.llava_model.config.mm_projector_lr = model.training_args.mm_projector_lr
    model.llava_model.config.mm_use_im_patch_token = model.model_args.mm_use_im_patch_token
    model.llava_model.initialize_vision_tokenizer(model.model_args, tokenizer=tokenizer)
    
    return tokenizer, model
