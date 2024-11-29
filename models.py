from torch import Tensor
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
# from utils import create_mask

import torchvision
from torch.nn.utils.rnn import pad_sequence
#import pytorchvideo.models.x3d as x3d
import utils as utils

""" PyTorch MBART model."""
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding, MBartEncoderLayer, _expand_mask

from collections import OrderedDict


import copy
import math
import random
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# global definition
from definition import *

from hpman.m import _
from pathlib import Path

from peft import LoraConfig, get_peft_model, TaskType


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


def make_resnet(name='resnet18'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model {}.'.format(_('resnet')))

    inchannel = model.fc.in_features
    model.fc = nn.Identity()
    return model

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True)
        return x
  
class TemporalConv(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.temporal_conv(x.permute(0,2,1))
        return x.permute(0,2,1)
    
def make_head(inplanes, planes, head_type):
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()

class TextCLIP(nn.Module):
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identy'):
        super(TextCLIP, self).__init__()

        # Load the MBart model
        mbart_model = MBartForConditionalGeneration.from_pretrained(config['model']['transformer'])

        # # ----------------------------------------------------------
        # # Apply LoRA to mbart_model
        # lora_config = LoraConfig(
        #     task_type=TaskType.FEATURE_EXTRACTION,
        #     r=8,
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     bias="none",
        #     target_modules=['q_proj', 'k_proj', 'v_proj', "out_proj"]
        # )
        # mbart_model = get_peft_model(mbart_model, lora_config)
        #
        #
        # # Freeze non-LoRA parameters
        # for name, param in mbart_model.named_parameters():
        #     if 'lora' not in name:
        #         param.requires_grad = False
        # # -----------------------------------------------------------

        # Use the encoder part of the LoRA-adapted MBart model
        self.model_txt = mbart_model.get_encoder()
        self.lm_head = make_head(inplanes, planes, head_type)

        # Ensure lm_head parameters are trainable
        for param in self.lm_head.parameters():
            param.requires_grad = True

    def forward(self, tgt_input):
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].cuda(),
                                    attention_mask=tgt_input['attention_mask'].cuda())[0]
        output = txt_logits[torch.arange(txt_logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        return self.lm_head(output), txt_logits

class ImageCLIP(nn.Module):
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear'):
        super(ImageCLIP, self).__init__()
        self.config = config
        self.model = FeatureExtracter()

        # Load the MBart model
        mbart_model = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'])

        # # ----------------------------------------------------------
        # # Apply LoRA to mbart_model
        # lora_config = LoraConfig(
        #     task_type=TaskType.FEATURE_EXTRACTION,
        #     r=8,
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     bias="none",
        #     target_modules=['q_proj', 'k_proj', 'v_proj', "out_proj"]
        # )
        # mbart_model = get_peft_model(mbart_model, lora_config)
        #
        # # Freeze non-LoRA parameters
        # for name, param in mbart_model.named_parameters():
        #     if 'lora' not in name:
        #         param.requires_grad = False
        #
        # # ----------------------------------------------------------

        # Use the encoder part of the LoRA-adapted MBart model
        self.trans_encoder = mbart_model.get_encoder()
        self.cls_token = nn.Parameter(torch.randn(1, 1, inplanes))

        self.lm_head = make_head(inplanes, planes, head_type)

        # Ensure lm_head and cls_token parameters are trainable
        for param in self.lm_head.parameters():
            param.requires_grad = True
        self.cls_token.requires_grad = True

    def forward(self, src_input):

        x = self.model(src_input['input_ids'].cuda(), src_input['src_length_batch']) # [b, n, c]
        attention_mask = src_input['attention_mask']

        B, N, C = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat((cls_token, x), dim=1)
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.)  # [b, 64] --> [b, 65]

        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask.cuda(), return_dict=True)
        last_hidden_state = outs['last_hidden_state']
        output = self.lm_head(last_hidden_state[:, 0, :])
        return output

class Text_Decoder(nn.Module):
    def __init__(self, config):
        super(Text_Decoder, self).__init__()

        # Load the MBart model
        mbart_model = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'])

        # # ----------------------------------------------------------
        # # Apply LoRA to mbart_model
        # lora_config = LoraConfig(
        #     task_type=TaskType.SEQ_2_SEQ_LM,
        #     r=8,
        #     lora_alpha=16,
        #     lora_dropout=0.1,
        #     bias="none",
        #     target_modules=['q_proj', 'k_proj', 'v_proj', "out_proj"]
        # )
        # mbart_model = get_peft_model(mbart_model, lora_config)
        #
        # # Freeze non-LoRA parameters
        # for name, param in mbart_model.named_parameters():
        #     if 'lora' not in name:
        #         param.requires_grad = False
        #     else:
        #         param.requires_grad = True
        # # ----------------------------------------------------------

        # Use the decoder part of the LoRA-adapted MBart model
        self.text_decoder = mbart_model.get_decoder()
        self.lm_head = mbart_model.get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, mbart_model.config.vocab_size)))

        # Ensure lm_head parameters are trainable
        for param in self.lm_head.parameters():
            param.requires_grad = True
    
    def forward(self, tgt_input, masked_tgt_input, model_txt):
        with torch.no_grad():
            _, encoder_hidden_states = model_txt(masked_tgt_input)

        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        decoder_out = self.text_decoder(
                    input_ids = decoder_input_ids,
                    attention_mask = tgt_input['attention_mask'].cuda(),
                    encoder_hidden_states = encoder_hidden_states,
                    encoder_attention_mask = masked_tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    )
        lm_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias

        return lm_logits
    
        
class SLRCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024) :
        super(SLRCLIP, self).__init__()
        self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim)
        self.model_images = ImageCLIP(config, inplanes=embed_dim, planes=embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # # ----------------------------------------------------------
        # # Optionally, unfreeze parameters that you want to train
        # for name, param in self.named_parameters():
        #     if 'lora' in name or 'backbone' in name or 'lm_head' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # # ----------------------------------------------------------

    def get_model_txt(self):
        return self.model_txt
    
    @property
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states
    
    def forward(self, src_input, tgt_input):
        image_features = self.model_images(src_input)
        text_features, self.encoder_hidden_states = self.model_txt(tgt_input)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        ground_truth = torch.eye(logits_per_image.shape[0], device=logits_per_text.device, dtype=logits_per_image.dtype, requires_grad=False)

        return logits_per_image, logits_per_text, ground_truth

class FeatureExtracter(nn.Module):
    def __init__(self, frozen=False):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet() # InceptionI3d()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)

        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False

    def forward(self,
                src: Tensor,
                src_length_batch
                ):
        src = self.conv_2d(src,src_length_batch)
        src = self.conv_1d(src)

        return src

class V_encoder(nn.Module):
    def __init__(self,
                 emb_size,
                 feature_size,
                 config,
                 ):
        super(V_encoder, self).__init__()
        
        self.config = config

        self.src_emb = nn.Linear(feature_size, emb_size)
        modules = []
        modules.append(nn.BatchNorm1d(emb_size))
        modules.append(nn.ReLU(inplace=True))
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,
                src: Tensor,
                ):
      
        src = self.src_emb(src)
        src = self.bn_ac(src.permute(0,2,1)).permute(0,2,1)

        return src

def config_decoder(config):
    from transformers import AutoConfig
    
    decoder_type = _('decoder_type', 'LD', choices=['LD', 'LLMD', 'MB50MMT'])
    if decoder_type == 'LD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['visual_encoder'])/'config.json'))
    elif decoder_type == 'LLMD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['transformer'])/'LLMD_config.json'))
    elif decoder_type == 'MB50MMT':
        return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['transformer'])/'MB50MMT_config_new.json'))

class gloss_free_model(nn.Module):
    def __init__(self, config, args, embed_dim=1024):
        super(gloss_free_model, self).__init__()
        self.config = config
        self.args = args

        self.backbone = FeatureExtracter(frozen=_('freeze_backbone', False))
        self.mbart = config_decoder(config)

        # Apply LoRA to mbart model
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=['q_proj',
                            'k_proj',
                            'v_proj',
                            "out_proj"]
        )
        self.mbart = get_peft_model(self.mbart, lora_config)

        # # Freeze non-LoRA parameters (we'll unfreeze them selectively after loading the checkpoint)
        # for name, param in self.mbart.named_parameters():
        #     if 'lora' not in name:


        if config['model']['sign_proj']:
            self.sign_emb = V_encoder(emb_size=embed_dim, feature_size=embed_dim, config=config)
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0

        # After applying LoRA
        if args.finetune:
            print('***********************************')
            print('Loading pretrained weights...')
            print('***********************************')
            checkpoint = torch.load(args.finetune, map_location='cpu')

            new_state_dict = OrderedDict()

            # Load backbone weights
            for k, v in checkpoint['model'].items():
                if k.startswith('model_images.model.conv_2d') or k.startswith('model_images.model.conv_1d'):
                    new_k = k.replace('model_images.model', 'backbone')
                    new_state_dict[new_k] = v

            # Load encoder weights (including LoRA weights)
            for k, v in checkpoint['model'].items():
                if k.startswith('model_images.trans_encoder'):
                    new_k = k.replace('model_images.trans_encoder', 'mbart.base_model.model.model.encoder')
                    new_state_dict[new_k] = v

            # Load decoder weights (including LoRA weights)
            for k, v in checkpoint['text_decoder'].items():
                if k.startswith('module.text_decoder'):
                    new_k = k.replace('module.text_decoder', 'mbart.base_model.model.model.decoder')
                    new_state_dict[new_k] = v
                elif k == 'module.lm_head.weight':
                    new_state_dict['mbart.base_model.model.lm_head.weight'] = v
                elif k == 'module.final_logits_bias':
                    new_state_dict['mbart.base_model.model.final_logits_bias'] = v





            # Load embeddings if they are in the checkpoint
            # Else, initialize embeddings from scratch or load from pre-trained MBart

            # Load the state dict into the model
            ret = self.load_state_dict(new_state_dict, strict=False)
            print('Missing keys: \n', '\n'.join(ret.missing_keys))
            print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

            # Optionally, unfreeze parameters that you want to train
            for name, param in self.mbart.named_parameters():
                param.requires_grad = False
                # if 'backbone' in name or 'mbart' in name:
                if 'lora' in name or 'backbone' in name:
                    param.requires_grad = True

                if 'lm_head' in name or 'shared' in name:
                    param.requires_grad = True

            for param in self.backbone.conv_2d.parameters():
                param.requires_grad = True

    def share_forward(self, src_input):
        frames_feature = self.backbone(src_input['input_ids'].cuda(), src_input['src_length_batch'])
        attention_mask = src_input['attention_mask']

        inputs_embeds = self.sign_emb(frames_feature)
        inputs_embeds = self.embed_scale * inputs_embeds

        return inputs_embeds, attention_mask

    def forward(self, src_input, tgt_input):
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart(inputs_embeds=inputs_embeds,
                         attention_mask=attention_mask.cuda(),
                         labels=tgt_input['input_ids'].cuda(),
                         decoder_attention_mask=tgt_input['attention_mask'].cuda(),
                         return_dict=True)
        return out['logits']

    def generate(self, src_input, max_new_tokens, num_beams, decoder_start_token_id):
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart.generate(inputs_embeds=inputs_embeds,
                                  attention_mask=attention_mask.cuda(),
                                  max_new_tokens=max_new_tokens,
                                  num_beams=num_beams,
                                  decoder_start_token_id=decoder_start_token_id)
        return out