import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import pandas as pd
from PIL import Image
# from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
# from src.components.helpers import tokenize_vicuna, get_activations_dir, get_vector_path, get_vector_dir

import argparse
from minigpt_utils import visual_attacker, prompt_wrapper, generator

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('[Initialization Finished]\n')

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".JPEG", ".bmp", ".gif")):
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            if img is not None:
                images.append(img)
    return images
data_path = "image/path"
image_list = load_images_from_folder("{}".format(data_path))
print("Len of image_list: ", len(image_list))
generator = generator.Generator(model=model)

# layers = [16, 32] # for 7B model
layers = [20, 40] # for 13B model
img_by_wrapper = {}
for index in range(len(image_list)):
    reference_img = image_list[index]
    reference_img = [processor(reference_img).unsqueeze(0).to('cuda')]
    prefix = prompt_wrapper.minigpt4_chatbot_prompt    
    query = prefix % ('What is the image about?')

    with torch.no_grad(), torch.amp.autocast('cuda'):
        prompt_wrap = prompt_wrapper.Prompt(model=model, 
                                            text_prompts=[query],
                                            img_prompts=[reference_img])
        index_input_ids = prompt_wrap.input_tokens
        
        responses, _ = generator.generate(prompt_wrap)
        prompt_wrap = prompt_wrapper.Prompt(model=model, 
                                            text_prompts=[query+responses],
                                            img_prompts=[reference_img])
        output_img = model.llama_model(inputs_embeds=prompt_wrap.context_embs[0], output_hidden_states=True)
        
    img_activations = {}
    for layer in layers:
        hidden_img = output_img.hidden_states[layer].detach().cpu()
        img_activations[layer] = torch.mean(hidden_img[0, index_input_ids[0].shape[1]+32+index_input_ids[1].shape[1]:], axis=0)
      
    img_by_wrapper[index] = {layer: [] for layer in layers}
    for layer in layers:
        img_by_wrapper[index][layer].append(img_activations[layer])
        
save_path = "./activations/qwen/reference"
torch.save(img_by_wrapper, os.path.join(save_path, f"reference_activations.pt"))