import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import pandas as pd
from PIL import Image
# from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
# from src.components.helpers import tokenize_vicuna, get_activations_dir, get_vector_path, get_vector_dir
import argparse
from transformers import LlavaProcessor, AutoTokenizer, LlavaForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

model_name = 'Qwen/Qwen2-VL-7B-Instruct'
model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', trust_remote_code=True)
print('[Initialization Finished]\n')

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith((".png", ".jpg", ".jpeg", ".JPEG", ".bmp", ".gif")):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                img = img.resize((224, 224))
                images.append(img)
    return images
data_path = "image/path"
image_list = load_images_from_folder("{}".format(data_path))
print("Len of image_list: ", len(image_list))

layers = [14, 20, 28] # for 7B model Qwen
img_by_wrapper = {}

# empty_image = [processor(empty_image).unsqueeze(0).to('cuda')]
for index in range(len(image_list)):
    reference_img = image_list[index]
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None,},
                    {"type": "text", "text": "What is the image about?"},
                ],
            }
        ]
    text_prompt_template = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    with torch.no_grad(), torch.amp.autocast('cuda'):  

        inputs = processor(text=[text_prompt_template], images=[reference_img], return_tensors="pt").to("cuda", torch.float16)
        index_input_ids = inputs["input_ids"].shape[1]
        # print(text_prompt_template, index_input_ids)

        generate_ids = model.generate(**inputs, do_sample=True, max_length=512, temperature=0.2, top_p=0.9,)
        response = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        # print(response)

        inputs = processor(text=[text_prompt_template+response], images=[reference_img], return_tensors="pt").to("cuda", torch.float16)
        output = model(**inputs, output_hidden_states=True)
    
    img_activations = {}
    for layer in layers:
        hidden_states = output.hidden_states[layer].detach().cpu()
        img_activations[layer] =  torch.mean(hidden_states[0, index_input_ids:], dim=0)
    
    img_by_wrapper[index] = {layer: [] for layer in layers}
    for layer in layers:
        img_by_wrapper[index][layer].append(img_activations[layer])
      
save_path = "./activations/qwen/reference"
torch.save(img_by_wrapper, os.path.join(save_path, f"reference_activations.pt"))