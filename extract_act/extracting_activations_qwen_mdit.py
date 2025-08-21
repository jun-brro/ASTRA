import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import torch
import pandas as pd
from PIL import Image
import argparse
from transformers import LlavaProcessor, AutoTokenizer, LlavaForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def get_all_categories():
    """Get all available MDIT-Bench categories from the dataset directory"""
    mdit_path = "./datasets/MDIT-Bench"
    categories = []
    for file in os.listdir(mdit_path):
        if file.endswith('.json') and file != 'collection.json':  # collection.json might be a summary file
            category = file.replace('.json', '')
            categories.append(category)
    return sorted(categories)

def parse_args():
    all_categories = get_all_categories()
    parser = argparse.ArgumentParser(description="Extract QWEN MDIT Activations")
    parser.add_argument('--category', type=str, default='all', 
                        choices=['all'] + all_categories,
                        help='MDIT-Bench category to process, or "all" for all categories')
    return parser.parse_args()

def process_category_activations(category, model, processor, tokenizer):
    """Process activations for a single category"""
    print(f"\n{'='*60}")
    print(f"Processing activations for category: {category}")
    print(f"{'='*60}")
    
    # Load datasets for this category
    data_img_path = f"./results/adv_img_attr/qwen_mdit/{category}"
    queries_path = f'./results/queries/qwen_mdit_{category}.csv'

    # Check if the required files exist
    if not os.path.exists(queries_path):
        print(f"Error: Query file not found: {queries_path}")
        print("Please run extract_qwen_mdit_attr.py first!")
        return None
    
    if not os.path.exists(data_img_path):
        print(f"Error: Image attribution path not found: {data_img_path}")
        print("Please run extract_qwen_mdit_attr.py first!")
        return None
    
    # Load query data
    data_query_df = pd.read_csv(queries_path)
    data_query = data_query_df['Query'].tolist()
    print(f"Loaded {len(data_query)} queries")
    
    # Load images
    data_adv_img = []
    data_adv_img_attr = []
    data_adv_img_mask = []
    
    for index in range(len(data_query)):
        try:
            img_path = f"{data_img_path}/img{index}.bmp"
            attr_path = f"{data_img_path}/img{index}_attr.bmp"
            mask_path = f"{data_img_path}/img{index}_mask.bmp"
            
            if os.path.exists(img_path) and os.path.exists(attr_path) and os.path.exists(mask_path):
                data_adv_img.append(Image.open(img_path).convert("RGB"))
                data_adv_img_attr.append(Image.open(attr_path).convert("RGB"))
                data_adv_img_mask.append(Image.open(mask_path).convert("RGB"))
                print(f"Loaded image set {index}: {img_path}")
            else:
                print(f"Warning: Missing image files for index {index}")
        except Exception as e:
            print(f"Error loading image {index}: {e}")
            continue
    
    print(f"Loaded {len(data_adv_img)} image sets")
    
    if len(data_query) != len(data_adv_img):
        print(f"Warning: Mismatch between queries ({len(data_query)}) and images ({len(data_adv_img)})")
        # Take the minimum to avoid index errors
        min_len = min(len(data_query), len(data_adv_img))
        data_query = data_query[:min_len]
        data_adv_img = data_adv_img[:min_len]
        data_adv_img_attr = data_adv_img_attr[:min_len]
        data_adv_img_mask = data_adv_img_mask[:min_len]
        print(f"Using {min_len} matched pairs")
    
    # Define layers for 7B model Qwen
    layers = [14, 20, 28]
    diff_attr_by_wrapper = {}

    for index, (query, img, attr, mask) in enumerate(zip(data_query, data_adv_img, data_adv_img_attr, data_adv_img_mask)):
        print(f"Processing activation {index+1}/{len(data_query)}")
        
        # Create messages template
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None,},
                        {"type": "text", "text": ''},
                    ],
                }
            ]
        
        text_prompt_template = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        
        try:
            with torch.no_grad(), torch.amp.autocast('cuda'): 
                inputs = processor(text=[text_prompt_template]*2, images=[img, mask], padding=True, return_tensors="pt").to(model.device)
                output = model(**inputs, output_hidden_states=True)

            # Extract diff attributions
            diff_attr_activations = {}
            for layer in layers:
                hidden_states = output.hidden_states[layer].detach().cpu()
                diff_attr_activations[layer] = hidden_states[0, -1] - hidden_states[1, -1]
            
            # Store activations
            diff_attr_by_wrapper[index] = {layer: [] for layer in layers}
            for layer in layers:
                diff_attr_by_wrapper[index][layer].append(diff_attr_activations[layer])
                
            print(f"  Successfully extracted activations for layers {layers}")
            
        except Exception as e:
            print(f"Error processing activation {index}: {e}")
            continue
    
    # Save activations
    variable_element = 'visual_jail'  # Using the same naming convention
    save_path = "./activations/qwen/mdit"
    os.makedirs(save_path, exist_ok=True)
    
    output_file = os.path.join(save_path, f"mdit_diff_attr_activations_{variable_element}_{category}.pt")
    torch.save(diff_attr_by_wrapper, output_file)
    
    print(f"Category {category} activation extraction completed!")
    print(f"Activations saved to: {output_file}")
    print(f"Total processed: {len(diff_attr_by_wrapper)} samples")
    print(f"Layers: {layers}")
    
    return len(diff_attr_by_wrapper)

# Main execution
args = parse_args()

# Initialize model (once for all categories)
print("Initializing Qwen model...")
model_name = 'Qwen/Qwen2-VL-7B-Instruct'
model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print('[Initialization Finished]\n')

# Determine which categories to process
if args.category == 'all':
    categories_to_process = get_all_categories()
    print(f"Processing activations for ALL {len(categories_to_process)} categories: {categories_to_process}")
else:
    categories_to_process = [args.category]
    print(f"Processing activations for single category: {args.category}")

# Process each category
total_processed = 0
successful_categories = 0

for category in categories_to_process:
    try:
        processed = process_category_activations(category, model, processor, tokenizer)
        if processed is not None:
            total_processed += processed
            successful_categories += 1
    except Exception as e:
        print(f"Error processing category {category}: {e}")
        continue

print(f"\n{'='*60}")
print(f"OVERALL ACTIVATION EXTRACTION SUMMARY")
print(f"{'='*60}")
print(f"Categories processed successfully: {successful_categories}/{len(categories_to_process)}")
print(f"Total activations extracted: {total_processed}")
print(f"Activation extraction completed for all available categories!")
