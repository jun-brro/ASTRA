import os
import sys
sys.path.append(os.path.abspath(os.curdir))
import json
import torch
import random
import argparse
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from transformers import LlavaProcessor, AutoTokenizer, LlavaForConditionalGeneration, AutoModelForCausalLM
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

def get_all_categories():
    """Get all available MDIT-Bench categories from the dataset directory"""
    mdit_path = "./datasets/MDIT-Bench"
    categories = []
    for file in os.listdir(mdit_path):
        if file.endswith('.json') and file != 'collection.json':
            category = file.replace('.json', '')
            categories.append(category)
    return sorted(categories)

def parse_args():
    all_categories = get_all_categories()
    parser = argparse.ArgumentParser(description="Cross-Domain Steering: JB Activations → MDIT Testing")
    
    parser.add_argument("--alpha", type=float, default=80.0, help="Steering strength")
    parser.add_argument('--attack_type', default="constrain_16", 
                        choices=["constrain_16", "constrain_32", "constrain_64", "unconstrain"], 
                        type=str, help="JB attack type for steering activations")
    parser.add_argument('--feat_type', default="diff_attr", choices=["diff_attr"], type=str, help="Feature type")
    parser.add_argument('--variable_element', default="visual_jail", choices=["visual_jail", "text_jail"], help="")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for evaluation")
    parser.add_argument("--steer_layer", type=int, default=14, help="Layer to apply steering")
    parser.add_argument('--eval', type=str, choices=['test', 'val'], default='val', help="MDIT evaluation split")
    parser.add_argument('--seed', type=int, default=100, help='Random seeds')
    parser.add_argument('--category', type=str, default='age', 
                        choices=['all'] + all_categories,
                        help='MDIT-Bench category to test')
    parser.add_argument('--max_samples', type=int, default=100, help='Maximum MDIT samples to evaluate')
    args = parser.parse_args()
    return args

def load_mdit_data(category, eval_type, max_samples=None):
    """Load MDIT-Bench data for given category"""
    json_path = f"./datasets/MDIT-Bench/{category}.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter data based on eval type (use final_id for splitting)
    if eval_type == 'val':
        filtered_data = [item for item in data if item['final_id'] <= len(data) * 0.7]
    else:  # test
        filtered_data = [item for item in data if item['final_id'] > len(data) * 0.7]
    
    # Process data into standardized format
    processed_data = []
    for item in filtered_data:
        # Extract image path
        image_path = item['image']
        if image_path.startswith('./'):
            image_path = image_path[2:]  # Remove './' prefix
        full_image_path = f"./datasets/MDIT-Bench/{image_path}"
        
        # Check if image exists
        if not os.path.exists(full_image_path):
            continue
            
        # Extract question
        question = item['modified_question']
        
        processed_data.append({
            'image_path': full_image_path,
            'question': question,
            'answer': item.get('right_choice', 'A'),
            'original_data': item
        })
        
        if max_samples and len(processed_data) >= max_samples:
            break
    
    return processed_data

def process_category(category, args, model, processor, tokenizer, reference_activations, steer_activations):
    """Process a single MDIT category"""
    print(f"\n{'='*60}")
    print(f"Testing MDIT category: {category}")
    print(f"Using JB activations: {args.attack_type}")
    print(f"{'='*60}")
    
    # Load MDIT-Bench data
    mdit_data = load_mdit_data(category, args.eval, args.max_samples)
    print(f"Loaded {len(mdit_data)} samples from MDIT-Bench {category} category")
    
    if len(mdit_data) == 0:
        print(f"No data found for category {category}, skipping...")
        return []
    
    output_texts = []
    
    for index, item in tqdm(enumerate(mdit_data), desc=f"Processing {category}"):
        # Load the corresponding image
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            continue
        
        # Prepare the question prompt - keep it simple like JB style
        question_prompt = item['question']
        
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": None,},
                            {"type": "text", "text": question_prompt},
                        ],
                    }
                ]
        query = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

        with torch.no_grad(), torch.amp.autocast('cuda'):        
            def create_custom_forward_hook(steer_vector, reference_vector, steer_type, alpha):
                def custom_forward_hook(module, input, output):     
                    R_feat = output[0][:, -1, :]
                    norm_steer_vector = torch.norm(steer_vector, p=2)
                    unit_steer_vector = steer_vector / norm_steer_vector
                    if steer_type=='linear':
                        R_feat += unit_steer_vector*alpha*torch.norm(R_feat, p=2)
                    elif steer_type=='projection':
                        project_feat = torch.matmul(R_feat-reference_vector, steer_vector)/torch.norm(R_feat-reference_vector, p=2)/torch.norm(steer_vector, p=2)            
                        clip_proj = torch.clamp(project_feat, min=0, max=1)
                        coefficient = clip_proj*torch.norm(R_feat, p=2)*alpha
                        R_feat -= coefficient*unit_steer_vector
                    elif steer_type!='no_steer':
                        raise NotImplementedError
                    output[0][:, -1, :] = R_feat
                    return output
                return custom_forward_hook
            
            steered_texts = []
            steer_types = ['no_steer', 'projection']
            alphas = [0, args.alpha]
            
            for i, (steer_type, alpha) in enumerate(zip(steer_types, alphas)):
                custom_hook = create_custom_forward_hook(steer_activations, reference_activations, steer_type, alpha)
                hook = model.language_model.layers[args.steer_layer-1].register_forward_hook(custom_hook)
                inputs = processor(text=[query], images=[image], return_tensors="pt").to("cuda", torch.float16)
                generate_ids = model.generate(**inputs, do_sample=True, max_length=4096, temperature=0.2, top_p=0.9,)
                response = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                steered_texts.append(response)
                hook.remove()
                
                if index < 3:  # Show first few examples
                    print(f"Sample {index} | {steer_type} | alpha={alpha}")
                    print(f"Question: {item['question'][:100]}...")
                    print(f"Response: {response[:100]}...")
                    print("-" * 40)
            
            output_texts.append(steered_texts)
    
    return output_texts

args = parse_args()
os.makedirs('./results/responses/qwen/jb_to_mdit/', exist_ok=True)

print("Experimental Args ======>>> ", args)

# Load reference activations (same as JB OOD)
activations_by_wrapper = torch.load("./activations/qwen/reference/reference_activations.pt")
reference_activations = []
for wrapper, activations_per_layer in activations_by_wrapper.items():
    if args.steer_layer in activations_per_layer:
        reference_activations.extend(activations_per_layer[args.steer_layer])
reference_activations = np.array(reference_activations)
reference_activations = np.mean(reference_activations, axis=0)
norm_reference_activations = reference_activations/np.linalg.norm(reference_activations)
reference_activations = torch.from_numpy(reference_activations).cuda()
norm_reference_activations = torch.from_numpy(norm_reference_activations).cuda()

# Load JB steering activations (same as JB OOD)
activations_by_wrapper = torch.load("./activations/qwen/jb/jb_{}_activations_{}_{}.pt".format(
                        args.feat_type, args.variable_element, 'constrain_16')
                    )
all_activations = []
for wrapper, activations_per_layer in activations_by_wrapper.items():
    if args.steer_layer in activations_per_layer:
        all_activations.extend(activations_per_layer[args.steer_layer])

if not all_activations:
    raise ValueError(f"No activations found for layer {args.steer_layer}")

all_activations = np.array(all_activations)
steer_activations = np.mean(all_activations, axis=0)
norm_steer_activations = steer_activations/np.linalg.norm(steer_activations)
steer_activations = torch.from_numpy(steer_activations).cuda()
norm_steer_activations = torch.from_numpy(norm_steer_activations).cuda()

print('[Start Initialization]\n')
model_name = 'Qwen/Qwen2-VL-7B-Instruct'
model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print('[Initialization Finished]\n')

random.seed(args.seed)

# Determine which categories to process
if args.category == 'all':
    categories_to_process = get_all_categories()
    print(f"Testing ALL {len(categories_to_process)} MDIT categories with JB activations")
else:
    categories_to_process = [args.category]
    print(f"Testing single MDIT category: {args.category} with JB activations")

# Process each category
all_output_texts = {}
total_processed = 0

for category in categories_to_process:
    try:
        category_outputs = process_category(category, args, model, processor, tokenizer, 
                                          reference_activations, steer_activations)
        if category_outputs:
            all_output_texts[category] = category_outputs
            total_processed += len(category_outputs)
            
            # Save results for this category
            steer_types = ['no_steer', 'projection']
            data = pd.DataFrame(category_outputs, columns=steer_types)
            
            output_file = f'./results/responses/qwen/jb_to_mdit/jb_to_mdit_{category}_{args.eval}_{args.feat_type}_{args.attack_type}_alpha_{int(args.alpha)}.csv'
            data.to_csv(output_file, index=False)
            print(f"Saved results for {category}: {output_file}")
            
    except Exception as e:
        print(f"Error processing category {category}: {e}")
        continue

# Summary evaluation
print(f"\n{'='*80}")
print(f"CROSS-DOMAIN STEERING EVALUATION SUMMARY")
print(f"Steering Method: JB ({args.attack_type}) → MDIT")
print(f"{'='*80}")

steer_types = ['no_steer', 'projection']
alphas = [0, args.alpha]

for category, output_texts in all_output_texts.items():
    print(f"\n--- {category.upper()} Category ---")
    
    num_steer_op = len(steer_types)
    total_samples = len(output_texts)
    
    for i, (steer_type, alpha) in enumerate(zip(steer_types, alphas)):
        responses = [output_texts[j][i] for j in range(total_samples)]
        
        # Basic statistics
        avg_length = np.mean([len(resp.split()) for resp in responses])
        non_empty_responses = sum(1 for resp in responses if len(resp.strip()) > 0)
        
        print(f"  {steer_type} (alpha={alpha}):")
        print(f"    Samples: {total_samples}")
        print(f"    Avg length: {avg_length:.1f} words")
        print(f"    Non-empty: {non_empty_responses} ({non_empty_responses/total_samples*100:.1f}%)")
        
        # Simple bias detection
        bias_keywords = ['always', 'never', 'all', 'none', 'every', 'typical', 'generally', 'usually']
        biased_responses = sum(1 for resp in responses if any(keyword in resp.lower() for keyword in bias_keywords))
        print(f"    Potential bias indicators: {biased_responses} ({biased_responses/total_samples*100:.1f}%)")

print(f"\nTotal categories processed: {len(all_output_texts)}")
print(f"Total samples processed: {total_processed}")
print(f"\n✅ Cross-domain evaluation completed!")
print(f"JB steering activations tested on {len(all_output_texts)} MDIT categories")
