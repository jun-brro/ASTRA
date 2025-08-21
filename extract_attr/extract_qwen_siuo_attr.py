import os
import sys
sys.path.append(os.path.abspath(os.curdir))
from image_attr import ImageAttr_QWEN
import torch
import pandas as pd
import json
import random
import argparse
from PIL import Image
import requests
import numpy as np
from datasets import load_dataset
import csv

def get_all_categories():
    """Get all available SIUO categories from the dataset"""
    json_path = "./datasets/SIUO/data/siuo_gen.json"
    categories = set()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        categories.add(item['category'])
    
    return sorted(list(categories))

def parse_args():
    all_categories = get_all_categories()
    parser = argparse.ArgumentParser(description="Extract QWEN SIUO Attribution")
    parser.add_argument('--category', type=str, default='all', 
                        choices=['all'] + all_categories,
                        help='SIUO category to process, or "all" for all categories')
    parser.add_argument('--max_samples', type=int, default=50, help='Maximum samples to process per category')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--task_type', type=str, default='gen', choices=['gen', 'mcqa'], 
                        help='Task type: gen (generation) or mcqa (multiple choice)')
    return parser.parse_args()

def load_siuo_data(category, task_type, max_samples=None):
    """Load SIUO data for given category"""
    if task_type == 'gen':
        json_path = "./datasets/SIUO/data/siuo_gen.json"
    else:
        json_path = "./datasets/SIUO/data/siuo_mcqa.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter data by category if specified
    if category != 'all':
        data = [item for item in data if item['category'] == category]
    
    # Process data and extract image-question pairs
    processed_data = []
    for item in data[:max_samples] if max_samples else data:
        # Extract image path
        image_filename = item['image']
        full_image_path = f"./datasets/SIUO/data/images/{image_filename}"
        
        # Check if image exists
        if not os.path.exists(full_image_path):
            print(f"Warning: Image not found: {full_image_path}")
            continue
            
        # Extract question
        question = item['question']
        
        processed_data.append({
            'question_id': item['question_id'],
            'image_path': full_image_path,
            'question': question,
            'category': item['category'],
            'original_data': item
        })
    
    return processed_data

def process_category(category, max_samples, seed, ImgAttr, top_k, task_type):
    """Process a single category"""
    print(f"\n{'='*60}")
    print(f"Processing SIUO category: {category} (task: {task_type})")
    print(f"{'='*60}")
    
    # Load SIUO data
    siuo_data = load_siuo_data(category, task_type, max_samples)
    print(f"Loaded {len(siuo_data)} samples from SIUO {category} category")
    
    if len(siuo_data) == 0:
        print(f"No data found for category {category}, skipping...")
        return 0, 0
    
    # Create output directories
    save_path = f"./results/adv_img_attr/qwen_siuo/{task_type}/{category}"
    os.makedirs(save_path, exist_ok=True)
    
    data_query = []
    processed_count = 0
    
    for index, item in enumerate(siuo_data):
        try:
            # Load image
            image = Image.open(item['image_path']).convert('RGB')
            print(f"Processing {index+1}/{len(siuo_data)}: {item['image_path']}")
            
            # Use the specific question from SIUO
            query = item['question']
            
            # Create messages for the model
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": None,},
                        {"type": "text", "text": query},
                    ],
                }
            ]
            
            text_prompt_template = ImgAttr.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Initialize image and query for attribution
            ImgAttr.init_image_query(image, text_prompt_template, num_ablations=64, ablation_keep_prob=0.50) 
            ImgAttr.partitioner.set_p(16)
            
            # For safety-related queries, we look for harmful response patterns
            ImgAttr.specify_output('Sure, ')
            results = ImgAttr.get_attributions(as_dataframe=True, top_k=top_k, verbose=False)
            
            # Extract attribution information
            patch_index = ImgAttr.partitioner.patch_index
            attr_index = []
            
            for _, (score, source, patch) in enumerate(zip(results['Score'], results['Source'], results['Patch'])):
                if score > 1e-4:
                    attr_index.append(patch[0] + patch[1] * patch_index)
            
            if len(attr_index) > 0:
                # Save attribution visualization
                img_attr = ImgAttr.partitioner.visualize_attr(np.array(attr_index))
                img_attr.save(f'{save_path}/img{index}_attr.bmp')
                
                # Save original image
                img_orig = ImgAttr.partitioner.get_image()
                img_orig.save(f'{save_path}/img{index}.bmp')
                
                # Save mask (inverted attribution)
                img_mask = ImgAttr.partitioner.visualize_attr(np.array(attr_index), flip=True)
                img_mask.save(f'{save_path}/img{index}_mask.bmp')
                
                # Record the query
                data_query.append({
                    'QuestionID': item['question_id'],
                    'Query': query,
                    'ImagePath': item['image_path'],
                    'Category': item['category'],
                    'TaskType': task_type
                })
                processed_count += 1
                
                print(f"  Processed successfully: {len(attr_index)} attribution patches found")
            else:
                print(f"  Skipped: No significant attributions found")
                
        except Exception as e:
            print(f"Error processing item {index}: {e}")
            continue
    
    # Save query data for this category
    print(f"\nSaving results for {processed_count} processed samples from {category}...")
    header = ['QuestionID', 'Query', 'ImagePath', 'Category', 'TaskType']
    data_df = pd.DataFrame(data_query, columns=header)
    queries_save_path = f'./results/queries/qwen_siuo_{task_type}_{category}.csv'
    os.makedirs(os.path.dirname(queries_save_path), exist_ok=True)
    data_df.to_csv(queries_save_path, index=False)
    
    print(f"Category {category} completed!")
    print(f"Images saved to: {save_path}")
    print(f"Queries saved to: {queries_save_path}")
    print(f"Total processed: {processed_count}/{len(siuo_data)}")
    
    return processed_count, len(siuo_data)

# Main execution
args = parse_args()
random.seed(args.seed)

# Initialize ImageAttr for Qwen (once for all categories)
print("Initializing ImageAttr_QWEN...")
ImgAttr = ImageAttr_QWEN.from_pretrained()
top_k = 15

# Determine which categories to process
if args.category == 'all':
    categories_to_process = get_all_categories()
    print(f"Processing ALL {len(categories_to_process)} categories: {categories_to_process}")
else:
    categories_to_process = [args.category]
    print(f"Processing single category: {args.category}")

# Process each category
total_processed = 0
total_samples = 0

for category in categories_to_process:
    try:
        processed, samples = process_category(category, args.max_samples, args.seed, ImgAttr, top_k, args.task_type)
        total_processed += processed
        total_samples += samples
    except Exception as e:
        print(f"Error processing category {category}: {e}")
        continue

print(f"\n{'='*60}")
print(f"OVERALL SUMMARY")
print(f"{'='*60}")
print(f"Task type: {args.task_type}")
print(f"Categories processed: {len(categories_to_process)}")
print(f"Total samples: {total_samples}")
print(f"Total processed: {total_processed}")
print(f"Success rate: {total_processed/total_samples*100:.1f}%" if total_samples > 0 else "No samples processed")
print(f"Attribution extraction completed for all categories!")
