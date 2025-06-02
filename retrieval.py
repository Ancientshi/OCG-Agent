from core.ocg_agent import candidate_retrieval
import json
import os
import sys
import argparse
import threading
import concurrent.futures
from datetime import datetime
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='OCG_movie_296')
parser.add_argument('--max_workers', type=int, default=1)
parser.add_argument('--pagenum', type=int, default=3)
parser.add_argument('--threshold', type=float, default=0.0)
parser.add_argument('--span', type=str2bool, default=False)
parser.add_argument('--complete', type=str2bool, default=True)
parser.add_argument('--max_loop_times', type=int, default=3)
parser.add_argument('--gpt_aug', type=str2bool, default=True)
parser.add_argument('--rewrite', type=str2bool, default=True)
parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='model name for gpt_aug')
#save_path
parser.add_argument('--save_path', type=str, default='AI_search_content', help='path to save the result')

args = parser.parse_args()

os.makedirs(os.path.join(args.save_path, args.dataset_name), exist_ok=True)

dataset_name = args.dataset_name
test_file = f'dataset/{dataset_name}.json'
with open(test_file, 'r') as f:
    test_data = json.load(f)


#test case
for index, test_case in enumerate(test_data):
    narrative_query = test_case['narrative query']
    candidate_retrieval(index, question=narrative_query,args=args)
    print(f"Test Case {index+1} processed.")
