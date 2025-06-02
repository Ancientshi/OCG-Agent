import os
import sys
import json
import pickle
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.utils import *  
from model import *
from core.QA import get_recommendations_LLM, Incremental_LLMRerank, AISearchEngine, DeepResearch, RAG
from copy import deepcopy

class JsonDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'rb') as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_data(self, index_list=[]):
        if len(index_list) == 0:
            return self.data
        else: 
            return [self.data[i] for i in index_list]

class Evaluator:
    def __init__(self, ks=[5, 10], log_dir='logs', dataset_name='default', method='default', 
                 metric_names=None):
        if metric_names is None:
            metric_names = ['precision_truth_predict', 'recall_truth_predict', 'f1_truth_predict', 
                            'ndcg_truth_predict', 'hit_truth_candidate', 'HRR', 'ILS_truth', 'ILS_predict', 'ILS_candidate',
                            'popularity_truth', 'popularity_predict', 'rating_truth', 'rating_predict']
        self.ks = ks
        self.metric_names = metric_names
        self.metrics_acc = {name: [] for name in self.metric_names}
        self.log_path = os.path.join(log_dir, dataset_name, method, 'log.jsonl')
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def evaluate_sample(self, truth, predict, candidate):
        local_truth=deepcopy(truth)
        local_predict=deepcopy(predict)
        local_candidate=deepcopy(candidate)
        for i in range(len(local_truth)):
            local_truth[i]=re.sub(r'\(.*?\)', '', local_truth[i])
            local_truth[i]=local_truth[i].replace(' ', '').lower()
        for i in range(len(local_predict)):
            local_predict[i]=re.sub(r'\(.*?\)', '', local_predict[i])
            local_predict[i]=local_predict[i].replace(' ', '').lower()
        for i in range(len(local_candidate)):
            local_candidate[i]=re.sub(r'\(.*?\)', '', local_candidate[i])
            local_candidate[i]=local_candidate[i].replace(' ', '').lower()


        local_predict = [x.replace('  ', ' ') for x in local_predict]
        seen = set()
        local_predict = [x for x in local_predict if not (x in seen or seen.add(x))]
        while len(local_predict) < len(local_truth):
            local_predict.append(' ')
         
        local_candidate = [x.replace('  ', ' ') for x in local_candidate]
        seen = set()
        local_candidate = [x for x in local_candidate if not (x in seen or seen.add(x))]
        while len(local_candidate) < len(local_truth):
            local_candidate.append(' ')
            
        
        metrics = {}
        for name in self.metric_names:
            if "precision" in name:
                if "truth" and "predict" in name:
                    metrics[name] = precision([local_truth], [local_predict], self.ks)
                elif "truth" and "candidate" in name:
                    metrics[name] = precision([local_truth], [local_candidate], self.ks)
            elif "recall" in name:
                if "truth" and "predict" in name:
                    metrics[name] = recall([local_truth], [local_predict], self.ks)
                elif "truth" and "candidate" in name:
                    metrics[name] = recall([local_truth], [local_candidate], self.ks)
            elif "f1" in name:
                if "truth" and "predict" in name:
                    metrics[name] = F1([local_truth], [local_predict], self.ks)
                elif "truth" and "candidate" in name:
                    metrics[name] = F1([local_truth], [local_candidate], self.ks)
            elif "ndcg" in name:
                if "truth" and "predict" in name:
                    metrics[name] = ndcg([local_truth], [local_predict], self.ks)
                elif "truth" and "candidate" in name:
                    metrics[name] = ndcg([local_truth], [local_candidate], self.ks)
            elif "hit" in name:
                if "truth" and "candidate" in name:
                    metrics[name] = hit([local_truth], [local_candidate], self.ks)
            elif name == "HRR":
                metrics[name] = HRR([local_truth], [local_predict], [local_candidate], self.ks)
            elif "ILS" in name:
                item_list = local_truth if "truth" in name else (local_predict if "predict" in name else local_candidate)
                metrics[name] = ILS(item_list, self.ks)
            elif "popularity" in name:
                item_list = local_truth if "truth" in name else local_predict
                metrics[name] = popularity(item_list, self.ks)
            elif "rating" in name:
                item_list = local_truth if "truth" in name else local_predict
                metrics[name] = rating(item_list, self.ks)
        
        return metrics

    def process_sample(self, truth, predict, candidate, query, extra_log=''):
        metrics = self.evaluate_sample(truth, predict, candidate)
        for name in self.metric_names:
            self.metrics_acc[name].append(metrics[name])
        
        log_entry = {
            "extra_log": extra_log,
            "query": query,
            "truth_ranking_list": truth,
            "predict_ranking_list": predict,
            "metrics": metrics
        }

        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def get_average_metrics(self):
        avg_metrics = {}
        for name, values in self.metrics_acc.items():
            avg_metrics[name] = np.mean(np.array(values), axis=0)
        return avg_metrics

    def log_overall(self, avg_metrics, result_path='exp_result/exp1/result.jsonl', args_info=''):
        overall_log = {
            "args_info": args_info,
            "evaluation": {}
        }

        for k in self.ks:
            idx = self.ks.index(k)
            overall_log["evaluation"][f"cutoff@{k}"] = {
                name: (avg_metrics[name][idx] if isinstance(avg_metrics[name], np.ndarray) else avg_metrics[name]) 
                for name in self.metric_names
            }

        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(overall_log, ensure_ascii=False) + "\n")

def process_predictions(predict_data, get_predict_ranking_fn, args, evaluator):
    for prediction_sample in tqdm(predict_data):
        query = prediction_sample['narrative query']
        profile = prediction_sample.get('profile', '')
        truth_ranking_list = prediction_sample['merged_result']
        if len(prediction_sample.get('OCG_list', [])) == 0:
            continue

        if args.rank_query == 'query':
            rank_query = query
        elif args.rank_query == 'profile':
            rank_query = profile
        elif args.rank_query == 'both':
            rank_query = query + '\n' + profile
        else:
            rank_query = query
            
        if args.GPT_Augmentation:
            gpt_predict_ranking_list = get_recommendations_LLM(rank_query, topk=args.topk, model_name=args.model_name)
            candidates_gpt = [{'Name': re.sub(r'\(.*?\)', '', x)} for x in gpt_predict_ranking_list]
            prediction_sample['OCG_list']= candidates_gpt + prediction_sample['OCG_list']
            

        predict_ranking_list = get_predict_ranking_fn(prediction_sample, rank_query, args)
        if len(predict_ranking_list) == 0: 
            print(f"Warning: prediction failed for query: {query}")
            continue
        
        candidates_list = [x['Name'] for x in prediction_sample['OCG_list']]
        
        truth_ranking_list=[re.sub(r'\(.*?\)', '', x) for x in truth_ranking_list]
        predict_ranking_list=[re.sub(r'\(.*?\)', '', x) for x in predict_ranking_list]
        candidates_list=[re.sub(r'\(.*?\)', '', x) for x in candidates_list]
        
        predict_ranking_list = align(truth_ranking_list, predict_ranking_list)
        candidates_list = align(truth_ranking_list, candidates_list)
        
        log_info = dict(vars(args))
        evaluator.process_sample(truth_ranking_list, predict_ranking_list, candidates_list, query, extra_log=log_info)

def get_ranking_model_based(prediction_sample, rank_query, args, model=None):
    candidate_documents_list = []
    candidate_index_list = []
    OCG_list = prediction_sample['OCG_list']
    for index, candidate in enumerate(OCG_list):
        candidate_index_list.append(index)
        if args.Knowledge_Augmentation:
            candidate_document = dict_to_str(candidate)
        else:
            candidate_document = candidate['Name']
        candidate_documents_list.append(candidate_document)

    if args.method =='LLMRerank':
        scores = model.predict(rank_query, candidate_documents_list, windows_size=args.topk*2,step=args.topk,model_name=args.model_name)
    else:
        scores = model.predict(rank_query, candidate_documents_list)
    if isinstance(scores, np.float64):
        scores = [scores]
    sorted_indices = [x for _, x in sorted(zip(scores, candidate_index_list), reverse=True)]
    return [OCG_list[i]['Name'] for i in sorted_indices]

def get_ranking_baseline(prediction_sample, rank_query, args):
    if args.method == 'LLM':
        return get_recommendations_LLM(rank_query, topk=args.topk, model_name=args.model_name)
    elif args.method == 'AISearch':
        return AISearchEngine(rank_query, topk=args.topk, engine = args.model_name)
    elif args.method == 'DeepResearch':
        return DeepResearch(rank_query, topk=args.topk, engine = args.model_name)
    elif args.method == 'Incremental_LLMRerank':
        candidates = prediction_sample['OCG_list']
        return Incremental_LLMRerank(rank_query, candidates, window_size=2*args.topk, topk=args.topk)
    elif args.method == 'RAG':
        return RAG(rank_query, topk=args.topk, model_name = args.model_name)
    else:
        print("Not defined method")
        sys.exit(0)
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='filtered_movie_sub296_len40_intersect')
    parser.add_argument('--method', type=str, default='LLMRerank', help='LLM, AISearch, DeepResearch for baseline; BM25REC, BGE, EasyRec, LLMRerank for OCG. RAG for paradigm baseline')
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini')
    parser.add_argument('--Knowledge_Augmentation', type=bool, default=True)
    parser.add_argument('--GPT_Augmentation', type=bool, default=True)
    parser.add_argument('--rank_query', type=str, default='query')
    parser.add_argument('--topk', type=int, default=5)
     
    args = parser.parse_args()

    dataset_name = args.dataset_name
    method = args.method
    rank_query = args.rank_query

    json_path = f'dataset/{dataset_name}.json'
    if not os.path.exists(json_path):
        print('json file not exist')
        sys.exit(0)

    dataset = JsonDataset(json_path)
    if args.dataset_name == 'edu_sub30':
        predict_index_list = [0, 1, 2, 4, 6, 7, 8, 13, 14, 15, 17]
    else:
        predict_index_list = []  
    predict_data = dataset.get_data(predict_index_list)

    metric_names_based=['precision_truth_predict', 'recall_truth_predict', 'f1_truth_predict', 'ndcg_truth_predict', 'hit_truth_candidate', 'HRR', 'ILS_truth', 'ILS_predict', 'ILS_candidate', 'popularity_truth', 'popularity_predict', 'rating_truth', 'rating_predict']
    metric_names_baseline=['precision_truth_predict', 'recall_truth_predict', 'f1_truth_predict', 'ndcg_truth_predict', 'ILS_truth', 'ILS_predict','popularity_truth', 'popularity_predict', 'rating_truth', 'rating_predict']
    args_info = dict(vars(args))

    if args.topk == 5:
        ks=[5]
    elif args.topk == 10:
        ks=[5, 10]    
    if method in ['BM25REC', 'BGE', 'EasyRec', 'LLMRerank']:
        model_class = {'BM25REC': BM25REC, 'BGE': BGE_Reranker, 'EasyRec': EasyRec, 'LLMRerank': LLMRerank}
        model = model_class[method]()
        
        evaluator = Evaluator(ks=ks, log_dir='logs', dataset_name=dataset_name, method=method, metric_names=metric_names_based)
    
        def get_predict_ranking(sample, rq, args_inner):
            return get_ranking_model_based(sample, rq, args_inner, model=model)
    elif method in ['LLM', 'AISearch', 'DeepResearch', 'Incremental_LLMRerank','RAG']:
        evaluator = Evaluator(ks=ks, log_dir='logs', dataset_name=dataset_name, method=method, metric_names= metric_names_baseline)
        
        def get_predict_ranking(sample, rq, args_inner):
            return get_ranking_baseline(sample, rq, args_inner)
    else:
        print("Not defined method")
        sys.exit(0)

    process_predictions(predict_data, get_predict_ranking, args, evaluator)
    avg_metrics = evaluator.get_average_metrics()
    evaluator.log_overall(avg_metrics, result_path=f'exp_result/exp1/{args.dataset_name}/result.jsonl', args_info=args_info)

if __name__ == '__main__':
    main()
