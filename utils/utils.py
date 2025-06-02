import copy
from copy import deepcopy
import openai
import requests
import os
import sys
import json
import time
import random
import logging
import re
from transformers import LongformerTokenizer
from rank_bm25 import BM25Okapi
import numpy as np
import torch
import html
import string
import math
import hashlib
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch



def GPT_QA_not_stream(prompt, model_name="gpt-4o-mini", t=0.0, historical_qa=None):
    url = "https://api.openai.com/v1/chat/completions"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    if len(prompt)>120000*3:
        prompt=prompt[:120000*3]
    
    
    messages = []
    if historical_qa is not None:
        messages = historical_qa   
    messages.append({"role": "user", "content": prompt})
    
    if model_name=='gpt-4o-mini':
        max_tokens=16384
    else:
        max_tokens=4096
        
    client = openai.OpenAI()
    

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=t,
            n=1,
            service_tier='default',
            max_tokens=max_tokens,
            stream=False
        )
    except Exception as e:
        print(f'Error: {str(e)}, Currently Unavailable.')
        return f'Error: {str(e)}, Currently Unavailable.'
    
    return response.choices[0].message.content



def SiliconFlow_QA_not_stream(prompt,model_name="Pro/deepseek-ai/DeepSeek-R1", t=0.0,historical_qa=None):
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    api_key=os.environ["SILICONFLOW_API_KEY"]
    if len(prompt)>64000*3:
        prompt=prompt[:64000*3]
    
    messages = []
    if historical_qa is not None:
        messages = historical_qa
    messages.append({"role": "user", "content": prompt})
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {
        "model": model_name,
        "messages": messages,
        "n": 1,
        "stream": False,
        "max_tokens": 4096,
        "temperature": t,
        "top_p": 0.8,
        "top_k": 50
    }
                
    response = requests.post(url, json=data, headers=headers, stream=False)
    if response.status_code == 200:
        dict_chunk = json.loads(response.text) 
        content=dict_chunk['choices'][0]['message']['content']
        return content
    else:
        return f"Error from SiliconFlow API: {response.text}"
        
    
def GPT_QA_reasoning_no_stream(prompt, model_name="o3-mini", t=0.0, historical_qa=None):
    url = "https://api.openai.com/v1/chat/completions"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
    if len(prompt)>120000*3:
        prompt=prompt[:120000*3]
    
    
    messages = []
    if historical_qa is not None:
        messages = historical_qa   
    messages.append({"role": "user", "content": prompt})
    
    max_completion_tokens=4096
        
    client = openai.OpenAI()
    

    try:
        response = client.chat.completions.create(
            model=model_name,
            reasoning_effort="medium",
            messages=messages,
            n=1,
            service_tier='default',
            max_completion_tokens=max_completion_tokens,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f'Error: {str(e)}, Currently Unavailable.')
        return f'Error: {str(e)}, Currently Unavailable.'
    

 
def perplexity(prompt=""):
    """
    Interact with OpenAI's chat completion API.

    Parameters:
        api_key (str): Your OpenAI API key.
        base_url (str): The base URL for the OpenAI API.
        model (str): The model to use for chat completions.
        system_message (str): The system's instruction or context for the assistant.
        user_message (str): The user's input message.
        stream (bool): Whether to stream the response or not. Default is False.

    Returns:
        If stream=False: str, the full response content.
        If stream=True: tuple, the concatenated content and a set of citations.
    """
    model="sonar"
    base_url="https://api.perplexity.ai"
    system_message="You need to help to solve user\'s information search question."
    
    # Initialize the client
    client = openai.OpenAI(api_key=os.environ['PERPLEXITY_API_KEY'], base_url=base_url)

    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=0.0
    )
    response_dict = response.to_dict()
    return response_dict["choices"][0]["message"]["content"], response_dict.get("citations", [])

def gpt_search(prompt):
    client = openai.OpenAI()

    response = client.responses.create(
        model="gpt-4o-mini",
        tools=[{"type": "web_search_preview"}],
        input=prompt
    )
    output_text=response.output_text

    return output_text
    
def gemini_search(prompt):
    client = genai.Client(api_key=Genini_api_key)
    model_id = "gemini-2.0-flash"

    google_search_tool = Tool(
        google_search = GoogleSearch()
    )

    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
    )
    all_text=''
    for each in response.candidates[0].content.parts:
        if each.text:
            all_text+=each.text
    return all_text



    


# def tiangong(prompt):
#     # -*- coding: utf-8 -*-
#     API_HOST='api.singularity-ai.com'
#     url = f'https://{API_HOST}/sky-saas-search/api/v1/search'
#     app_key = tiangong_app_key        
#     app_secret = tiangong_app_secret  
#     timestamp = str(int(time.time()))
#     sign_content = app_key + app_secret + timestamp
#     sign_result = hashlib.md5(sign_content.encode('utf-8')).hexdigest()


#     headers={
#         "app_key": app_key,
#         "timestamp": timestamp,
#         "sign": sign_result,
#         "Content-Type": "application/json",
#     }

#     data = {
#         "content": prompt,
#         "stream_resp_type": "all" 
#     }

#     response = requests.post(url, headers=headers, json=data, stream=False)
#     for line in response.iter_lines():
#         if line:
#             return line.decode('utf-8')        
        

def perplexity_deepresearch(prompt):
    if 'I want to study further.' in prompt:
        path='dataset/edu_pplx_deepresearch.json'
    else:
        path='dataset/pplx_deepresearch.json'
    with open(path, 'r') as f:
        data = json.load(f)
    query_content_dict={}
    for d in data:
        query_content_dict[d['narrative query']]=d['generated_content']
    
    return query_content_dict[prompt]

def open_deepresearch(prompt):
    if 'I want to study further.' in prompt:
        path='dataset/edu_open_deepresearch.json'
    else:
        path='dataset/open_deepresearch.json'
    with open(path, 'r') as f:
        data = json.load(f)
    query_content_dict={}
    for d in data:
        query_content_dict[d['narrative query']]=d['generated_content']
    return query_content_dict[prompt]

def RAG_paradigm(prompt):
    if 'I want to study further.' in prompt:
        path='dataset/edu_RAG_result.json'
    else:
        path='dataset/RAG_result.json'
    with open(path, 'r') as f:
        data = json.load(f)
    query_content_dict={}
    for d in data:
        query_content_dict[d['narrative query']]=d['RAG_result']
    
    return query_content_dict[prompt]
    
    
def filter_content_bm25(page_content, query, threshold=3,maxlength = 10000):
    if threshold==0:
        return page_content
    
    toker = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    
    sentences = page_content.split("\n\n")
    sentences = [sentence for sentence in sentences if sentence.strip()]
    
    tokenized_sentences = [toker.encode(sentence.lower()) for sentence in sentences]
    tokenized_query = toker.encode(query.lower())

    bm25 = BM25Okapi(tokenized_sentences)

    scores = bm25.get_scores(tokenized_query)



    percentile_score = np.percentile(scores, 80)
    threshold=max(percentile_score,threshold)
    relevant_sentences = [sentences[i] for i in range(len(sentences)) if scores[i] >= threshold and len(sentences[i])>5]
    
    page_content = '\n\n'.join(relevant_sentences)
    
    words_list=page_content.split(' ')
    words_count = len(words_list)
    maxlength=min(maxlength-300,words_count)
    page_content = ' '.join(words_list[:maxlength])


    return page_content


def FrontWrapper_Header(text_content,type='mindmap'):
    if type=='mindmap':
        json_data = {
            "header": {"content": text_content, "type": "mindmap"},
            "content": None,
            "reasoningContent": None,
            "additionalContent": None,
            "note": None
        }
    else:
        pass
    return json_data
    



def update(old_item, new_item):
    '''
    Merge two dictionaries: old_item and new_item.
    For each key present in both dictionaries, if both values are strings and neither equals "NOT FOUND", concatenate them; if one value is "NOT FOUND", keep the other; if both are "NOT FOUND", the result remains "NOT FOUND".
    '''
    merged_item={}
    old_item_keys=list(old_item.keys())
    new_item_keys=list(new_item.keys())
    merged_item_keys=list(set(old_item_keys).union(set(new_item_keys)))
    for key in merged_item_keys:
        old_value=old_item.get(key, 'NOT FOUND')
        new_value=new_item.get(key, 'NOT FOUND')
        
        if old_value == 'NOT FOUND' and new_value == 'NOT FOUND':
            merged_item[key] = 'NOT FOUND'
        elif old_value == 'NOT FOUND':
            merged_item[key] = new_value
        elif new_value == 'NOT FOUND':
            merged_item[key] = old_value
        else:
            if isinstance(old_value, str) and isinstance(new_value, str):
                if old_value.isdigit() and new_value.isdigit() and old_value == new_value:
                    merged_item[key] = new_value
                elif old_value.isdigit() and new_value.isdigit() and old_value != new_value:
                    max_value = max(int(old_value), int(new_value))
                    merged_item[key] = str(max_value)
                elif isinstance(old_value, str) and isinstance(new_value, str):
                    merged_item[key] = old_value + ' ' + new_value
                 
            elif isinstance(old_value, list) and isinstance(new_value, list):
                try:
                    merged_item[key] = list(set(old_value).union(set(new_value)))
                except:
                    merged_item[key] = old_value + new_value
            elif isinstance(old_value, dict) and isinstance(new_value, dict):
                merged_item[key] = {**old_value, **new_value}
            else:
                merged_item[key] = new_value
    return merged_item
                
        


def candidate_items_list_merge(candidate_items_list_global):
    '''
    candidate_items_list is a list of lists; 
    each inner list contains dictionaries, and each dictionary holds the content of a document.
    '''
    candidate_items_dict={}
    for candidate_list_local in candidate_items_list_global:
        if isinstance(candidate_list_local, list):
            for candidate_item in candidate_list_local:
                name=candidate_item.get('Name', 'NOT FOUND')
                if name == 'NOT FOUND':
                    continue
                

                if name in list(candidate_items_dict.keys()):
                    print(f"Updating existing candidate item: {name}")
                    new_Additional_Information=candidate_item.get('AdditionalInformation', {})
                    original_Additional_Information=candidate_items_dict[name].get('AdditionalInformation', {})
                    
                    if new_Additional_Information=='NOT FOUND' or new_Additional_Information=={}:
                        new_Additional_Information= {}
                    elif type(new_Additional_Information) is str:
                        new_Additional_Information= {'TextInfo':new_Additional_Information}
                    elif type(new_Additional_Information) is list:
                        new_Additional_Information= {'ListInfo':new_Additional_Information}
                    else:
                        pass
                    
                    if original_Additional_Information=='NOT FOUND' or original_Additional_Information=={}:
                        original_Additional_Information= {}
                    elif type(original_Additional_Information) is str:
                        original_Additional_Information= {'TextInfo':original_Additional_Information}
                    elif type(original_Additional_Information) is list:
                        original_Additional_Information= {'ListInfo':original_Additional_Information}
                    else:
                        pass
                    
                    merged_Additional_Information= {**original_Additional_Information, **new_Additional_Information}
                    
                    old_item=deepcopy(candidate_items_dict[name])
                    new_item=deepcopy(candidate_item)
                    merge_item=update(old_item, new_item)
                    merge_item['AdditionalInformation']=merged_Additional_Information
                    candidate_items_dict[name]=merge_item
                else:
                    print(f"Adding new candidate item: {name}")
                    candidate_items_dict[name]=candidate_item
        else:
            pass

    candidate_items_list=list(candidate_items_dict.values())
    return candidate_items_list


def content_post_process(content):
    content = html.unescape(content)
    content = re.sub(r'<!-- image -->', '', content)
    content = re.sub(r'(&nbsp;|&amp;)', '', content)
    content = re.sub(r'\n\s*\n', '\n', content)
    return content.strip()


def match(name1, name2):
    """
    Calculate the Jaccard similarity and determine whether the two names match
    """
    name1 = re.sub(r"\(\d+\)", "", name1).strip()
    name2 = re.sub(r"\(\d+\)", "", name2).strip()
    
    name1 = re.sub(r'<[^>]+>', '', name1)
    name2 = re.sub(r'<[^>]+>', '', name2)

    set_name1 = set(name1.split())
    set_name1 = set([word.strip(string.punctuation) for word in set_name1])

    set_name2 = set(name2.split())
    set_name2 = set([word.strip(string.punctuation) for word in set_name2])

    intersection = set_name1.intersection(set_name2)
    union = set_name1.union(set_name2)
    
    Jaccard = len(intersection) / len(union) if union else 0  
    return Jaccard >= 0.6  

def align(test_truth_list, predict_ranking_list):
    aligned_predict_ranking_list = []
    for candidate in predict_ranking_list:
        matched = False
        for truth in test_truth_list:
            if match(candidate, truth):
                aligned_predict_ranking_list.append(truth)
                matched = True
                break
        if not matched:
            aligned_predict_ranking_list.append(candidate)
    return aligned_predict_ranking_list


def ndcg(test_truth_list, test_prediction_list, topk):
    ndcgs = []
    for k in topk:
        ndcg_list = []
        
        for ind, test_truth in enumerate(test_truth_list):
            dcg = 0
            idcg = 0
            test_truth_index = set(test_truth)
            
            if len(test_truth_index) == 0:
                continue
            
            top_sorted_index = test_prediction_list[ind][0:k]
            
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    dcg += 1.0 / np.log2(index + 2)  

            sorted_truth_index = list(test_truth)[:k]
            for ideal_index in range(min(len(sorted_truth_index), k)):
                idcg += 1.0 / np.log2(ideal_index + 2)

            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
                
            ndcg_list.append(ndcg)
        
        if len(ndcg_list) > 0:
            ndcgs.append(np.mean(ndcg_list))
        else:
            ndcgs.append(0.0)
    
    return ndcgs


def hit(test_truth_list, test_prediction_list, topk):
    hits = []
    for k in topk:
        hit_list = []
        for truth, prediction in zip(test_truth_list, test_prediction_list):
            truncated_truth = truth[:k]
            
            if not truncated_truth:
                continue
            
            hit_count = sum(1 for item in truncated_truth if item in prediction)
            
            hit_ratio = hit_count / len(truncated_truth)
            hit_list.append(hit_ratio)
        
        if hit_list:
            hits.append(np.mean(hit_list))
        else:
            hits.append(0.0)
    
    return hits


def mrr(test_truth_list, test_prediction_list, topk):
    mrrs = []
    for k in topk:
        mrr_list = []
        
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_set = set(test_truth)
            
            if len(test_truth_set) == 0:
                continue
            
            top_sorted_index = test_prediction_list[ind][0:k]
            
            rr = 0
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_set:
                    rr = 1.0 / (index + 1)  
                    break  
            
            mrr_list.append(rr)
        
        if len(mrr_list) > 0:
            mrrs.append(np.mean(mrr_list))
        else:
            mrrs.append(0.0)
    
    return mrrs
    
def recall(test_truth_list, test_prediction_list, topk=[20]):
    recalls = []
    for k in topk:
        recall_list = []
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            recall_dem = len(test_truth_index)
            top_sorted_index = set(test_prediction_list[ind][0:k])
            hit_num = len(top_sorted_index.intersection(test_truth_index))
            recall_list.append(hit_num * 1.0 / (recall_dem + 1e-20))
        recall = np.mean(recall_list)
        recalls.append(recall)
    return recalls

def precision(test_truth_list, test_prediction_list, topk=[20]):
    precisions = []
    for k in topk:
        precision_list = []
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            top_sorted_index = set(test_prediction_list[ind][0:k])
            hit_num = len(top_sorted_index.intersection(test_truth_index))
            precision_list.append(hit_num * 1.0 / (k + 1e-20))
        precision = np.mean(precision_list)
        precisions.append(precision)
    return precisions


def F1(test_truth_list, test_prediction_list, topk=[20]):
    f1_scores = []
    for k in topk:
        precision_scores = []
        recall_scores = []
        
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                precision_scores.append(0)
                recall_scores.append(0)
                continue

            top_sorted_index = set(test_prediction_list[ind][:k])  
            hit_num = len(top_sorted_index.intersection(test_truth_index))

            precision = hit_num / (k + 1e-20)
            recall = hit_num / (len(test_truth_index) + 1e-20)

            precision_scores.append(precision)
            recall_scores.append(recall)
        
        f1_list = []
        for p, r in zip(precision_scores, recall_scores):
            if p + r == 0:
                f1_list.append(0)
            else:
                f1_list.append(2 * p * r / (p + r))
        
        f1_score = np.mean(f1_list)
        f1_scores.append(f1_score)
    
    return f1_scores

def HRR(test_truth_list, test_prediction_list, candidate_items_list, topk=[20]):
    hrrs = []
    for k in topk:
        hrr_list = []
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_index = set(test_truth[:k])
            if len(test_truth_index) == 0:
                continue
            top_sorted_index = set(test_prediction_list[ind][:k])
            candidate_index = set(candidate_items_list[ind])
            hit_num = len(top_sorted_index.intersection(test_truth_index))
            retrieved_num = len(candidate_index.intersection(test_truth_index))
            hrr = hit_num / (retrieved_num + 1e-20)
            hrr_list.append(hrr)
        hrr = np.mean(hrr_list)
        hrrs.append(hrr)
    return hrrs
    

import re
def dict_to_str(data, level=0):
    if not isinstance(data, dict):
        return str(data)  
    
    result = ""
    tab = "\t" * level  
    
    for key, value in data.items():
        if isinstance(value, dict):
            result += f"{tab}{key}:\n{dict_to_str(value, level + 1)}"
        else:
            value = re.sub(r'<[^>]+>', '', str(value))
            result += f"{tab}{key}: {value}\n"
    
    return result

def get_embeddings(documents):
    response = requests.post(f"http://localhost:8502/get_embedding", json={"documents": documents})
    embeddings = response.json()['embeddings']
    return embeddings


def similarity_matrix(rec_list):
    embeddings = np.array(get_embeddings(rec_list))
    similarity_matrix = np.dot(embeddings, embeddings.T)
    return similarity_matrix
    

def ILS(rec_list, topk=[20]):
    total_similarity_list=[]
    for k in topk:
        topk_rec_list = rec_list[:k]  
        sim_matrix = similarity_matrix(topk_rec_list)
        
        indices = np.triu_indices(len(topk_rec_list), k=1)  
        total_similarity = np.mean(sim_matrix[indices]) if len(indices[0]) > 0 else 0
        total_similarity_list.append(total_similarity)
    return total_similarity_list


def calculate_log_popularity_difference(user_history, recommended_list):
    user_logs = [math.log(phi) for phi in user_history]
    avg_user = sum(user_logs) / len(user_logs)
    
    rec_logs = [math.log(phi) for phi in recommended_list]
    avg_rec = sum(rec_logs) / len(rec_logs)
    
    return avg_rec - avg_user


def tmdb_search_movie(query):
    url = f"https://api.themoviedb.org/3/search/movie?query={query}&include_adult=true&language=en-US&page=1"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {tmdb_key}"
    }

    response = requests.get(url, headers=headers)
    reponse_json = response.json()
    if len(reponse_json['results'])==0:
        return {'popularity':-1,'vote_average':-1,'vote_count':-1}
    else:
        popularity=reponse_json['results'][0]['popularity']
        vote_average=reponse_json['results'][0]['vote_average']
        vote_count=reponse_json['results'][0]['vote_count']
        return {'popularity':popularity,'vote_average':vote_average,'vote_count':vote_count}


def popularity(rec_list, topk=[20]):
    return [0]*len(topk)
    # popularity_list=[]
    # for candidate in rec_list:
    #     candidate=re.sub(r"\(\d+\)", "", candidate).strip()
    #     movie_json = tmdb_search_movie(candidate)
    #     popularity=movie_json['popularity']
    #     popularity_list.append(popularity)
    
    # total_popularity_list=[]
    # for k in topk:
    #     count_invalid=len([i for i in popularity_list[:k] if i==-1])
    #     count_valid=k-count_invalid
    #     sum_valid_popularity=sum([i for i in popularity_list[:k] if i!=-1])
    #     avg_popularity =  sum_valid_popularity / count_valid if count_valid > 0 else 0
    #     total_popularity_list.append(avg_popularity)
    # return total_popularity_list


def rating(rec_list, topk=[20]):
    return [0]*len(topk)
    # rating_list=[]
    # for candidate in rec_list:
    #     candidate=re.sub(r"\(\d+\)", "", candidate).strip()
    #     movie_json = tmdb_search_movie(candidate)
    #     vote_average=movie_json['vote_average']
    #     rating_list.append(vote_average)
    
    # total_rating_list=[]
    # for k in topk:
    #     count_invalid=len([i for i in rating_list[:k] if i==-1])
    #     count_valid=k-count_invalid
    #     sum_valid_rating=sum([i for i in rating_list[:k] if i!=-1])
    #     avg_rating =  sum_valid_rating / count_valid if count_valid > 0 else 0
    #     total_rating_list.append(avg_rating)
    # return total_rating_list

def get_year(text):
    match = re.search(r"\(([^)]+)\)", text)
    if match:
        inside_brackets = match.group(1)
        year_str = re.sub(r"\D", "", inside_brackets)
        if year_str:
            year = int(year_str.strip())
            revised_text = re.sub(r"\([^)]+\)", f"({year})", text)
            return year, revised_text
    return 2000, text