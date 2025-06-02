from utils.utils import *
from core.prompt import *
from tool.AISearch import self_AI_search

def extract_json_from_text(text):
    """Extract JSON from text that might contain non-JSON content"""
    if not text:
        return None
    
    # Try to find JSON object in the text using regex
    json_match = re.search(r'({.*})', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # If regex didn't work, try to clean up the text
    try:
        # Remove any non-JSON content before the first {
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx >= 0 and end_idx > start_idx:
            clean_json = text[start_idx:end_idx+1]
            return json.loads(clean_json)
    except:
        pass
    
    return None






def process_recommendations(recommendations_text, topk=20):
    """Process recommendation text and ensure it's valid JSON format"""
    recommendations_text=recommendations_text.replace('```json','').replace('```','').strip()
    try:
        # First try to parse the text directly as JSON
        try:
            data = json.loads(recommendations_text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the text
            data = extract_json_from_text(recommendations_text)
            if not data:
                print(f"Error: Unable to parse JSON. Raw response: \n{recommendations_text}")
                return None
        
        # Check if it contains the ranked_result key
        if 'ranked_result' not in data:
            print("Error: Recommendation result missing 'ranked_result' key")
            # Try to create a compatible structure if we can identify a list of items
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    data = {"ranked_result": value}
                    print(f"Found alternative list under key '{key}', using it as ranked_result")
                    break
            else:
                return None
        
        # Check if ranked_result is a list and contains items
        ranked_results = data['ranked_result']
        if not isinstance(ranked_results, list):
            print("Error: 'ranked_result' is not a list")
            return None
        
        if len(ranked_results) == 0:
            print("Error: No items in the recommendation list")
            return None
        
        if len(ranked_results) != topk:
            print(f"Warning: Recommendation result contains {len(ranked_results)} items instead of {topk}")
        
        return data
    except json.JSONDecodeError as e:
        print(f"Error: Unable to parse JSON: {e}")
        return None
    except Exception as e:
        print(f"Error processing recommendations: {e}")
        return None
        

def rewrite(question='',profile=''):
    rewrite_prompt= REWRITE_TEMPLATE.replace("{{query}}", question).replace("{{profile}}", profile)
    response=GPT_QA_not_stream(rewrite_prompt, model_name="gpt-4o-mini", t=0.0)
        
    subquestion_list=[]
    for line in response.split('\n'):
        line=line.strip()
        if line:
            subquestion_list.append(line)
    return subquestion_list

def generate_single_query(in_context_situation):
    single_query_generation_prompt=SINGLE_QUERY_GENERATION_TEMPLATE.replace("{{in_context_situation}}", in_context_situation)
    response=GPT_QA_not_stream(single_query_generation_prompt, model_name="gpt-4o-mini", t=0.0)
    generated_query=response.split('```json')[1].split('```')[0].strip()
    generated_query=json.loads(generated_query)
    return generated_query['query']


def SpanPredict(adt='', article=''):
    SpanPredict_prompt= SPANPREDICT_TEMPLATE_v2_shn.replace("{{ADT}}", adt).replace("{{article}}", article)
    spanned_content=GPT_QA_not_stream(SpanPredict_prompt, model_name="gpt-4o-mini", t=0.0)
    # shn：这里由于提示词变化，因此split('Reformatted Article')[1] 暂时改为split('Reformatted Article')[0]
    spanned_content=spanned_content.split('Reformatted Article')[1].replace('```xml', '\n').replace('```', '\n').strip()
    return spanned_content
    


def ColdRead(query=''):
    ColdRead_prompt= COLDREAD_TEMPLATE.replace("{{query}}", query)
    profile_content=GPT_QA_not_stream(ColdRead_prompt, model_name="gpt-4o-mini", t=0.2)
    return profile_content


def ADT_generation(question=''):
    ADT_prompt= ADT_TEMPLATE.replace("{{query}}", question)
    adt=GPT_QA_not_stream(ADT_prompt, model_name="gpt-4o-mini", t=0.0)
    adt=adt.split('```json')[1].split('```')[0].strip()
    return adt

def Extract(article='', ADT=''):
    extract_prompt = EXTRACT_TEMPLATE.replace("{{article}}", article).replace("{{ADT}}", ADT)
    
    for _ in range(2):  # 尝试两次
        try:
            response = GPT_QA_not_stream(extract_prompt, model_name="gpt-4o-mini", t=0.0)
            candidate_items_str = response.split('```json')[1].split('```')[0].strip()
            candidate_items_list = json.loads(candidate_items_str)
            
            if not isinstance(candidate_items_list, list):
                return []
            
            candidate_items_list_valid = [
                item for item in candidate_items_list 
                if isinstance(item, dict) and item.get('Name') != 'NOT FOUND'
            ]
            
            return candidate_items_list_valid
        except Exception:
            continue  # 如果异常，继续下一次尝试
    
    return []  # 如果两次都失败，返回空列表


def complete(candidate_item='',article='',ADT=''):
    candidate_item_str=json.dumps(candidate_item)
    complete_prompt= COMPLETE_TEMPLATE.replace("{{candidate_item}}", candidate_item_str).replace("{{article}}", article).replace("{{ADT}}", ADT)
    response=GPT_QA_not_stream(complete_prompt, model_name="gpt-4o-mini", t=0.0)
    completed_candidate_str=response.split('```json')[1].split('```')[0].strip()
    completed_candidate=json.loads(completed_candidate_str)
    return completed_candidate


def Incremental_LLMRerank(query, candidates, window_size=20, topk=10):
    '''
    query: str, the query
    candidates: list of dict
    window_size: sliding window size
    topk: int, the number of candidates to return
    '''
    tmp_dict={}
    for index,c in enumerate(candidates):
        #给c加一个key为ID
        c['ID']= f'C{index}'
        tmp_dict[f'C{index}']= c
        
    rank_result=''
    
    chunked_candidates_list= [candidates[i:i+window_size] for i in range(0, len(candidates), window_size)]
    for chunked_candidates in chunked_candidates_list:
        candidate_list_str=''
        for c in chunked_candidates:
            candidate_list_str+=dict_to_str(c)+'\n'
        
        prompt=LLMRERANK.replace("{{query}}", query).replace("{{candidate_list}}", candidate_list_str).replace("{{topk}}", str(topk)).replace("{{rank_result}}", rank_result)
        response=GPT_QA_not_stream(prompt, model_name="gpt-4o-mini", t=0.0)
        print(response)
        rank_result=response.split('```json')[1].split('```')[0].strip()
        rank_result_list=json.loads(rank_result)
        print(rank_result_list)
        
        for ID in rank_result_list:
            try:
                rank_result+=dict_to_str(tmp_dict[ID])+'\n'
            except:
                print(f"Error: {ID} not found in candidates")
                pass
    rank_result_name_list=[]
    for ID in rank_result_list:
        rank_result_name_list.append(tmp_dict[ID]['Name'])
    return rank_result_name_list


def generative_retrieval(narrative_query, adt,topk=20, model_name='gpt-4o-mini'):
    """Get item recommendations using OpenAI API"""
    if 'I want to study further' in narrative_query:
        edu_dataset=True
    else:
        edu_dataset=False
    if not edu_dataset:
        year= re.findall(r'\((\d{4})\)', narrative_query)
        year = max(map(int, year)) if year else 2000
        prompt = GENERATIVE_RETRIEVAL_TEMPLATE.replace("{{narrative_query}}", narrative_query).replace("{{topk}}", str(topk)).replace("{{adt}}", adt).replace("{{year}}", str(year))
    else:
        prompt = GENERATIVE_RETRIEVAL_TEMPLATE_EDU.replace("{{narrative_query}}", narrative_query).replace("{{topk}}", str(topk)).replace("{{adt}}", adt)
    if model_name in ['gpt-4o-mini', 'gpt-4o']:
        response = GPT_QA_not_stream(prompt, model_name=model_name, t=0.0)
    else:
        response= SiliconFlow_QA_not_stream(prompt, model_name=model_name, t=0.0)
    response=response.split('```json')[1].split('```')[0].strip()
    ranked_result=json.loads(response)['ranked_result']
    return ranked_result
    
        

def get_recommendations_LLM(narrative_query,topk=20,model_name="gpt-4o-mini"):
    """Get item recommendations using OpenAI API"""
    if 'I want to study further' in narrative_query:
        edu_dataset=True
    else:
        edu_dataset=False
    if not edu_dataset:
        prompt = DIRECT_TEMPLATE.replace("{{narrative_query}}", narrative_query).replace("{{topk}}", str(topk))
    else:
        prompt = DIRECT_TEMPLATE_EDU.replace("{{narrative_query}}", narrative_query).replace("{{topk}}", str(topk))
        
    if model_name in ['gpt-4o-mini', 'gpt-4o']:
        response = GPT_QA_not_stream(prompt, model_name=model_name, t=0.0)
    else:
        response= SiliconFlow_QA_not_stream(prompt, model_name=model_name, t=0.0)
    
    processed_data = process_recommendations(response,topk)
    ranked_result=processed_data.get("ranked_result", [])
    return ranked_result
    

def answer_question(question="",historical_qa=None,external_knowledge_str=None,topk=20,model_name='gpt-4o-mini'):
    assert model_name in ['gpt-4o-mini','Pro/deepseek-ai/DeepSeek-R1']
    year_matches = re.findall(r'\((\d{4})\)', question)
    year = max(map(int, year_matches)) if year_matches else 2000
    prompt = ANSWER_QUESTION_TEMPLATE.replace("{{query}}", question).replace("{{knowledge}}", external_knowledge_str).replace("{{topk}}", str(topk)).replace("{{year}}", str(year))
    if model_name in ['gpt-4o-mini', 'gpt-4o']:
        response = GPT_QA_not_stream(prompt, model_name=model_name, t=0.0)
    else:
        response= SiliconFlow_QA_not_stream(prompt, model_name=model_name, t=0.0)
    return response

def extract_mentioned_items(article):
    EXTRACT_ITEM_TEMPLATE='''
    ### Task
    You need to extract the recommended item names mentioned in the [Article].
    
    ### Article
    {{article}}
    
    ### Important Note
    1. Do not include any other information, just the recommended item names.
    2. Do not farbricate any information.
    3. If there are no item names mentioned in the article, please return "No recommended item names found".
    4. Maintain the items in the same order as they appear in the article.
    
    ### Output Format
    Please return the item names split by commas.
    '''
    prompt = EXTRACT_ITEM_TEMPLATE.replace("{{article}}", article)
    response = GPT_QA_not_stream(prompt, model_name="gpt-4o-mini", t=0.0)
    return response

def manual_search(narrative_query,knowledge_augmentation='False',enhance_retrieval='False'):
    def get_ground_truth(query):
        with open('dataset/filtered_movie_sub296.json', 'r') as f:
            data = json.load(f)
        for d in data:
            if d['narrative query'] == query:
                return d['merged_result']
        
    external_knowledge_dict,citation_list=self_AI_search(query=narrative_query,pagenum=3,threshold=0,existed_citation_list=[])
    
    external_knowledge_str=''''''
    index=1
    for key, value in external_knowledge_dict.items():
        external_knowledge_str+= f"{index}. {key}:\n{value}\n"
            
    if knowledge_augmentation=='True':
        #保留搜索到的内容即可
        pass
    elif knowledge_augmentation=='False':
        #要提取出来搜索到的内容，只保留item名称
        external_knowledge_str=extract_mentioned_items(external_knowledge_str)
    else:
        raise ValueError("knowledge_augmentation must be 'True' or 'False'")
    if enhance_retrieval=='True':
        #要用true label 来增强召回
        enc_knowledge=get_ground_truth(narrative_query)
        enc_knowledge_str=', '.join(enc_knowledge)
        external_knowledge_str+=f"There are also the following candidate item resources to consider：{enc_knowledge_str}\n"
    elif enhance_retrieval=='False':
        #无需GPT生成
        pass
    else:
        raise ValueError("enhance_retrieval must be 'True' or 'False'")

    # print('narrative_query:', narrative_query)
    # print('external_knowledge_str:', external_knowledge_str)
    # aa=input('pause')
    answer=answer_question(question=narrative_query,historical_qa=None,external_knowledge_str=external_knowledge_str,topk=20,model_name='gpt-4o-mini')
    return answer
    
def AISearchEngine(narrative_query, topk=20, engine='Perplexity'):
    if 'I want to study further' in narrative_query:
        edu_dataset=True
    else:
        edu_dataset=False
    if not edu_dataset:
        year_matches = re.findall(r'\((\d{4})\)', narrative_query)
        year = max(map(int, year_matches)) if year_matches else 2000
    attempts = 0
    max_attempts = 2

    if not edu_dataset:
        prompt = AISEARCHENGINE_TEMPLATE.replace("{{narrative_query}}", narrative_query).replace("{{topk}}", str(topk)).replace("{{year}}", str(year))
    else:
        prompt = AISEARCHENGINE_TEMPLATE_EDU.replace("{{narrative_query}}", narrative_query).replace("{{topk}}", str(topk))
    while attempts < max_attempts:
        try:
            if engine=='Perplexity':
                content, citations = perplexity(prompt)
            elif engine=='TianGong':
                content = tiangong(prompt)
            elif engine=='GPT':
                content = gpt_search(prompt)
            elif engine=='Gemini':
                content = gemini_search(prompt)
            elif 'Manual' in engine:
                engine,knowledge_augmentation,enhance_retrieval=engine.split('_')
                content = manual_search(narrative_query,knowledge_augmentation,enhance_retrieval)
            else:
                sys.exit("Error: Unsupported search engine")
            if not edu_dataset:
                prompt = READ_SEARCH_CONTENT.replace("{{article}}", content)
            else:
                prompt = READ_SEARCH_CONTENT_EDU.replace("{{article}}", content)
            response = GPT_QA_not_stream(prompt, model_name="gpt-4o-mini", t=0.0)
            processed_data = process_recommendations(response,topk)
            ranked_result=processed_data.get("ranked_result", [])
            return ranked_result
        
        except Exception as e:
            #打印错误信息
            print(f"Error: {e}")
            attempts += 1
            if attempts >= max_attempts:
                return []  # 如果达到最大尝试次数，则返回空列表
            
def DeepResearch(narrative_query, topk=20, engine='Perplexity'):
    if 'I want to study further' in narrative_query:
        edu_dataset=True
    else:
        edu_dataset=False
        
    year_matches = re.findall(r'\((\d{4})\)', narrative_query)
    year = max(map(int, year_matches)) if year_matches else 2000
    attempts = 0
    max_attempts = 2

    prompt=narrative_query
    while attempts < max_attempts:
        try:
            if engine=='Perplexity':
                content = perplexity_deepresearch(prompt)
            elif engine=='Open':
                content = open_deepresearch(prompt)
            elif engine=='GPT':
                content = gpt_deepresearch(prompt)
            else:
                sys.exit("Error: Unsupported search engine")
            if not edu_dataset:
                prompt = READ_SEARCH_CONTENT.replace("{{article}}", content).replace("{{query}}", str(narrative_query))
            else:
                prompt = READ_SEARCH_CONTENT_EDU.replace("{{article}}", content).replace("{{query}}", str(narrative_query))
            response = GPT_QA_not_stream(prompt, model_name="gpt-4o-mini", t=0.0)
            processed_data = process_recommendations(response,topk)
            ranked_result=processed_data.get("ranked_result", [])
            return ranked_result
        
        except Exception as e:
            #打印错误信息
            print(f"Error: {e}")
            attempts += 1
            if attempts >= max_attempts:
                return []  # 如果达到最大尝试次数，则返回空列表

def RAG(narrative_query, topk=10, model_name='gpt-4o-mini'):
    ranked_result=RAG_paradigm(narrative_query)[:topk]
    return ranked_result