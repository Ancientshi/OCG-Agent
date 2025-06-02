from openai import OpenAI
from utils.utils import GPT_QA_not_stream, filter_content_bm25, content_post_process
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent
import time
from .Docling import docling_read_file
import os
from config import serper_search_key

class SerperSearch:
    def __init__(self,key=''):
        self.api_key=key
        self.url = "https://google.serper.dev/search"
        self.max_chars=100000
    
    def searchserper(self,query,pagenum):
        payload = json.dumps({
            "q": query
        })
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
            
        # Call the API
        retry_interval_exp = 0
        while True:
            try:
                response = requests.request("POST", self.url, headers=headers, data=payload)
                response.raise_for_status()
                return response.json()
            except Exception as ex:
                if retry_interval_exp > 3:
                    return {}
                time.sleep(max(2, 0.5 * (2 ** retry_interval_exp)))
                retry_interval_exp += 1
    
    def search(self, query, pagenum,threshold=1.0,existed_citation_list=[]):
        external_knowledge = {}
        search_results = self.searchserper(query,pagenum)
        
        def process_result(result):
            title = result["title"]
            if 'link' in result.keys():
                url=result['link']
                if url in existed_citation_list:
                    return None
            else:
                return None    
            
            #page_content = extract_page(url, True, True)
            page_content = docling_read_file(url)
            if page_content != 'Not available':
                content_length = len(page_content)
                max_chars = min(self.max_chars, content_length)
                page_content = page_content[:max_chars]
                try:
                    page_content=filter_content_bm25(page_content, query, threshold=threshold,maxlength = 10000)
                except:
                    pass
                formatted_title = title + ':' + url
                return formatted_title, page_content
            return None

        citation_list=[]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_result, result) for result in search_results['organic'][:pagenum]]
            index=1
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    formatted_title, page_content=result
                    _title, _url = formatted_title.split(':http')
                    citation_list.append('http'+_url)
                    external_knowledge.update({f'{_title}[{index}]': page_content})
        return external_knowledge, citation_list

search_tool=SerperSearch(serper_search_key)





def self_AI_search(query="",pagenum=1,threshold=0,existed_citation_list=[]):
    external_knowledge_dict,citation_list=search_tool.search(query, pagenum,threshold,existed_citation_list)
    for key in external_knowledge_dict:
        external_knowledge_dict[key] = content_post_process(external_knowledge_dict[key])
    return external_knowledge_dict,citation_list

def self_AI_search_edu(query=""):
    response = requests.post(f"NOT PUBLIC", json={"requirement": query})
    external_knowledge_str = response.json()['response']
    external_knowledge_dict=json.loads(external_knowledge_str)
    return filtered_external_knowledge_dict, []


# Example usage:
if __name__ == "__main__":
    pass