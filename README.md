
# OCG Agent
**We propose a retrieval–reranking paradigm for the narrative-based recommendation task. Specifically, we introduce the Open Candidate Generation Agent, a deep‐research–oriented information retrieval module designed to retrieve structured candidates both comprehensively and in depth. Its goal is to assemble a high‐quality, sufficiently large candidate set that can effectively support the downstream recommendation pipeline.**  

## Usage  
1. clone this project.
2. create `.env` file, we provide the `.env.example`, the `OPENAI_API_KEY` and `SERPER_SEARCH_KEY` is necessary.
3. `pip install -r requirements.txt`
4. clone the [Docling](https://github.com/docling-project/docling), which is used for reading url content.
5. `bash retrieval.sh`, the generated candiddates will be stored in `./AI_search_content/movie_296`.

## Notes  
More details will be released soon.
