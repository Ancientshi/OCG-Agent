source /home86/yunxshi/.bashrc

# METHOD_LIST=('TFIDF' 'BGE' 'EasyRec')

# METHOD_LIST=('BM25REC' 'BGE' 'EasyRec' 'LLMRerank')
METHOD_LIST=('GPT')
DATASET_LIST=('movie')
# RANK_QUERY_LIST=('query' 'profile' 'both')
RANK_QUERY_LIST=('gpt')

for DATASET in "${DATASET_LIST[@]}"
do
    for METHOD in "${METHOD_LIST[@]}"
    do  
        for RANK_QUERY in "${RANK_QUERY_LIST[@]}"
        do
            python rec_ocg.py \
                --dataset_name $DATASET \
                --method $METHOD \
                --rank_query $RANK_QUERY
        done
    done
done

wait 
echo "All done!"