set -a                
source .env           
set +a

python retrieval.py \
  --dataset_name "movie_296"


