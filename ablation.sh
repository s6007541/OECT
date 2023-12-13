
DATASETS=("yahoo_answers" "20_newsgroup" "ag_news" "dbpedia" "isear" "polarity" "sms_spam" "subjectivity")
CLUSTERING_ALGOS=("kmeans", "sib")
SEEDS=(0 1 2)

for DATASET in "${DATASETS[@]}"; do
    for C_ALGO in "${CLUSTERING_ALGOS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            python run_experiment.py --train_file datasets/${DATASET}/train.csv --eval_file datasets/${DATASET}/test.csv --num_clusters 50 --labeling_budget 64 --finetuning_epochs 10 --inter_training_epochs 1 --random_seed ${SEED} --clustering_algo ${C_ALGO} --pipeline SECF --cuda "0"
        done
    done
done
