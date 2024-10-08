#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=selfragasqa10doc # le nom du job (voir commande squeue)
#SBATCH --nodes=1 # le nombre de noeuds
#SBATCH --gpus=1 # nombre de gpu
#SBATCH --ntasks-per-node=1 # nombre de tache par noeud 
#SBATCH --time=1-90:00:00             # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=jz_%j_%x.out     # nom du fichier de sortie
#SBATCH --error=errjz_%j_%x.out      # nom du fichier d'erreur (ici commun avec la sortie)

# Source l'environement par example ~/.bashrc
source ~/.bashrc
# activer l'environement python
conda activate selfrag
cd /home/djeddal/Documents/Code/self-rag/retrieval_lm

#python run_long_form_static.py   --model_name selfrag/selfrag_llama2_7b --input_file data/hagrid_monot5_100top20.csv  --ndocs 10 --max_new_tokens 1000 --threshold 0.2   --use_grounding --use_utility --use_seqscore   --task hagrid    --output_file selfrag_hagrid_with_retrieval_10docs.json --max_depth 7 --mode always_retrieve
python run_long_form_static.py  --model_name selfrag/selfrag_llama2_7b  --ndocs 10 --max_new_tokens 300 --threshold 0.2 --use_grounding --use_utility --use_seqscore --task asqa --input_file /home/djeddal/Documents/Code/ALCE/data/asqa_eval_gtr_top100.json --output_file selfrag_asqa_with_retrieval_eval_gtr_top100_10docs.json --max_depth 7 --mode always_retrieve 