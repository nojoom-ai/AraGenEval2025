# AraGenEval2025/config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # ai model
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    #data
    TRAIN_FILE = "data/AuthorshipStyleTransferTrain.xlsx"
    VAL_FILE = "data/AuthorshipStyleTransferVal.xlsx"
    #baseline model
    OUTPUT_FILE = "models/baseline_model/results/predictions.xlsx"
    LOG_FILE = "models/baseline_model/results/generation_log.json"
    FEWSHOT_EXAMPLES_FILE = "models/baseline_model/fewshot_sampling/fewshot_examples_per_author.csv"
    NUM_FEWSHOT_EXAMPLES = 3
    # fine tuned model
    MODEL_LOG_FILE = "models/fine_tuned_model/results/{MODEL_NAME}/generation_log.json"
    MODEL_WEIGHTS_FOLDER = "models/fine_tuned_model/model_weights/"
    #eval prepared file
    LABELED_LOG = "evaluation/labeled_log.csv"
    PER_AUTHOR_METRICS_LOG = "evaluation/per_author_metrics.csv"
    
    
