import os
import pandas as pd
from train import train_and_save
from inference import predict_texts
from logger import logger


df = pd.read_csv("bns_sections.csv")  # <-- update with your filename
text_col = 'Description'
label_col = 'Chapter_grouped'
RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_SAMPLES_TO_KEEP = 2  # classes with < this will be grouped into "Other"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
train_and_save(df, ARTIFACT_DIR, text_col, label_col,
               TEST_SIZE, RANDOM_STATE, MIN_SAMPLES_TO_KEEP)
logger.info('Testing prediction:---')
logger.info('Testing sample:---')
logger.info(predict_texts(["A short legal provision text ..."]))
