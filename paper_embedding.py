import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

data = pd.read_csv("data/arxiv_papers.csv", index_col=False)

model = SentenceTransformer("all-MiniLM-L6-v2")
data['text'] = data['title'] + '. ' + data['abstract']
data['embedding'] = model.encode(data['text'].tolist(), show_progress_bar=True).tolist()
data = data.drop(columns=['text'])

data.to_csv('data/papers_final.csv')