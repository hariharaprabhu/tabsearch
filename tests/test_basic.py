import logging
logging.getLogger('hybrid_vectorizer').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
import numpy as np
from hybrid_vectorizer import HybridVectorizer
import pandas as pd
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

import sys
from contextlib import redirect_stdout
from io import StringIO

# ✅ Load stock data
df = pd.read_csv("sp500_companies.csv")

# ✅ Select Columns to Use
df = df[[
    "Exchange","Symbol", "Sector", "Industry", "Currentprice", "Marketcap","Ebitda", "Revenuegrowth", 
    "City", "State", "Country", "Fulltimeemployees", "Longbusinesssummary", "Weight"
]]
import logging
logging.getLogger('hybrid_vectorizer').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

# Also suppress any print statements from the library
import sys
from contextlib import redirect_stdout
from io import StringIO

# Your existing code...
hv = HybridVectorizer(index_column="Symbol")

print("🔄 Fitting model... (this may take a moment)")
with redirect_stdout(StringIO()):  # Suppress debug prints
    vectors = hv.fit_transform(df)

print(f"✅ Generated {vectors.shape[0]} vectors with {vectors.shape[1]} dimensions")

query = df.loc[df['Symbol']=='GOOGL'].iloc[0].to_dict()

print("🔍 Searching for similar companies...")
with redirect_stdout(StringIO()):  # Suppress debug prints
    results = hv.similarity_search(
        query, 
        ignore_exact_matches=True,
        block_weights={'text': 0.5, 'numerical': 1.0, 'categorical': 1.0}
    )

print("📊 Results:")
print(results[['Symbol', 'Sector', 'Industry', 'similarity']].head())
