import logging
import numpy as np
from hybrid_vectorizer import HybridVectorizer
import pandas as pd


# ✅ Load stock data
df = pd.read_csv("sp500_companies.csv")

# ✅ Select Columns to Use
df = df[[
    "Exchange","Symbol", "Sector", "Industry", "Currentprice", "Marketcap","Ebitda", "Revenuegrowth", 
    "City", "State", "Country", "Fulltimeemployees", "Longbusinesssummary", "Weight"
]]


# Your existing code...
hv = HybridVectorizer(index_column="Symbol")

print("🔄 Fitting model... (this may take a moment)")
vectors = hv.fit_transform(df)

print(f"✅ Generated {vectors.shape[0]} vectors with {vectors.shape[1]} dimensions")

query = df.loc[df['Symbol']=='GOOGL'].iloc[0].to_dict()

print("🔍 Searching for similar companies...")
results = hv.similarity_search(
        query, 
        ignore_exact_matches=True,
        block_weights={'text': 0.5, 'numerical': 1.0, 'categorical': 1.0}
    )

print("📊 Results:")
print(results[['Symbol', 'Sector', 'Industry', 'similarity']].head())
