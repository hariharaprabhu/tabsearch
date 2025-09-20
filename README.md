# tabsearch

**Most similarity search breaks on real-world mixed datasets (text + numbers + categories). This package fixes that.**

[![PyPI](https://img.shields.io/pypi/v/tabsearch.svg)](https://pypi.org/project/tabsearch/)  
[![Downloads](https://static.pepy.tech/badge/tabsearch)](https://pepy.tech/project/tabsearch)  
[![License](https://img.shields.io/badge/license-Apache--2.0-black.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> tabsearch makes **real-world datasets searchable** — by meaning, category, and numbers — all at once.

---

🚀 Why This Package?

- **Quick to use**: Install → run demo → see results in 60 seconds
- **Actually works**: Find similar items across text, categories, and numbers
- **Intelligently automated**: Automatic type detection with configurable weights

```python
# Text-only similarity: "GOOGL" ≈ ["Nike", "Starbucks"] 🤦
# tabsearch: "GOOGL" ≈ ["MSFT", "AMZN", "META"] ✅
```

---

## 🔧 Quick Start

```bash
pip install tabsearch
```

**Hello World example:**
```python
import pandas as pd
from mixed_data_vectorizer import HybridVectorizer

# Simple mixed dataset
df = pd.DataFrame({
    "id": [1, 2, 3],
    "category": ["Tech", "Retail", "Tech"], 
    "price": [100, 50, 200],
    "description": ["AI software", "Online store", "Cloud platform"]
})

hv = HybridVectorizer(index_column="id")
hv.fit_transform(df)
results = hv.similarity_search(df.iloc[0].to_dict(), ignore_exact_matches=True)
print(results)  # Finds id=3 (both Tech + high price) over id=2 (different category)
```

**Real-world example with S&P 500:**

```python
import pandas as pd
from mixed_data_vectorizer import HybridVectorizer

# Load real S&P 500 dataset
df = pd.read_csv("https://raw.githubusercontent.com/hariharaprabhu/tabsearch/main/Examples/Similar%20Stock%20Tickers/sp500_companies.csv")

# Select mixed-type columns
df = df[["Symbol", "Sector", "Industry", "Currentprice", "Marketcap", 
         "Fulltimeemployees", "Longbusinesssummary"]].copy()

# Automatic setup - detects column types
hv = HybridVectorizer(index_column="Symbol")
vectors = hv.fit_transform(df)
print(f"Generated {vectors.shape[0]} vectors with {vectors.shape[1]} dimensions")

# Find companies similar to Google
query = df.loc[df['Symbol']=='GOOGL'].iloc[0].to_dict()
results = hv.similarity_search(query, ignore_exact_matches=True)

print(results[['Symbol', 'Sector', 'Industry', 'similarity']].head())
#   Symbol  Sector      Industry                    similarity
#   MSFT    Technology  Software—Infrastructure     0.92
#   AMZN    Technology  Internet Retail             0.87
#   META    Technology  Internet Content & Info     0.85
#   AAPL    Technology  Consumer Electronics        0.83
```

---

## 🔍 How It Works

```
[text embeddings]  [categorical encoding]  [numerical scaling]
         │                    │                      │
         └──────────── weighted fusion ──────────────┘  → similarity score
```

- 📝 **Text** → Sentence transformer embeddings (Longbusinesssummary)
- 🏷️ **Categories** → Automatic encoding (Sector, Industry)  
- 🔢 **Numbers** → Normalized scaling (Currentprice, Marketcap, Fulltimeemployees)
- ⚖️ **Fusion** → Configurable weighted combination

**The key insight:** Each data type contributes equally to similarity, regardless of dimension count.

---

## ⚙️ Configuration

```python
# Control similarity focus with real S&P 500 data
query = df.loc[df['Symbol']=='GOOGL'].iloc[0].to_dict()

# Emphasize business description similarity
text_heavy = hv.similarity_search(
    query,
    block_weights={'text': 1.0, 'numerical': 0.5, 'categorical': 0.5},
    ignore_exact_matches=True
)

# Emphasize financial metrics
numeric_heavy = hv.similarity_search(
    query,
    block_weights={'text': 0.3, 'numerical': 1.0, 'categorical': 0.5},
    ignore_exact_matches=True
)

# Emphasize sector/industry similarity
categorical_heavy = hv.similarity_search(
    query,
    block_weights={'text': 0.3, 'numerical': 0.5, 'categorical': 1.0},
    ignore_exact_matches=True
)

print("Text-heavy results:", text_heavy[['Symbol', 'Sector']].head(3).values.tolist())
print("Numeric-heavy results:", numeric_heavy[['Symbol', 'Sector']].head(3).values.tolist()) 
print("Categorical-heavy results:", categorical_heavy[['Symbol', 'Sector']].head(3).values.tolist())
```

---

## ✅ Features

- **Automatic type detection** - No manual column specification needed
- **Block weight tuning** - Control text vs numerical vs categorical importance
- **Model persistence** - Save/load trained models
- **FAISS integration** - Speed up search on large datasets
- **Encoding inspection** - See how each column was processed

---

## 💡 Use Cases

- **Investment research** → Find companies by business model + financial metrics
- **E-commerce** → Product recommendations by description + category + price
- **Customer analytics** → Segment users by demographics + behavior + purchase history
- **Content matching** → Similar articles by topic + engagement + metadata

---

## 🔗 Advanced Usage

**Model persistence:**
```python
# Save trained model
hv.save('sp500_model.pkl')

# Load later
hv2 = HybridVectorizer.load('sp500_model.pkl')
results = hv2.similarity_search(query)
```

**FAISS integration for large datasets:**
```python
import faiss

# L2 normalize vectors for cosine similarity
def l2norm(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

normalized_vectors = l2norm(vectors)

# Build FAISS index
index = faiss.IndexFlatIP(normalized_vectors.shape[1])
index.add(normalized_vectors.astype('float32'))
hv.set_vector_db(index)

# Now similarity_search uses FAISS internally
results = hv.similarity_search(query, top_n=10, ignore_exact_matches=True)
```

**Inspect encoding:**
```python
# See how each column was processed
report = hv.get_encoding_report()
for column, encoding_type in report.items():
    print(f"{column}: {encoding_type}")

# Output example:
# Symbol: categorical
# Sector: categorical  
# Industry: categorical
# Currentprice: numerical
# Marketcap: numerical
# Fulltimeemployees: numerical
# Longbusinesssummary: text
```

## ❓ FAQ

**Q: Can I use a different text model instead of the default?**  
Yes. By default, `tabsearch` uses the `all-MiniLM-L6-v2` model from `sentence-transformers`.  
Advanced users can override this in two ways:

- Pass a different model name:  
  ```python
  from mixed_data_vectorizer import HybridVectorizer
  hv = HybridVectorizer(default_text_model="multi-qa-mpnet-base-dot-v1")
Pass a pre-loaded model:

python
Copy code
from mixed_data_vectorizer import HybridVectorizer
from sentence_transformers import SentenceTransformer

custom_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
hv = HybridVectorizer(default_text_model=custom_model)
Q: Does this scale to millions of rows?
Yes, via external vector databases.
By default, embeddings are stored in memory as NumPy arrays.
For large datasets, connect to FAISS (or pgvector, Milvus, Qdrant) using:

python
Copy code
hv.set_vector_db(faiss_index)
Q: How do I evaluate similarity quality without labels?
We recommend:

Compare hybrid vs text-only vs numeric-only neighbors

Use metrics like precision@k or nDCG if you have partial labels

For exploratory use, inspect block-level contributions with explain_similarity()

Q: How is this different from just using FAISS or Pinecone directly?
Vector databases index embeddings but don’t handle mixed tabular data.
tabsearch automatically:

Detects column types

Encodes each block appropriately

Fuses them with configurable weights

Provides explainable results
---
## ❓ FAQ

**Q: Can I use a different text model instead of the default?**  
Yes. By default, `tabsearch` uses the `all-MiniLM-L6-v2` model from `sentence-transformers`.  
Advanced users can override this in two ways:

- Pass a different model name:  
  ```python
  from mixed_data_vectorizer import HybridVectorizer
  hv = HybridVectorizer(default_text_model="multi-qa-mpnet-base-dot-v1")
  ```

- Pass a pre-loaded model:  
  ```python
  from mixed_data_vectorizer import HybridVectorizer
  from sentence_transformers import SentenceTransformer

  custom_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")
  hv = HybridVectorizer(default_text_model=custom_model)
  ```

---

**Q: Does this scale to millions of rows?**  
Yes, via external vector databases.  
By default, embeddings are stored in memory as NumPy arrays.  
For large datasets, connect to FAISS (or pgvector, Milvus, Qdrant) using:  

```python
hv.set_vector_db(faiss_index)
```

---

**Q: How do I evaluate similarity quality without labels?**  
We recommend:  
- Compare hybrid vs text-only vs numeric-only neighbors  
- Use metrics like precision@k or nDCG if you have partial labels  
- For exploratory use, inspect block-level contributions with `explain_similarity()`

---

**Q: How is this different from just using FAISS or Pinecone directly?**  
Vector databases index embeddings but don’t handle **mixed tabular data**.  
`tabsearch` automatically:  
- Detects column types  
- Encodes each block appropriately  
- Fuses them with configurable weights  
- Provides explainable results
---

## 📊 Example Results

On the real S&P 500 dataset (500 companies, 7 mixed columns):

| Method            | GOOGL → Top 3 Results       | Why This Matters |
|-------------------|-----------------------------|------------------|
| Text-only         | Consumer/retail companies   | Ignores financial scale and tech sector |
| Numbers-only      | Any large companies         | Ignores business model similarity |
| **Hybrid (this)** | **MSFT, AMZN, META**        | **Captures business + financial + sector similarity** |

---

⚡ **Scaling Note:**  
The default in-memory NumPy backend works well up to ~100k rows.  
For larger datasets, use `hv.set_vector_db()` with FAISS or another vector DB.  
See the [Scaling Guide](docs/scaling.md) for examples.


## 🛠️ Installation

**Basic:**
```bash
pip install tabsearch
```

**With FAISS for large datasets:**
```bash
pip install tabsearch faiss-cpu
```

**With GPU support:**
```bash
pip install tabsearch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 🧪 Try It Now

Run this demo in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mURspqx-rsfgnoH1cKQM49AmxT_mCFf_)


## 🗺️ Roadmap

Planned improvements and extensions:

- **Hugging Face ecosystem integration**  
  Allow users to plug in any Hugging Face text embedding model, with plans to publish a Hugging Face Space demo.

- **Benchmarks on open datasets**  
  Evaluate performance on UCI Adult, Amazon product reviews, and other public datasets to demonstrate real-world gains.

- **Streamlit demo UI**  
  Build an interactive demo app to explore mixed-data similarity search without writing code.

- **LangChain retriever wrapper** (future)  
  Provide a retriever class for hybrid similarity search inside LangChain workflows.

- **Community feedback loop**  
  Expand tutorials, examples, and features based on user input and contributions.

- **Performance Testing**
  Test this package on various datasets and record the benchmarks

---

## 📄 License

Apache-2.0

---

**Try it because:** tabsearch is the fastest way to get meaningful similarity search on datasets that combine business descriptions, financial metrics, and categorical data.