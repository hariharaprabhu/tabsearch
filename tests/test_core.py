import math
import numpy as np
import pandas as pd
import pytest

from hybrid_vectorizer import HybridVectorizer

# ---------- Fixtures ----------
@pytest.fixture
def df_num_only():
    # id, one numeric column with a clear nearest neighbor
    return pd.DataFrame({
        "id": ["A", "B", "C", "D"],
        "price": [100.0, 101.0, 250.0, 99.5],
    })

@pytest.fixture
def df_cat_only():
    # id, one categorical column with repeated categories
    return pd.DataFrame({
        "id": ["A", "B", "C", "D"],
        "category": ["x", "y", "x", "z"],
    })

@pytest.fixture
def df_mixed_small():
    # Numeric + categorical; lets us test weight effects
    return pd.DataFrame({
        "id": ["A", "B", "C", "D"],
        "category": ["x", "x", "y", "y"],
        "price":   [100.0, 130.0, 101.0, 400.0],
    })

# ---------- Tests ----------
def test_numeric_only_vectors_and_search(df_num_only):
    hv = HybridVectorizer(index_column="id")
    vecs = hv.fit_transform(df_num_only)
    assert vecs.shape[0] == len(df_num_only)
    assert vecs.shape[1] > 0

    q = df_num_only.iloc[0].to_dict()  # id A, price 100.0
    res = hv.similarity_search(q, top_n=2, ignore_exact_matches=True)
    # nearest by price should be D (99.5) or B (101.0); D is closer
    top1 = res.iloc[0]["id"]
    assert top1 in ("D", "B")
    # D should generally beat B here; allow B as fallback if implementation differs slightly
    if res.iloc[0]["id"] == "B":
        # Then D must be second
        assert "D" in set(res["id"][:2])

def test_categorical_only_prefers_same_category(df_cat_only):
    hv = HybridVectorizer(index_column="id")
    _ = hv.fit_transform(df_cat_only)

    q = df_cat_only.iloc[0].to_dict()  # A with category x
    res = hv.similarity_search(q, top_n=2, ignore_exact_matches=True)
    # C has same category "x" -> should be the top neighbor
    assert res.iloc[0]["id"] == "C"

def test_weights_shift_ranking(df_mixed_small):
    hv = HybridVectorizer(index_column="id")
    _ = hv.fit_transform(df_mixed_small)

    q = df_mixed_small.iloc[0].to_dict()  # A: category x, price 100

    # Emphasize categorical -> prefer B (same category x) despite price gap
    res_cat = hv.similarity_search(
        q, top_n=1, ignore_exact_matches=True,
        block_weights={"categorical": 2.0, "numerical": 0.5}
    )
    assert res_cat.iloc[0]["id"] == "B"

    # Emphasize numerical -> prefer C (price 101, category y)
    res_num = hv.similarity_search(
        q, top_n=1, ignore_exact_matches=True,
        block_weights={"categorical": 0.5, "numerical": 2.0}
    )
    assert res_num.iloc[0]["id"] == "C"

def test_missing_values_do_not_crash():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "category": ["x", None, "x", "y"],
        "price": [np.nan, 10.0, 10.0, None],
    })
    hv = HybridVectorizer(index_column="id")
    vecs = hv.fit_transform(df)
    assert vecs.shape[0] == 4
    # Smoke search
    _ = hv.similarity_search(df.iloc[0].to_dict(), top_n=2, ignore_exact_matches=True)

def test_index_uniqueness_enforced():
    df = pd.DataFrame({"id": [1,1,2], "price": [1.0, 2.0, 3.0]})
    hv = HybridVectorizer(index_column="id")
    with pytest.raises(Exception):
        hv.fit_transform(df)

def test_ignore_exact_matches(df_num_only):
    hv = HybridVectorizer(index_column="id")
    _ = hv.fit_transform(df_num_only)
    q = df_num_only.iloc[0].to_dict()
    res = hv.similarity_search(q, top_n=3, ignore_exact_matches=True)
    assert "A" not in set(res["id"])  # A is the query row

def test_save_load_roundtrip(df_num_only, tmp_path):
    hv = HybridVectorizer(index_column="id")
    _ = hv.fit_transform(df_num_only)
    p = tmp_path / "hv.pkl"
    hv.save(str(p))

    hv2 = HybridVectorizer.load(str(p))
    q = df_num_only.iloc[0].to_dict()
    r1 = hv.similarity_search(q, top_n=3, ignore_exact_matches=True)
    r2 = hv2.similarity_search(q, top_n=3, ignore_exact_matches=True)
    # Rankings should match
    assert list(r1["id"]) == list(r2["id"])

@pytest.mark.skipif(
    pytest.importorskip("faiss", reason="faiss-cpu not installed") is None,
    reason="faiss not available"
)
def test_faiss_integration_numeric(df_num_only):
    import faiss
    hv = HybridVectorizer(index_column="id")
    vecs = hv.fit_transform(df_num_only)

    # L2-normalize for cosine via inner product
    n = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs_norm = (vecs / n).astype("float32")
    index = faiss.IndexFlatIP(vecs_norm.shape[1])
    index.add(vecs_norm)
    # If your API provides this hook:
    hv.set_vector_db(index)

    q = df_num_only.iloc[0].to_dict()
    res = hv.similarity_search(q, top_n=2, ignore_exact_matches=True)
    assert len(res) == 2
