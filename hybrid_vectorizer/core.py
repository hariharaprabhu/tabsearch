## import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional, Dict, Union, List
import logging
import joblib
import numpy as np
import torch
from .exceptions import HybridVectorizerError, ModelNotFittedError, InvalidQueryError, UnsupportedDataTypeError, DimensionMismatchError

SentenceTransformer._encode = SentenceTransformer.encode
SentenceTransformer.encode = lambda self, *args, **kwargs: self._encode(*args, **{**kwargs, "show_progress_bar": False})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("HybridVectorizer")

class HybridVectorizer:
    """
    HybridVectorizer: Robust fusion of tabular, text, and optional image/audio features
    for similarity search and hybrid ML pipelines.

    **Supported Modalities**
    - Numerical columns (auto-detects, normalizes)
    - Categorical columns (one-hot or frequency encoding)
    - Text columns (SentenceTransformer or user-provided model)
    - Date columns (ignored by default, but can extract year/month/day if set)
    - Index/ID columns (ignored for similarity, always preserved in results)

    **Key Features**
    - Late fusion (default): More robust to missing blocks, easier to reason about
    - Ignore columns by name or dtype (date, id, etc.) with `ignore_columns` or `ignore_by_dtype`
    - User-friendly error handling for constant/all-NaN columns (logged & skipped)
    - Always returns all ignored/index/date columns in similarity search results for context
    - Pluggable vector DB (e.g., FAISS)
    - Logging and detailed encoding report

    **Example Usage**
    >>> hv = HybridVectorizer(
    ...     column_encodings={'desc': 'text', 'region': 'categorical'},
    ...     ignore_columns=['id', 'created_at'],
    ...     index_column='id',
    ...     date_feature_mode='extract'  # 'extract' or None
    ... )
    >>> vectors = hv.fit_transform(df)
    >>> hv.similarity_search(query_row, top_n=5)

    For more, see the README: https://github.com/yourrepo

    Parameters
    ----------
    column_encodings : dict, optional
        Manual override for column types, e.g. {'desc': 'text'}
    ignore_columns : list of str, optional
        Explicitly skip these columns (preserved in output, not embedded)
    ignore_by_dtype : list of str, optional
        Auto-ignore columns by dtype (e.g. ['date', 'timedelta'])
    index_column : str, optional
        Column name for index/ID (never used for similarity, always returned)
    id_column : str, optional
        Alias for index_column if needed
    date_feature_mode : str, optional
        If 'extract', extracts year/month/day/weekday as numerical features from date columns.
        If None, skips date columns entirely (default).
    default_text_model : str or SentenceTransformer
        Text model for embedding text columns.
    onehot_threshold : int
        Max unique values for one-hot encoding categorical columns.
    text_batch_size : int
        Batch size for text embedding.
    vector_db : object, optional
        Optional external vector DB (e.g., FAISS).
    default_block_weights : dict, optional
        Default weights for block types in late fusion.
    """

    def __init__(
        self,
        column_encodings: Optional[Dict[str, str]] = None,
        ignore_columns: Optional[List[str]] = None,
        ignore_by_dtype: Optional[List[str]] = None,
        index_column: Optional[str] = None,
        id_column: Optional[str] = None,
        date_feature_mode: Optional[str] = None,
        default_text_model: str = 'all-MiniLM-L6-v2',
        onehot_threshold: int = 10,
        text_batch_size: int = 128,
        vector_db=None,
        default_block_weights: Optional[Dict[str, float]] = None,
        verbose: bool = False
    ):
        # Check dependencies at startup
        try:
            import torch
            import sentence_transformers
            import sklearn
        except ImportError as e:
            raise ImportError(
                f"Missing required dependency: {e}\n"
                "Install with: pip install torch sentence-transformers scikit-learn"
            )
        
        # Check GPU availability
        if torch.cuda.is_available():
            logger.info("GPU detected - will use for text encoding")
        else:
            logger.info("Using CPU for text encoding (GPU not available)")
        logger.info("Initializing HybridVectorizer")
        self.column_encodings = column_encodings or {}
        self.ignore_columns = set(ignore_columns or [])
        self.ignore_by_dtype = set(ignore_by_dtype or [])
        self.index_column = index_column
        self.id_column = id_column or index_column  # Alias for convenience
        self.date_feature_mode = date_feature_mode
        self.onehot_threshold = onehot_threshold
        self.text_batch_size = text_batch_size
        self.vector_db = vector_db
        self.default_block_weights = default_block_weights or {'text':2, 'categorical':2, 'numerical':2}
        self.verbose = verbose
        # Text model loading
        if not verbose:
            logging.getLogger('hybrid_vectorizer').setLevel(logging.WARNING)
        if isinstance(default_text_model, str):
            logger.info(f"Loading text model: {default_text_model}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.default_text_model = SentenceTransformer(default_text_model,device=device)
            self.default_text_model_name = default_text_model
        else:
            self.default_text_model = default_text_model
            self.default_text_model_name = type(default_text_model).__name__
        self.encoding_report = {}
        self.feature_names = []
        self.vectors = None
        self.df = None
        self.df_index = None
        self.fitted_encoders = {}  # {col: encoder/freq_map}
        self.text_col_dims = {}

    def set_vector_db(self, vector_db):
        logger.info("Setting external vector database.")
        self.vector_db = vector_db

    def _frequency_encode(self, series):
        logger.debug(f"Frequency encoding for {series.name}")
        freq = series.value_counts(normalize=True)
        return series.map(freq).fillna(0.0), freq.to_dict()

    def _normalize_numeric(self, series, scaler=None):
        if scaler is None:
            logger.debug(f"Normalizing (fit) numeric: {series.name}")
            scaler = MinMaxScaler()
            arr = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
            return arr, scaler
        logger.debug(f"Normalizing (transform) numeric: {series.name}")
        arr = scaler.transform(np.array(series).reshape(-1, 1)).flatten()
        return arr, scaler

    def _detect_type(self, series, col):
        # Manual override
        if col in self.column_encodings:
            val = self.column_encodings[col]
            logger.info(f"[{col}] Manual override: {val}")
            logger.debug(f"Column {col} type override: {val}")
            if isinstance(val, type) and issubclass(val, (float, int)):
                logger.info(f"[{col}] -> numerical (manual override, type)")
                return "numerical"
            elif isinstance(val, str):
                logger.info(f"[{col}] -> {val} (manual override, str)")
                return val
            else:
                logger.info(f"[{col}] -> custom (manual override, unknown type)")
                return "custom"
        # Dtype detection
        
        if pd.api.types.is_datetime64_any_dtype(series):
            logger.info(f"[{col}] -> date (dtype: {series.dtype})")
            return "date"
        if pd.api.types.is_numeric_dtype(series):
            logger.info(f"[{col}] -> numerical (dtype: {series.dtype})")
            return "numerical"
        if pd.api.types.is_string_dtype(series) or series.dtype == "object":
            s = series.dropna().astype(str)
            nunique = s.nunique()
            avg_len = s.map(len).mean() if not s.empty else 0
            logger.info(f"[{col}] string column: nunique={nunique}, avg_len={avg_len:.1f}")
            if nunique > 10:
                logger.info(f"[{col}] -> text (nunique > 10)")
                return "text"
            else:
                logger.info(f"[{col}] -> categorical (not enough unique)")
                return "categorical"

        logger.info(f"[{col}] -> categorical (fallback)")
        return "categorical"
    def _get_adaptive_weights(self, query, block_weights=None):
        """Auto-adjust block weights to 0 if query doesn't contain relevant data."""
        if hasattr(query, 'to_dict'):
            query_dict = query.to_dict()
        else:
            query_dict = query
        
        if block_weights is None:
            block_weights = self.default_block_weights.copy()
        else:
            block_weights = block_weights.copy()
        
        # Check what data types are in the query
        query_has_text = False
        query_has_categorical = False  
        query_has_numerical = False
        
        for col, value in query_dict.items():
            if value is not None and (not isinstance(value, str) or value.strip()):
                if col in self.encoding_report:
                    col_type = self.encoding_report[col].get('type', '')
                    if col_type == 'text':
                        query_has_text = True
                    elif col_type == 'categorical':
                        query_has_categorical = True
                    elif col_type == 'numerical':
                        query_has_numerical = True
        
        # Set weights to 0 for missing data types with logging
        original_weights = block_weights.copy()
        
        if not query_has_text:
            block_weights['text'] = 0.0
            if not self.verbose:
                print("📊 No text data in query - focusing on numerical/categorical similarity")
        
        if not query_has_categorical:
            block_weights['categorical'] = 0.0
            if not self.verbose:
                print("📊 No categorical data in query - focusing on text/numerical similarity")
        
        if not query_has_numerical:
            block_weights['numerical'] = 0.0
            if not self.verbose:
                print("📊 No numerical data in query - focusing on text/categorical similarity")
        
        # Show final weights if anything changed
        if block_weights != original_weights and not self.verbose:
            active_blocks = [k for k, v in block_weights.items() if v > 0]
            print(f"🎯 Using similarity from: {', '.join(active_blocks)}")
        
        return block_weights

    def fit_transform(self, df: pd.DataFrame, feature_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        # Add at the very beginning:
        if not isinstance(df, pd.DataFrame):
            raise UnsupportedDataTypeError("Input must be a pandas DataFrame")
        
        

        if df.empty:
            raise InvalidQueryError("DataFrame cannot be empty")
        
        if len(df.columns) == 0:
            raise InvalidQueryError("DataFrame must have at least one column")

        if self.verbose:
            logger.info("Fitting and transforming input DataFrame")
            logger.info("Starting fit_transform...")
        else:
            print("🔄 Processing data")
        
        # Check for invalid column references
        if self.index_column and self.index_column not in df.columns:
            raise InvalidQueryError(f"Index column '{self.index_column}' not found in DataFrame")

        
        # Warn about invalid ignore columns (don't error, just warn)
        invalid_ignore = [col for col in self.ignore_columns if col not in df.columns]
        if invalid_ignore:
            logger.warning(f"Ignore columns not found: {invalid_ignore}")


        logger.info("Fitting and transforming input DataFrame")
        logger.info("Starting fit_transform...")
        if feature_weights is None:
            feature_weights = {'text': 5, 'categorical': 2, 'numerical': 7}
        self.df = df.copy()
        processed_blocks = []
        self.feature_names = []
        self.encoding_report = {}
        self.fitted_encoders = {}
        self.text_col_dims = {}
        self.df_index = df.index

        block_types = ['numerical', 'categorical', 'text']
        block_arrays = {b: [] for b in block_types}
        block_names = {b: [] for b in block_types}
        self.block_feature_names = {}

        for col in df.columns:
            # ---- Ignore columns logic
            if col in self.ignore_columns:
                logger.info(f"Ignored column (user request): {col}")
                self.encoding_report[col] = {"skipped": True, "reason": "ignore_columns"}
                continue
            if col == self.index_column or col == self.id_column:
                logger.info(f"Ignored column (index/id): {col}")
                self.encoding_report[col] = {"skipped": True, "reason": "index/id column"}
                continue
            col_type = self._detect_type(df[col], col)
            # ---- Ignore by dtype logic
            if col_type in self.ignore_by_dtype:
                logger.info(f"Ignored column (dtype): {col}")
                self.encoding_report[col] = {"skipped": True, "reason": f"ignore_by_dtype ({col_type})"}
                continue
            # ---- Ignore date columns unless extracting features
            if col_type == "date":
                if self.date_feature_mode == "extract":
                    dt = pd.to_datetime(df[col], errors='coerce')
                    for dt_feat in ["year", "month", "day", "weekday"]:
                        arr = getattr(dt.dt, dt_feat).fillna(0).astype(int).values
                        processed_blocks.append(('numerical', arr.reshape(-1, 1), [f"{col}_{dt_feat}"]))
                        self.encoding_report[f"{col}_{dt_feat}"] = {"type": "numerical", "from": "date"}
                else:
                    logger.info(f"Ignored date column: {col}")
                    self.encoding_report[col] = {"skipped": True, "reason": "date column"}
                    continue
            # ---- All-NaN/Constant skip
            series = df[col]
            nunique = series.nunique(dropna=True)
            if series.dropna().empty or nunique <= 1:
                logger.warning(f"Skipped column {col}: all NaN or constant")
                self.encoding_report[col] = {"skipped": True, "reason": "all NaN/constant"}
                continue
            # --- Encoding logic
            if col_type == "numerical":
                series = pd.to_numeric(series, errors="coerce")
                finite_vals = series.replace([np.inf, -np.inf], np.nan).dropna()
                max_val = finite_vals.max() if not finite_vals.empty else 0
                min_val = finite_vals.min() if not finite_vals.empty else 0
                series = series.replace(np.inf, max_val).replace(-np.inf, min_val)
                median = series.median()
                series = series.fillna(median)
                arr, scaler = self._normalize_numeric(series)
                processed_blocks.append(('numerical', arr.reshape(-1, 1), [col]))
                self.encoding_report[col] = {"type": "numerical", "encoding": "minmax"}
                self.fitted_encoders[col] = scaler
            elif col_type == "categorical":
                nunique = series.nunique(dropna=True)
                if nunique <= self.onehot_threshold:
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    encoded = encoder.fit_transform(series.values.reshape(-1, 1))
                    categories = encoder.categories_[0]
                    processed_blocks.append(('categorical', encoded, [f"{col}_oh_{cat}" for cat in categories]))
                    self.encoding_report[col] = {"type": "categorical", "encoding": "onehot", "categories": list(categories)}
                    self.fitted_encoders[col] = encoder
                else:
                    encoded, freq_map = self._frequency_encode(series.astype(str))
                    processed_blocks.append(('categorical', encoded.values.reshape(-1, 1), [col]))
                    self.encoding_report[col] = {"type": "categorical", "encoding": "frequency"}
                    self.fitted_encoders[col] = freq_map
            elif col_type == "text":
                text_filled = series.fillna("").astype(str)
                batch_size = self.text_batch_size if len(series) > self.text_batch_size else min(len(series), 32)
                if self.verbose:
                    logger.info(f"Encoding text column: {col} in batches of {batch_size}")
                else:
                    print(f"📝 Encoding text: {col}...") 
                embeddings = self.default_text_model.encode(
                    text_filled.tolist(),
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                dim = embeddings.shape[1]
                processed_blocks.append(('text', embeddings, [f"{col}_emb_{i}" for i in range(dim)]))
                self.encoding_report[col] = {
                    "type": "text",
                    "encoding": f"sentence-transformer ({self.default_text_model_name})"
                }
                self.fitted_encoders[col] = self.default_text_model
                self.text_col_dims[col] = dim
            else:
                logger.warning(f"Skipped column: {col}, unknown type: {col_type}")
                self.encoding_report[col] = {"skipped": True, "reason": f"unknown type {col_type}"}
                continue

        stacked_blocks = {}
        for btype in block_types:
            block_b = [arr for bt, arr, fn in processed_blocks if bt == btype]
            names_b = [fname for bt, arr, fn in processed_blocks if bt == btype for fname in fn]
            if block_b:
                stacked_blocks[btype] = np.hstack(block_b).astype(np.float32)
                self.block_feature_names[btype] = names_b
            else:
                stacked_blocks[btype] = np.zeros((len(df), 0), dtype=np.float32)
                self.block_feature_names[btype] = []

        self.block_vectors = {b: stacked_blocks[b] for b in block_types}

        self.feature_names = [fname for b in block_types for fname in self.block_feature_names[b]]
        self.vectors = np.hstack([stacked_blocks[b] for b in block_types])
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / (norms + 1e-8)
        self.vectors = self.vectors.astype(np.float32)
        del processed_blocks
        del block_arrays  
        del block_names
        del stacked_blocks
        if self.verbose:
            logger.info("fit_transform complete: %d rows, %d features", self.vectors.shape[0], self.vectors.shape[1])
        else:
            print(f"✅ Generated {self.vectors.shape[0]} vectors with {self.vectors.shape[1]} dimensions")  
        return self.vectors

    def _encode_single_feature(self, feat, val):
        """
        Encode a single feature value for query transformation.
        
        NOTE: This method is designed to handle individual feature encoding,
        but the real issue is in how it's called from _transform_query_generic.
        """
        try:
            if "_oh_" in feat:
                # One-hot encoded feature
                col, _, cat = feat.partition("_oh_")
                
                if col not in self.fitted_encoders:
                    logger.warning(f"Encoder not found for column: {col}")
                    return [0.0]
                
                encoder = self.fitted_encoders[col]
                cats = encoder.categories_[0]
                
                if val is not None and val in cats:
                    # Create proper one-hot vector
                    result = np.zeros(len(cats), dtype=float)
                    idx = np.where(cats == val)[0]
                    if len(idx) > 0:
                        result[idx[0]] = 1.0
                    return result.tolist()
                else:
                    return [0.0] * len(cats)
                    
            elif "_emb_" in feat:
                # Text embedding feature - THIS IS THE KEY ISSUE
                base_col = feat.split("_emb_")[0]
                
                if base_col not in self.fitted_encoders:
                    logger.warning(f"Text encoder not found for column: {base_col}")
                    return [0.0]
                
                # ❌ THE MAIN PROBLEM: You shouldn't be extracting individual dimensions here!
                # This method should return the FULL embedding, not just one dimension
                
                try:
                    dim_idx = int(feat.split("_emb_")[1])
                except (IndexError, ValueError):
                    logger.warning(f"Invalid embedding feature name: {feat}")
                    return [0.0]
                
                model = self.fitted_encoders[base_col]
                
                if val and str(val).strip():
                    emb = model.encode([str(val)], show_progress_bar=False)
                    emb_flat = emb.flatten()
                    
                    if dim_idx < len(emb_flat):
                        return [float(emb_flat[dim_idx])]
                    else:
                        logger.warning(f"Dimension index {dim_idx} out of bounds for embedding")
                        return [0.0]
                else:
                    return [0.0]
                    
            elif feat in self.fitted_encoders and isinstance(self.fitted_encoders[feat], dict):
                # Frequency encoding
                freq_map = self.fitted_encoders[feat]
                return [float(freq_map.get(str(val), 0.0))]
                
            elif feat in self.fitted_encoders and hasattr(self.fitted_encoders[feat], 'transform'):
                # Numerical scaling
                scaler = self.fitted_encoders[feat]
                
                if val is None or pd.isna(val):
                    val = 0.0
                
                try:
                    val_float = float(val)
                    arr = np.array([[val_float]])
                    result = scaler.transform(arr).flatten()[0]
                    return [float(result)]
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert value {val} to float for feature {feat}")
                    return [0.0]
            else:
                logger.warning(f"No encoder found for feature: {feat}")
                return [0.0]
                
        except Exception as e:
            logger.error(f"Error encoding feature {feat} with value {val}: {e}")
            return [0.0]

    def _transform_query_generic(self, query, block_filter=None):
        if hasattr(query, 'to_dict'):
            query_dict = query.to_dict()
        else:
            query_dict = query
        
        vec = []
        if block_filter is None:
            feat_list = self.feature_names
        else:
            feat_list = self.block_feature_names[block_filter]
        
        processed_cols = set()
        
        for feat in feat_list:
            if "_oh_" in feat:
                col, _, cat = feat.partition("_oh_")
                if col not in processed_cols:
                    val = query_dict.get(col, None)
                    encoder = self.fitted_encoders.get(col)
                    if encoder:
                        cats = encoder.categories_[0]
                        if val is not None and val in cats:
                            oh_vec = (cats == val).astype(float)
                        else:
                            oh_vec = np.zeros(len(cats))
                        vec.extend(oh_vec.tolist())
                    processed_cols.add(col)
                # ✅ SKIP remaining one-hot features for this column
                continue  # <-- ADD THIS!
                    
            elif "_emb_" in feat:
                base_col = feat.split("_emb_")[0]
                if base_col not in processed_cols:
                    val = query_dict.get(base_col, "")
                    model = self.fitted_encoders.get(base_col)
                    if model and val and str(val).strip():
                        emb = model.encode([str(val)], show_progress_bar=False)
                        vec.extend(emb.flatten().tolist())
                    else:
                        emb_dim = self.text_col_dims.get(base_col, 384)
                        vec.extend([0.0] * emb_dim)
                    processed_cols.add(base_col)
                # ✅ SKIP remaining embedding features for this column  
                continue  # <-- ADD THIS!
                    
            elif feat not in processed_cols:
                val = query_dict.get(feat, None)
                encoded = self._encode_single_feature(feat, val)
                vec.extend(encoded)
                processed_cols.add(feat)
        
        arr = np.array(vec, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def similarity_search_late_fusion(self, query, top_n=5, block_weights=None, 
                                      extra_return_columns=None, ignore_exact_matches=True,
                                      return_explanations=False, adaptive_weights = True):
        if not self.verbose:
            print("🔍 Searching for similar items...")  # ADD THIS LINE
        else:
            logger.info("Late fusion similarity search")
        
        if adaptive_weights:
            block_weights = self._get_adaptive_weights(query, block_weights)
    
        if block_weights is None:
            block_weights = {k: 1.0 for k in self.block_vectors if self.block_vectors[k].shape[1] > 0}
        else:
            block_weights = {k: float(block_weights.get(k, 1.0)) for k in self.block_vectors}

        # Calculate total weight for proportional contribution
        active_blocks = [k for k in self.block_vectors 
                        if self.block_vectors[k].shape[1] > 0 and 
                        k in block_weights and 
                        block_weights.get(k, 1.0) != 0]
        total_weight = sum(block_weights[k] for k in active_blocks)
        
        if total_weight == 0:
            logger.warning("All block weights are zero, returning empty results")
            return self.df.iloc[:0].copy()

        block_sims = []
        block_contributions = {}  # Store individual block similarities for explanation
        n_rows = len(self.df)
        
        for block_type, block_matrix in self.block_vectors.items():
            if block_matrix.shape[1] == 0 or block_type not in block_weights or block_weights.get(block_type, 1.0) == 0:
                continue
            query_vec = self._transform_query_generic(query, block_filter=block_type)
            if query_vec.shape[1] != block_matrix.shape[1]:
                logger.warning(f"[{block_type}] dimension mismatch: block_matrix {block_matrix.shape}, query_vec {query_vec.shape}")
                continue
            sims = cosine_similarity(block_matrix, query_vec).flatten()
            '''
            if block_type == 'text':
                sims_norm = sims  # No normalization for text
            else:
            '''
            sim_min, sim_max = np.min(sims), np.max(sims)
            if sim_max > sim_min:
                sims_norm = (sims - sim_min) / (sim_max - sim_min)
            else:
                    sims_norm = np.zeros_like(sims)
            
            # Proportional weighting
            contribution_weight = block_weights[block_type] / total_weight
            weighted_sims = contribution_weight * sims_norm
            block_sims.append(weighted_sims)
            
            # Store for explanations
            if return_explanations:
                block_contributions[block_type] = {
                    'raw_similarities': sims,
                    'normalized_similarities': sims_norm,
                    'weighted_similarities': weighted_sims,
                    'contribution_weight': contribution_weight
                }
            
            logger.info(f"[{block_type}] weight proportion: {contribution_weight:.3f}")

        if block_sims:
            sims_total = np.sum(block_sims, axis=0)
        else:
            sims_total = np.zeros(n_rows)

        idx_all = np.argsort(-sims_total)
        if ignore_exact_matches:
            similarity_cap = 1  # Max possible score is 1.0 with proportional weighting
            filtered = [(i, s) for i, s in zip(idx_all, sims_total[idx_all]) if s < similarity_cap]
            idx = [i for i, _ in filtered][:top_n]
        else:
            idx = idx_all[:top_n]
        
        out_df = self.df.iloc[idx].copy()
        out_df['similarity'] = sims_total[idx]
        
        # Add block-level similarity scores for explanation
        if return_explanations:
            for block_type in active_blocks:
                out_df[f'{block_type}_similarity'] = block_contributions[block_type]['weighted_similarities'][idx]

        # Always include index/id/date/ignored/extra columns in output
        always_return = set()
        if self.index_column:
            always_return.add(self.index_column)
        if self.id_column:
            always_return.add(self.id_column)
        always_return |= self.ignore_columns
        if extra_return_columns:
            always_return |= set(extra_return_columns)
        for col in always_return:
            if col and col not in out_df.columns and col in self.df.columns:
                out_df[col] = self.df[col].iloc[idx].values

        # LOG SUMMARY STATISTICS across all results
        # LOG SUMMARY STATISTICS across all results
        logger.info("=== TOP RESULTS SUMMARY ===")
        for block_type in active_blocks:
            if return_explanations:
                weighted_contribs = block_contributions[block_type]['weighted_similarities'][idx]
            else:
                # Find the corresponding block_sims array
                block_idx = list(active_blocks).index(block_type)
                weighted_contribs = block_sims[block_idx][idx] if block_idx < len(block_sims) else np.zeros(len(idx))
            
            avg_contrib = np.mean(weighted_contribs)
            max_contrib = np.max(weighted_contribs)
            min_contrib = np.min(weighted_contribs)
            contribution_weight = block_weights[block_type] / total_weight
            
            logger.info(f"[{block_type}] avg: {avg_contrib:.4f}, max: {max_contrib:.4f}, min: {min_contrib:.4f} (weight: {contribution_weight:.3f})")
        
        logger.info("Late fusion search done, returning %d results", len(idx))
        return out_df

    def explain_similarity(self, query, company_name_or_index, block_weights=None):
        """
        Explain why a specific company was similar to the query by showing
        block-level contributions and suggesting weight adjustments.
        """
        results = self.similarity_search_late_fusion(
            query, top_n=50, block_weights=block_weights, return_explanations=True
        )
        
        if isinstance(company_name_or_index, str):
            # Find by company name
            match = results[results['name'] == company_name_or_index]
            if match.empty:
                return f"Company '{company_name_or_index}' not found in top 50 results"
            row = match.iloc[0]
        else:
            # Use by index position in results
            if company_name_or_index >= len(results):
                return f"Index {company_name_or_index} out of range (only {len(results)} results)"
            row = results.iloc[company_name_or_index]
        
        explanation = []
        explanation.append(f"Similarity breakdown for: {row['name']}")
        explanation.append(f"Overall similarity: {row['similarity']:.4f}")
        explanation.append("")
        
        # Show block contributions
        active_blocks = ['text', 'numerical', 'categorical']
        for block in active_blocks:
            col_name = f'{block}_similarity'
            if col_name in row:
                contrib = row[col_name]
                percentage = (contrib / row['similarity']) * 100
                explanation.append(f"{block.capitalize()} block: {contrib:.4f} ({percentage:.1f}% of total)")
        
        explanation.append("")
        explanation.append("Suggestions:")
        
        # Find dominant block
        block_contribs = {block: row.get(f'{block}_similarity', 0) for block in active_blocks}
        max_block = max(block_contribs, key=block_contribs.get)
        max_contrib = block_contribs[max_block]
        
        if max_contrib > row['similarity'] * 0.6:  # If one block dominates >60%
            explanation.append(f"• {max_block.capitalize()} block is dominating ({(max_contrib/row['similarity']*100):.1f}%)")
            explanation.append(f"• Consider reducing {max_block} weight if results seem irrelevant")
            explanation.append(f"• Try: block_weights={{'{max_block}': 0.5, 'others': 1.0}}")
        else:
            explanation.append("• Good balance across blocks - no single modality dominating")
        
        return "\n".join(explanation)


    
    def similarity_search(
        self, 
        query: Union[Dict, pd.Series, str], 
        top_n: int = 5, 
        text_column: Optional[str] = None, 
        mode: str = "late_fusion", 
        block_weights: Optional[Dict[str, float]] = None,
        extra_return_columns: Optional[List[str]] = None,
        ignore_exact_matches: bool = True 
    ) -> pd.DataFrame:
        """
            Find the most similar items to the query.
            
            Parameters
            ----------
            query : dict, pd.Series, or str
                Query to search for. Can be:
                - dict: {'column': 'value', 'other_column': 123}
                - str: 'text to search' (requires text_column parameter)
                - pd.Series: pandas series with column names as index
            
            top_n : int, default=5
                Number of most similar items to return
            
            Returns
            -------
            pd.DataFrame
                DataFrame with top_n most similar items, including:
                - All original columns from training data
                - 'similarity' column with similarity scores (0-1)
                
            Examples
            --------
            >>> # Dictionary query
            >>> results = vectorizer.similarity_search(
            ...     {'description': 'AI platform', 'category': 'tech'}, 
            ...     top_n=10
            ... )
            
            >>> # Text-only query  
            >>> results = vectorizer.similarity_search(
            ...     'machine learning startup',
            ...     text_column='description'
            ... )
            
            Raises
            ------
            ModelNotFittedError
                If fit_transform() hasn't been called yet
            ValueError
                If query is empty or invalid
        """
        # Add at the beginning:
        if self.vectors is None:
            raise ModelNotFittedError()

        if top_n <= 0:
            raise InvalidQueryError("top_n must be positive")

        if isinstance(query, dict) and not query:
            raise InvalidQueryError("Query dictionary cannot be empty")

        if isinstance(query, str) and not query.strip():
            raise InvalidQueryError("Query string cannot be empty")
        logger.info("Similarity search (mode=%s)", mode)
        if mode == "late_fusion":
            return self.similarity_search_late_fusion(query, top_n=top_n, block_weights=block_weights,
                                                       extra_return_columns=extra_return_columns, ignore_exact_matches=ignore_exact_matches)
        # EARLY FUSION
        if self.vectors is None:
            logger.error("Vectors not fitted. Call fit_transform first!")
            raise ModelNotFittedError()
        if isinstance(query, str):
            if not text_column:
                text_cols = [col for col, info in self.encoding_report.items() if info.get("type", "").startswith("text")]
                if not text_cols:
                    logger.error("No text columns found for text query.")
                    raise InvalidQueryError("No text columns found for text query.")
                text_column = text_cols[0]
            dim = self.text_col_dims[text_column]
            start = self.feature_names.index(f"{text_column}_emb_0")
            end = start + dim
            model = self.fitted_encoders[text_column]
            query_emb = model.encode([query])
            sims = cosine_similarity(self.vectors[:, start:end], query_emb).flatten()
            idx = np.argsort(-sims)[:top_n+1]
            sims = sims[idx]
        else:
            query_vec = self._transform_query_generic(query)
            query_index = None
            if hasattr(query, 'name') and query.name in self.df_index:
                query_index = np.where(self.df_index == query.name)[0][0]
            if self.vector_db is not None:
                D, I = self.vector_db.search(query_vec.astype(np.float32), top_n+1)
                sims = D[0]
                idx = I[0]
            else:
                sims = cosine_similarity(self.vectors, query_vec).flatten()
                idx_all = np.argsort(-sims)
                #similarity_cap = sum(block_weights.values()) # configurable if you want
                similarity_cap = 1
                filtered = [(i, s) for i, s in zip(idx_all, sims[idx_all]) if s < similarity_cap]
                idx = [i for i, _ in filtered][:top_n]
                sims = [s for _, s in filtered][:top_n]
                
        out_df = self.df.iloc[idx].copy()
        out_df['similarity'] = sims[:top_n]
        # Always include index/id/date/ignored/extra columns in output
        always_return = set()
        if self.index_column:
            always_return.add(self.index_column)
        if self.id_column:
            always_return.add(self.id_column)
        always_return |= self.ignore_columns
        if extra_return_columns:
            always_return |= set(extra_return_columns)
        for col in always_return:
            if col and col not in out_df.columns and col in self.df.columns:
                out_df[col] = self.df[col].iloc[idx].values
        logger.info("Similarity search done, returning %d results", len(out_df))
        return out_df

    def get_encoding_report(self):
        return self.encoding_report

    def get_feature_names(self):
        return self.feature_names

    def get_vectors(self):
        return self.vectors

    def save(self, filepath: str):
        logger.info(f"Saving HybridVectorizer to {filepath}")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        logger.info(f"Loading HybridVectorizer from {filepath}")
        return joblib.load(filepath)
