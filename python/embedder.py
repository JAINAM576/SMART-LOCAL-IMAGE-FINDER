import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import faiss
import pickle
import os
from typing import List, Tuple, Dict, Optional
import json
from loadmodelinitially import embed_onnx_model,sentance_model

import logging
import transformers

transformers.utils.logging.set_verbosity_error()

# Global variables to store loaded models and paths
_session = None
_tokenizer = None
_db_path = None
_embedding_dim = None



def initialize_system(onnx_model_path: str, tokenizer_path: str, db_path: str = "./embedding_db"):
    """
    Initialize the embedding system with ONNX model and database path
    
    Args:
        onnx_model_path: Path to ONNX model file
        tokenizer_path: Path to tokenizer directory
        db_path: Directory to store FAISS index and mappings
    """
    global _session, _tokenizer, _db_path, _embedding_dim
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    # Load ONNX model and tokenizer
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session = ort.InferenceSession(onnx_model_path,sess_options=sess_opts, providers=providers)
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    _db_path = db_path
    
    # Create database directory
    os.makedirs(db_path, exist_ok=True)
    
    # Get embedding dimension
    _embedding_dim = _get_embedding_dimension()
    
def _get_embedding_dimension() -> int:
    """Get embedding dimension from model"""
    dummy_input = _tokenizer("test", return_tensors="np", padding=True, truncation=True)
    output = _session.run(None, {
        'input_ids': dummy_input['input_ids'].astype(np.int64),
        'attention_mask': dummy_input['attention_mask'].astype(np.int64)
    })[0]
    return output.shape[-1]

def _get_paths():
    """Get database file paths"""
    return {
        'index': os.path.join(_db_path, "faiss_index.bin"),
        'mapping': os.path.join(_db_path, "hex_mapping.pkl"),
        'metadata': os.path.join(_db_path, "metadata.json")
    }

def _load_faiss_index() -> faiss.Index:
    """Load existing FAISS index or create new one"""
    paths = _get_paths()
    if os.path.exists(paths['index']):
        return faiss.read_index(paths['index'])
    else:
        # Create new index (Inner Product for cosine similarity)
        return faiss.IndexFlatIP(_embedding_dim)

def _save_faiss_index(index: faiss.Index):
    """Save FAISS index to disk"""
    paths = _get_paths()
    faiss.write_index(index, paths['index'])

def _load_mapping() -> Dict[str, int]:
    """Load hex code to index mapping"""
    paths = _get_paths()
    if os.path.exists(paths['mapping']):
        with open(paths['mapping'], 'rb') as f:
            return pickle.load(f)
    else:
        return {}

def _save_mapping(hex_to_idx: Dict[str, int]):
    """Save hex code mapping to disk"""
    paths = _get_paths()
    with open(paths['mapping'], 'wb') as f:
        pickle.dump(hex_to_idx, f)

def _save_metadata(total_embeddings: int):
    """Save metadata"""
    paths = _get_paths()
    metadata = {
        'total_embeddings': total_embeddings,
        'embedding_dim': _embedding_dim,
        'last_updated': str(np.datetime64('now'))
    }
    with open(paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)


def encode_text(text: str) -> np.ndarray:
    """
    Encode single text to embedding
    
    Args:
        text: Input text
        
    Returns:
        Normalized embedding vector
    """
    if _session is None or _tokenizer is None:
        raise RuntimeError("System not initialized. Call initialize_system() first.")
    
    # Tokenize
    encoded = _tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np"
    )
    
    # Run inference
    model_output = _session.run(
        None, 
        {
            'input_ids': encoded['input_ids'].astype(np.int64),
            'attention_mask': encoded['attention_mask'].astype(np.int64)
        }
    )[0]
    
    # Mean pooling
    attention_mask = encoded['attention_mask']
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, model_output.shape).astype(np.float32)
    
    sum_embeddings = np.sum(model_output * input_mask_expanded, axis=1)
    sum_mask = np.sum(input_mask_expanded, axis=1)
    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
    
    sentence_embedding = sum_embeddings / sum_mask
    
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(sentence_embedding, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized_embedding = sentence_embedding / norms
    
    return normalized_embedding[0]  # Remove batch dimension

def encode_batch(texts: List[str]) -> np.ndarray:
    """
    Encode multiple texts to embeddings
    
    Args:
        texts: List of input texts
        
    Returns:
        Array of normalized embeddings
    """
    embeddings = []
    for text in texts:
        embedding = encode_text(text)
        embeddings.append(embedding)
    return np.array(embeddings)

def add_embedding(file_path: str, text: str) -> str:
    """
    Add embedding to FAISS index with file path mapping
    """
    if _session is None:
        raise RuntimeError("System not initialized. Call initialize_system() first.")

    # Load current state
    index = _load_faiss_index()
    file_to_idx = _load_mapping()
    idx_to_file = {v: k for k, v in file_to_idx.items()}
    next_idx = len(file_to_idx)

    # Check if already exists
    if file_path in file_to_idx:
        print(f"File {file_path} already exists. Skipping...")
        return file_path

    # Generate embedding
    embedding = encode_text(text)

    # Add to FAISS index
    index.add(embedding.reshape(1, -1).astype(np.float32))

    # Update mappings
    file_to_idx[file_path] = next_idx

    # Save everything
    _save_faiss_index(index)
    _save_mapping(file_to_idx)
    _save_metadata(index.ntotal)

    print(f"Added embedding for {file_path}")
    return file_path

def add_batch_embeddings(file_paths: List[str], texts: List[str]) -> List[str]:
    """
    Add multiple embeddings in batch
    """
    if _session is None:
        raise RuntimeError("System not initialized. Call initialize_system() first.")

    if len(file_paths) != len(texts):
        raise ValueError("file_paths and texts must have same length")

    # Load current state
    index = _load_faiss_index()
    file_to_idx = _load_mapping()
    next_idx = len(file_to_idx)

    added_files = []
    embeddings = []

    for file_path, text in zip(file_paths, texts):
        if file_path in file_to_idx:
            print(f"Skipping {file_path} - already exists")
            continue

        embedding = encode_text(text)
        embeddings.append(embedding)
        added_files.append(file_path)

    if embeddings:
        # Add all embeddings to index
        embeddings_array = np.array(embeddings).astype(np.float32)
        index.add(embeddings_array)

        # Update mappings
        for file_path in added_files:
            file_to_idx[file_path] = next_idx
            next_idx += 1

        # Save everything
        _save_faiss_index(index)
        _save_mapping(file_to_idx)
        _save_metadata(index.ntotal)

        print(f"Added {len(added_files)} embeddings")

    return added_files


def search_similar(query_text: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Search for similar embeddings
    """
    if _session is None:
        raise RuntimeError("System not initialized. Call initialize_system() first.")

    # Load current state
    index = _load_faiss_index()
    file_to_idx = _load_mapping()
    idx_to_file = {v: k for k, v in file_to_idx.items()}

    if index.ntotal == 0:
        return []

    # Generate query embedding
    query_embedding = encode_text(query_text)

    # Search
    scores, indices = index.search(
        query_embedding.reshape(1, -1).astype(np.float32),
        min(k, index.ntotal)
    )

    # Convert to file paths
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx in idx_to_file:
            file_path = idx_to_file[idx]
            results.append((file_path, float(score)))

    return results

def get_embedding_by_file_path(file_path: str) -> Optional[np.ndarray]:
    """
    Get embedding by file path
    """
    if _session is None:
        raise RuntimeError("System not initialized. Call initialize_system() first.")

    # Load current state
    index = _load_faiss_index()
    file_to_idx = _load_mapping()

    if file_path not in file_to_idx:
        return None

    idx = file_to_idx[file_path]
    embedding = index.reconstruct(idx)
    return embedding


def remove_embedding(file_path: str) -> bool:
    """
    Remove embedding by file path (logical deletion)
    """
    if _session is None:
        raise RuntimeError("System not initialized. Call initialize_system() first.")

    # Load current mappings
    file_to_idx = _load_mapping()

    if file_path not in file_to_idx:
        return False

    # Remove from mappings
    del file_to_idx[file_path]

    # Save mappings
    _save_mapping(file_to_idx)

    # Update metadata
    index = _load_faiss_index()
    _save_metadata(len(file_to_idx))

    print(f"Removed embedding {file_path}")
    return True


def get_stats() -> Dict:
    """Get database statistics"""
    if _db_path is None:
        return {"error": "System not initialized"}
    
    paths = _get_paths()
    
    # Load current state
    index = _load_faiss_index()
    hex_to_idx = _load_mapping()
    
    return {
        'total_embeddings': index.ntotal,
        'active_mappings': len(hex_to_idx),
        'embedding_dimension': _embedding_dim,
        'index_size_mb': os.path.getsize(paths['index']) / (1024*1024) if os.path.exists(paths['index']) else 0,
        'mapping_size_mb': os.path.getsize(paths['mapping']) / (1024*1024) if os.path.exists(paths['mapping']) else 0
    }

def list_all_hex_codes() -> List[str]:
    """Get all hex codes in database"""
    if _session is None:
        raise RuntimeError("System not initialized. Call initialize_system() first.")
    
    hex_to_idx = _load_mapping()
    return list(hex_to_idx.keys())

def clear_database():
    """Clear entire database (use with caution!)"""
    if _db_path is None:
        raise RuntimeError("System not initialized. Call initialize_system() first.")
    
    paths = _get_paths()
    
    # Remove files
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)
    
    print("Database cleared")



def make_embedd(image_path,caption):

    print("Start Embedding Conversion....")
    
    # Initialize
    initialize_system(embed_onnx_model, sentance_model, "../assests/test_db")
 
    add_embedding(image_path, caption)
    
  


def testing_similarity(query,k):
    k=int(k)
    
    initialize_system(embed_onnx_model, sentance_model, "../assests/test_db")

    results = search_similar(query, k=k)
    return results