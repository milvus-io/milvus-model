from scipy.sparse import csr_array, vstack

def stack_sparse_embeddings(sparse_embs):
    return vstack([sparse_emb.reshape((1,-1)) for sparse_emb in sparse_embs])

