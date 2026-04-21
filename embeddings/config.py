MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

#FAISS parameters
INDEX_TYPE = "IVF"  # options: FLAT, IVF
N_LIST = 100        # number of clusters for IVF
N_PROBE = 10        # clusters to search at query time

#Embedding settings
BATCH_SIZE = 32

#Paths
FAISS_INDEX_PATH = "embeddings/faiss_index/index.bin"
METADATA_PATH = "embeddings/faiss_index/metadata.pkl"