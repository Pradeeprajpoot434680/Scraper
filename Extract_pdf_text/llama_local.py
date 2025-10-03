# #!/usr/bin/env python3
# import os
# import time
# import json
# import pprint
# import weaviate
# from dotenv import load_dotenv

# # LlamaIndex imports
# from llama_index.readers.file import PyMuPDFReader
# from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader, Settings
# from llama_index.core.node_parser import HierarchicalNodeParser
# from llama_index.vector_stores.weaviate import WeaviateVectorStore
# from llama_index.llms.google_genai import GoogleGenAI
# from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# # Weaviate v4 typed helpers
# from weaviate.classes.config import Configure, Property, DataType

# load_dotenv()

# # ---------------- Config ----------------
# INDEX_NAME = "Curaj"
# TEXT_KEY = "text"
# META_KEYS_TO_KEEP = ["file_name"]
# LOCAL_DOC_DIR = "./test"
# BATCH_SIZE = 64  # Weaviate batch insert size

# # If you prefer explicit URL instead of connect_to_local(), set WEAVIATE_URL="http://localhost:8080" in .env
# WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# # LLM & embedding settings (keep your chosen models)
# Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash")
# Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")

# print("[INFO] LLM:", Settings.llm.__dict__.get("model"))
# print("[INFO] Embed model:", getattr(Settings.embed_model, "model_name", type(Settings.embed_model)))

# # ---------------- Main ----------------
# # Option A: helper for local instances
# # client = weaviate.connect_to_local()

# # Option B: explicit client construction (works for both local and remote)
# # client = weaviate.Client(url=WEAVIATE_URL)
# client = weaviate.connect_to_local()
# def run_llama():
#     # quick readiness check
#     if not client.is_ready():
#         raise RuntimeError(f"Weaviate at {WEAVIATE_URL} is not ready. Is the Docker container running?")

#     print("[DEBUG] client type:", type(client))

#     # Use the collections (v4) typed API
#     if not client.collections.exists(INDEX_NAME):
#         # define properties for the typed collection
#         props = [
#             Property(name=TEXT_KEY, data_type=DataType.TEXT),
#             Property(name="file_name", data_type=DataType.TEXT),
#         ]
#         vc = Configure.Vectors.self_provided()  # we will push our own vectors
#         client.collections.create(INDEX_NAME, properties=props, vector_config=vc)
#         print(f"[INFO] Created collection '{INDEX_NAME}' (v4 typed).")
#     else:
#         print(f"[INFO] Collection '{INDEX_NAME}' already exists.")

#     coll = client.collections.get(INDEX_NAME)
#     try:
#         total = coll.aggregate.over_all().total_count
#     except Exception:
#         total = None

#     print("[INFO] collection aggregate count (pre):", total)

#     # Insert only if empty
#     if not total or total == 0:
#         print("[INFO] No objects found — loading local docs, chunking, embedding, and batch-inserting into Weaviate.")
#         reader = SimpleDirectoryReader(input_dir=LOCAL_DOC_DIR, recursive=True)
#         splitter = HierarchicalNodeParser.from_defaults()
#         docs = reader.load_data()
#         print(f"[INFO] Loaded {len(docs)} local documents from {LOCAL_DOC_DIR}")

#         # Prepare docs (clean metadata)
#         for d in docs:
#             d.text_template = "Metadata:\\n{metadata_str}\\n---\\nContent:\\n{content}"
#             for k in list(d.metadata.keys()):
#                 if k not in META_KEYS_TO_KEEP:
#                     continue
#                 v = d.metadata.get(k)
#                 if isinstance(v, (dict, list)):
#                     d.metadata[k] = json.dumps(v)
#                 else:
#                     d.metadata[k] = str(v)
#             d.excluded_embed_metadata_keys = [k for k in d.metadata.keys() if k not in META_KEYS_TO_KEEP]
#             d.excluded_llm_metadata_keys = [k for k in d.metadata.keys() if k not in META_KEYS_TO_KEEP]

#         nodes = splitter.get_nodes_from_documents(docs)
#         print(f"[INFO] Chunked into {len(nodes)} nodes.")

#         # Build texts and metadata arrays
#         texts = []
#         meta_for_text = []
#         for node in nodes:
#             try:
#                 content_text = node.get_content(metadata_mode=None)
#             except Exception:
#                 content_text = getattr(node, "text", "") or ""
#             texts.append(content_text)
#             meta_for_text.append({"file_name": node.metadata.get("file_name", "unknown")})

#         print("[INFO] Computing embeddings ...")
#         embeddings = []
#         chunk_size = 64
#         for i in range(0, len(texts), chunk_size):
#             batch_texts = texts[i : i + chunk_size]

#             # LlamaIndex GoogleGenAIEmbedding expects Document objects for __call__
#             doc_batch = [Document(text=t) for t in batch_texts]
#             raw = Settings.embed_model(doc_batch)

#             for item in raw:
#                 if isinstance(item, dict) and "embedding" in item:
#                     vec = item["embedding"]
#                 elif hasattr(item, "embedding"):
#                     vec = getattr(item, "embedding")
#                 else:
#                     vec = list(item)
#                 # normalize to list of floats
#                 embeddings.append([float(x) for x in list(vec)])

#         print(f"[INFO] Obtained {len(embeddings)} embeddings; sample dim = {len(embeddings[0]) if embeddings else 'N/A'}")

#         print(f"[INFO] Inserting {len(embeddings)} objects into Weaviate in batches of {BATCH_SIZE}...")
#         with coll.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
#             for i, (vec, props, txt) in enumerate(zip(embeddings, meta_for_text, texts)):
#                 properties = {TEXT_KEY: txt, "file_name": props.get("file_name", "unknown")}
#                 try:
#                     batch.add_object(properties=properties, vector=vec)
#                 except Exception as e:
#                     print(f"[WARN] batch.add_object failed for index {i}: {e}")

#         # diagnostics
#         time.sleep(0.6)
#         try:
#             coll = client.collections.get(INDEX_NAME)
#             agg = coll.aggregate.over_all().total_count
#             print("[DIAG] aggregate total_count after insert:", agg)
#             try:
#                 failed = coll.batch.failed_objects
#                 print("[DIAG] collection.batch.failed_objects (len):", len(failed) if failed is not None else None)
#                 if failed:
#                     pprint.pprint(failed[:6])
#             except Exception as e:
#                 print("[DIAG] could not read coll.batch.failed_objects:", e)
#         except Exception as e:
#             print("[WARN] Could not fetch collection diagnostics after insertion:", e)
#     else:
#         print("[INFO] Collection already has objects; skipping insertion.")

#     print("[INFO] Building WeaviateVectorStore wrapper for LlamaIndex...")
#     vector_store = WeaviateVectorStore(weaviate_client=client, index_name=INDEX_NAME, text_key=TEXT_KEY)
#     try:
#         index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
#     except Exception:
#         try:
#             index = VectorStoreIndex(vector_store=vector_store)
#         except Exception as e:
#             print("[ERROR] Could not construct VectorStoreIndex from vector store:", e)
#             raise

#     print("\n--- Starting Query Engine ---")
#     print("Type 'exit' or 'quit' to end the session.")
#     query_engine = index.as_query_engine(similarity_top_k=5)
#     while True:
#         try:
#             query_text = input("\nEnter your query: ")
#         except KeyboardInterrupt:
#             print("\nInterrupted. Exiting.")
#             break
#         if not query_text:
#             continue
#         if query_text.lower() in ("exit", "quit"):
#             print("Goodbye.")
#             break
#         try:
#             response = query_engine.query(query_text)
#             print("\n--- Query Response ---")
#             try:
#                 pprint.pprint(response.response)
#             except Exception:
#                 print(response)
#             print("\n--- Source Nodes ---")
#             if getattr(response, "source_nodes", None):
#                 for node in response.source_nodes:
#                     print(f"Source: {node.metadata.get('file_name', 'N/A')}, Score: {node.score:.4f}")
#             else:
#                 print("No source nodes found.")
#         except Exception as e:
#             print("[ERROR] Query failed:", e)


    

# run_lamma_locally.py
import os
import time
import json
import pprint
import weaviate
from dotenv import load_dotenv

from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from weaviate.classes.config import Configure, Property, DataType

load_dotenv()

# CONFIG
INDEX_NAME = "Curaj"
TEXT_KEY = "text"
META_KEYS_TO_KEEP = ["file_name"]
CLEANED_DIR = "./sorted_data/cleaned_texts"
BATCH_SIZE = 64

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# LLM/embed (kept as-is; embeddings are computed via Google GenAI here)
Settings.llm = GoogleGenAI(model="models/gemini-2.5-flash")
Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")

print("[INFO] LLM:", Settings.llm.__dict__.get("model"))
print("[INFO] Embed model:", getattr(Settings.embed_model, "model_name", type(Settings.embed_model)))

# connect
client = weaviate.connect_to_local(port=8080)

def run_llama():
    if not client.is_ready():
        raise RuntimeError(f"Weaviate at {WEAVIATE_URL} is not ready.")

    # create typed collection if not exists
    if not client.collections.exists(INDEX_NAME):
        props = [
            Property(name=TEXT_KEY, data_type=DataType.TEXT),
            Property(name="file_name", data_type=DataType.TEXT),
        ]
        vc = Configure.Vectors.self_provided()
        client.collections.create(INDEX_NAME, properties=props, vector_config=vc)
        print(f"[INFO] Created collection '{INDEX_NAME}' (v4 typed).")
    else:
        print(f"[INFO] Collection '{INDEX_NAME}' already exists.")

    coll = client.collections.get(INDEX_NAME)
    try:
        total = coll.aggregate.over_all().total_count
    except Exception:
        total = None
    print("[INFO] collection aggregate count (pre):", total)

    # load cleaned docs
    if not os.path.isdir(CLEANED_DIR):
        print(f"[WARN] Cleaned directory {CLEANED_DIR} not found. Nothing to index.")
        return

    # Use SimpleDirectoryReader to load cleaned text files
    reader = SimpleDirectoryReader(input_dir=CLEANED_DIR, recursive=True)
    docs = reader.load_data()
    if not docs:
        print("[WARN] No documents found in cleaned directory.")
        return

    # Chunk
    parser = HierarchicalNodeParser.from_defaults()
    for d in docs:
        d.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"
        d.excluded_embed_metadata_keys = [k for k in d.metadata if k != "file_name"]
        d.excluded_llm_metadata_keys = [k for k in d.metadata if k != "file_name"]

    nodes = parser.get_nodes_from_documents(docs)
    print(f"[INFO] Chunked into {len(nodes)} nodes.")

    # Prepare texts and embeddings via Settings.embed_model
    texts = []
    metas = []
    for node in nodes:
        try:
            txt = node.get_content(metadata_mode=None)
        except Exception:
            txt = getattr(node, "text", "") or ""
        texts.append(txt)
        metas.append({"file_name": node.metadata.get("file_name", "unknown")})

    print("[INFO] Computing embeddings via configured embed_model...")
    embeddings = []
    chunk_size = 64
    for i in range(0, len(texts), chunk_size):
        batch_texts = texts[i: i + chunk_size]
        doc_batch = [Document(text=t) for t in batch_texts]
        raw = Settings.embed_model(doc_batch)
        for item in raw:
            if isinstance(item, dict) and "embedding" in item:
                vec = item["embedding"]
            elif hasattr(item, "embedding"):
                vec = getattr(item, "embedding")
            else:
                vec = list(item)
            embeddings.append([float(x) for x in list(vec)])

    print(f"[INFO] Obtained {len(embeddings)} embeddings; inserting to Weaviate...")

    with coll.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
        for i, (vec, meta, txt) in enumerate(zip(embeddings, metas, texts)):
            properties = {TEXT_KEY: txt, "file_name": meta.get("file_name", "unknown")}
            try:
                batch.add_object(properties=properties, vector=vec)
            except Exception as e:
                print(f"[WARN] failed to add object {i}: {e}")

    time.sleep(0.5)
    try:
        agg = client.collections.get(INDEX_NAME).aggregate.over_all().total_count
        print("[DIAG] total objects after insert:", agg)
    except Exception:
        pass

    # wrap in vector store & query
    vector_store = WeaviateVectorStore(weaviate_client=client, index_name=INDEX_NAME, text_key=TEXT_KEY)
    try:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    except Exception:
        index = VectorStoreIndex(vector_store=vector_store)

    print("\n--- Query Engine Ready ---")
    query_engine = index.as_query_engine(similarity_top_k=5)
    while True:
        q = input("\nEnter your query (or 'exit'): ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        try:
            resp = query_engine.query(q)
            print("\n--- Response ---")
            pprint.pprint(resp.response)
            if getattr(resp, "source_nodes", None):
                print("\n--- Sources ---")
                for node in resp.source_nodes:
                    print(f"{node.metadata.get('file_name', 'N/A')} (score={node.score:.4f})")
        except Exception as e:
            print("[ERROR] query:", e)
