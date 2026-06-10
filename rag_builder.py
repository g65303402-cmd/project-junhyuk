__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
"""
rag_builder.py — ChromaDB 임베딩 빌더 (최초 1회 실행)

실행 방법:
    conda deactivate
    /opt/conda/bin/python3 rag_builder.py

생성 경로: ./chroma_db/
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import json
import os
import chromadb
from chromadb.utils import embedding_functions

# ── 경로 설정 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset_new", "train_final.json")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── 임베딩 모델 ────────────────────────────────────────────
EMBEDDING_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

def build_db():
    print(f"[1/4] 데이터셋 로딩: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"      총 {len(data)}개 로드 완료")

    print(f"[2/4] 임베딩 모델 로딩: {EMBEDDING_MODEL}")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    print(f"[3/4] ChromaDB 초기화: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 기존 컬렉션 있으면 삭제 후 재생성
    try:
        client.delete_collection("junhyuk_rag")
        print("      기존 컬렉션 삭제됨")
    except Exception:
        pass

    collection = client.create_collection(
        name="junhyuk_rag",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    print("[4/4] 임베딩 및 저장 중...")
    ids = []
    documents = []
    metadatas = []

    for i, item in enumerate(data):
        user_text = item.get("user", "").strip()
        assistant_text = item.get("assistant", "").strip()
        category = item.get("category", "기타").strip()

        if not user_text or not assistant_text:
            continue

        ids.append(f"doc_{i}")
        documents.append(user_text)  # user 발화로 검색
        metadatas.append({
            "category": category,
            "user": user_text,
            "assistant": assistant_text
        })

    # 배치 처리 (한 번에 너무 많으면 메모리 부담)
    batch_size = 100
    total = len(ids)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end]
        )
        print(f"      {end}/{total} 저장 완료...")

    print(f"\n✅ 완료! {total}개 임베딩 → {CHROMA_DIR}")
    print(f"   컬렉션 이름: junhyuk_rag")


if __name__ == "__main__":
    build_db()
