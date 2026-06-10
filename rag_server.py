"""
rag_server.py — RAG 검색 전용 Flask 서버 (포트 5002)

실행: /usr/bin/python3 rag_server.py
"""

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
from flask import Flask, request, jsonify
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

app = Flask(__name__)

# ── 싱글톤 컬렉션 ──────────────────────────────────────────
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        print("[RAG Server] 임베딩 모델 로딩 중...")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(
            name="junhyuk_rag",
            embedding_function=ef
        )
        print("[RAG Server] 로딩 완료")
    return _collection


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/rag", methods=["POST"])
def rag_search():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    n_results = data.get("n_results", 3)

    if not user_message:
        return jsonify({"context": ""})

    try:
        collection = get_collection()
        results = collection.query(
            query_texts=[user_message],
            n_results=n_results,
            include=["metadatas", "distances"]
        )

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not metadatas:
            return jsonify({"context": ""})

        lines = ["\n\n[참고 대화 예시 — 아래 예시의 말투와 구조를 참고해서 응답해]"]
        for i, (meta, dist) in enumerate(zip(metadatas, distances)):
            if dist > 0.5:
                continue
            lines.append(f"\n예시{i+1} [{meta.get('category', '')}]")
            lines.append(f"사용자: {meta['user']}")
            lines.append(f"준혁: {meta['assistant']}")

        if len(lines) == 1:
            return jsonify({"context": ""})

        return jsonify({"context": "\n".join(lines)})

    except Exception as e:
        print(f"[RAG Server] 검색 오류: {e}")
        return jsonify({"context": ""})


if __name__ == "__main__":
    # 시작 시 미리 로딩
    get_collection()
    app.run(host="0.0.0.0", port=5002, debug=False)
