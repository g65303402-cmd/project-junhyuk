"""
rag_query.py — RAG 서버(포트 5002)에 HTTP 요청하는 클라이언트

api_server.py에서 import해서 사용:
    from rag_query import get_rag_context, warmup
"""

import logging
import urllib.request
import json

RAG_SERVER_URL = "http://127.0.0.1:5002/rag"
logger = logging.getLogger(__name__)


def get_rag_context(user_message: str, n_results: int = 3) -> str:
    """
    RAG 서버에 HTTP POST로 유사 대화 예시 요청.
    서버 없거나 오류 시 빈 문자열 반환 (서버 다운 방지).
    """
    try:
        payload = json.dumps({
            "message": user_message,
            "n_results": n_results
        }).encode("utf-8")

        req = urllib.request.Request(
            RAG_SERVER_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("context", "")

    except Exception as e:
        logger.warning("[RAG] 서버 호출 실패 (무시하고 계속): %s", e)
        return ""


def warmup():
    """서버 연결 확인용"""
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:5002/health",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("[RAG] 서버 연결 확인 완료")
    except Exception as e:
        logger.warning("[RAG] 서버 연결 실패: %s", e)
