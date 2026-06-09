from flask import Flask, request, jsonify
from voiceclone import load_model, generate_tts

app = Flask(__name__)

# 서버 시작할 때 모델 한 번만 로드
load_model()

@app.route("/tts", methods=["POST"])
def tts():
    data = request.json
    text = data["text"]
    output_path = data.get("output", "voiceclone/output.wav")
    
    generate_tts(test_text=text, output_path=output_path)
    return jsonify({"status": "ok", "output": output_path})

if __name__ == "__main__":
    app.run(port=5001)



# # api_server.py에서 호출 -> api_server.py의  작성하시면 됩니다.
# import httpx

# def run_tts(text: str, output_path: str = "voiceclone/output.wav"):
#     response = httpx.post("http://localhost:5001/tts", json={
#         "text": result["message"],
#         "output": output_path
#     })
#      return result


# # api_server.py 수정하기 -> /api/chat에 tts 서버 연결하기
# @app.post("/api/chat")
# def api_chat(req: ChatRequest):
#     result = generate_response(req.message)
    
#     # TTS 실행
#     tts_response = httpx.post("http://localhost:5001/tts", json={
#         "text": result["message"],
#         "output": f"voiceclone/output.wav"
#     })
    
#     return {
#         "message": result["message"],
#         "emotion": result["emotion"],
#         "audioUrl": "/api/tts/audio/output.wav"  # 프론트에서 이 URL로 재생
#     }




# # **실행 순서**
# # 터미널 1: TTS 서버 먼저 띄우기
# /root/miniconda3/envs/qwen-tts/bin/python voiceclone/tts_server.py

# # 터미널 2: 메인 서버 실행
# python api_server.py