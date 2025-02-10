from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse 
import os ,sys
import shutil
import uuid
import modularization_v4 as mo

import fitz
import asyncio

app = FastAPI()

# 임시 폴더 경로
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)
sys.path.append(os.path.abspath("./AI/combine"))

# 현재 연결된 WebSocket 클라이언트 저장
websocket_connections = set()

@app.websocket("/api/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket을 통해 실시간 진행 상태를 수신"""
    await websocket.accept()
    websocket_connections.add(websocket)
    print("✅ WebSocket 연결됨")

    try:
        while True:
            await asyncio.sleep(1)  # Keep connection alive
    except WebSocketDisconnect:
        print("❌ WebSocket 연결 종료")
    finally:
        websocket_connections.remove(websocket)  # 연결이 끊어지면 제거

#########################################################
#파일 업로드 및 텍스트 변환
#POST /api/upload → HWP 파일 업로드 및 텍스트 변환
@app.post("/api/upload/")
async def upload_file(file: UploadFile = File(...)):
    # 업로드된 파일이 PDF인지 확인
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")
 
    # 임시 파일 경로 생성
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(TEMP_DIR, unique_filename)

    # 임시 텍스트 파일 경로 생성
    #txt_output_path = file_path.replace('.pdf', '_temp.txt')

    # 업로드된 파일 임시저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        print("파일이 저장되었습니다.")

    
    # 모델 초기화 및 파이프라인 실행
    mo.initialize_models()

    # ✅ WebSocket을 통해 진행 상태 전송
    async def process_with_progress():
        async for progress in mo.pipline(file_path, websocket_connections):
            yield progress

    results = {
        "indentification_results": [],
        "summary_results": []
    }

    async for progress in process_with_progress():  # ✅ 이렇게 실행해야 함
        print(progress)
        for ws in list(websocket_connections):
            try:
                await ws.send_json(progress)
                if "done" in progress:
                    # ✅ progress["results"]를 참조하는 게 아니라 직접 키값을 참조해야 함
                    results["indentification_results"] = progress.get("indentification_results", [])
                    results["summary_results"] = progress.get("summary_results", [])
            except Exception as e:
                print(f"⚠️ WebSocket 전송 실패: {e}")
                websocket_connections.remove(ws)  # 오류 발생 시 제거


    # 임시 저장된 파일 삭제
    os.remove(file_path)
   
    return JSONResponse(content=results)
    



#요약 요청
#POST /api/summarize → 변환된 텍스트 요약
#@app.post("/upload/")

'''
return article_number: int
    # clause_number : int
    # subclause_number: int
    article_content: str

'''

#독소조항 판별 요청
#POST /api/check-unfair-clauses → 변환된 텍스트 독소조항 판별
#@app.post("/upload/")

'''
return {article_number: int
    clause_number : int
    subclause_number: int                        
    Unfair: str
    Toxic: str
    explain: str}
''' 

#전체 처리 (필요한 경우 통합 요청, 선택적 제공)
#POST /api/full-analysis → 파일 업로드부터 요약 및 독소조항 판별까지 한 번에 처리






