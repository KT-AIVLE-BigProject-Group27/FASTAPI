from fastapi import FastAPI, UploadFile, File, HTTPException , WebSocket, WebSocketDisconnect
import os ,sys
import shutil
import uuid
import modularization_v4 as mo
import PDF_to_text as ptt
import fitz
import asyncio

app = FastAPI()

# 임시 폴더 경로
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

sys.path.append(os.path.abspath("./AI/combine"))

# 진행 상태 전송 #
######################################################
async def send_progress(websocket: WebSocket, message: str):
    """WebSocket을 통해 상태 메시지를 전송"""
    if websocket.client_state.name != "CONNECTED":
        return False
    try:
        await websocket.send_json({"status": message})
        return True
    except Exception:
        return False
        


@app.websocket("/api/ws/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket 연결을 유지하고 진행 상태를 전송"""
    await websocket.accept()
    print("✅ WebSocket 연결 성공")

    try:
        steps = [
            "대기 중...",
            "텍스트 추출 중...",
            "계약서를 조항별로 분리 중...",
            "독소 조항을 식별하는 중...",
            "계약서를 요약하는 중...",
            "분석이 완료됐습니다!"
        ]

        for step in steps:
            is_sent = await send_progress(websocket, step)
            if not is_sent:
                break  
            await asyncio.sleep(3)  # 3초마다 상태 업데이트

            # WebSocket 유지용 핑 메시지 추가
            try:
                await websocket.send_json({"ping": "keepalive"})
            except Exception:
                print("⚠️ WebSocket 핑 실패, 연결 종료")
                break

    except WebSocketDisconnect:
        print("❌ WebSocket 연결이 끊어졌습니다.")
    finally:
        print("🔌 WebSocket 세션 종료됨")

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
    txt_output_path = file_path.replace('.pdf', '_temp.txt')

    # 업로드된 파일 임시저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        print("파일이 저장되었습니다.")

    # PDF 텍스트 추출
    #extracted_text = ptt.extract_text_from_pdf(file_path, txt_output_path)
    
    # 모델 초기화 및 파이프라인 실행
    mo.initialize_models()

    results = {
    "indentification_results": [],
    "summary_results": []
}

    async def process_with_progress():
        async for progress in mo.pipline(file_path):
            print(progress)  # ✅ 진행 상태 출력
            yield progress  # 진행 상태를 그대로 반환

    results = {
        "indentification_results": [],
        "summary_results": []
    }

    # 진행 상태 받기 #
    async for progress in process_with_progress():
        print(progress)

        # ✅ "done" 키가 포함된 경우 결과 저장
        if "done" in progress:
            results["indentification_results"] = progress["results"].get("indentification_results", [])
            results["summary_results"] = progress["results"].get("summary_results", [])

    # 임시 저장된 파일 삭제
    os.remove(file_path)
    
    return results
    



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






