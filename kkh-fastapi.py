from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextStreamer, GenerationConfig

# 모델과 토크나이저 초기화
model_name = 'davidkim205/komt-mistral-7b-v1'
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

# FastAPI 애플리케이션 초기화
app = FastAPI()

# 인증 토큰 설정
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 요청 본문 정의
class QueryRequest(BaseModel):
    query: str

# 인증 함수
def verify_token(token: str = Depends(oauth2_scheme)):
    if token != "elite12":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# 응답 생성 함수
def gen(x: str) -> str:
    generation_config = GenerationConfig(
        temperature=0.8,
        top_p=0.8,
        top_k=100,
        max_new_tokens=1024,
        early_stopping=True,
        do_sample=True,
    )
    q = f"[INST]{x} [/INST]"
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    result_str = tokenizer.decode(gened[0])

    start_tag = f"\n\n### Response: "
    start_index = result_str.find(start_tag)

    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()
    return result_str

@app.post("/generate")
def generate_response(request: QueryRequest, token: str = Depends(verify_token)):
    response = gen(request.query)
    return {"response": response}

# uvicorn main:app --host 0.0.0.0 --port 30423 --reload