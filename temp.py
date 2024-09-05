import os
import platform
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from rouge_score import rouge_scorer

# 상단에 변수 정의
MODEL_NAME = "digit82/kobart-summarization"
NUM_BEAMS = 4
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 64
EARLY_STOPPING_PATIENCE = 3
CLIP_GRAD_NORM = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 경로 설정
os_name = platform.system()
if os_name == 'Windows':
    PRE_PATH = ''
elif os_name == 'Linux':
    PRE_PATH = '/kkh/'
elif os_name == 'Darwin':  # 맥
    PRE_PATH = '/kkh/'
DATA_PATH = PRE_PATH + "data/"
OUTPUT_PATH = PRE_PATH + "output/"
PREDICTION_PATH = PRE_PATH + "prediction/"
TRAIN_PATH = DATA_PATH + "train_new2.csv"
VALID_PATH = DATA_PATH + "dev_new2.csv"
TEST_PATH = DATA_PATH + "test.csv"

# 데이터 로딩 및 전처리 함수 정의
def load_data(file_path):
    # 데이터 로딩 및 전처리 코드 (예: pandas 사용)
    import pandas as pd
    from transformers import BartTokenizer
    
    data = pd.read_csv(file_path)
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    # 데이터 전처리 및 토크나이징 (예시)
    return data, tokenizer

# ROUGE 점수 계산 함수 정의
def compute_rouge(preds, targets):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    for pred, target in zip(preds, targets):
        score = scorer.score(target, pred)
        scores['rouge1'] += score['rouge1'].fmeasure
        scores['rouge2'] += score['rouge2'].fmeasure
        scores['rougeL'] += score['rougeL'].fmeasure
    
    # 평균 ROUGE 점수 계산
    num_examples = len(preds)
    scores = {k: v / num_examples for k, v in scores.items()}
    
    return scores

# 모델 훈련 함수 정의
def train_model(train_loader, valid_loader, model, tokenizer, device=DEVICE):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * 3  # 총 훈련 스텝 수 (에포크 수에 따라 조정)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_rouge = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    patience_counter = 0

    for epoch in range(3):  # 에포크 수 (조정 가능)
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()
            scheduler.step()

        # 평가
        rouge_scores = evaluate_model(valid_loader, model, tokenizer, device)
        print(f"Epoch {epoch + 1} ROUGE scores: {rouge_scores}")

        # 얼리 스탑핑 체크
        if all(rouge_scores[k] > best_rouge[k] for k in best_rouge):
            best_rouge = rouge_scores
            patience_counter = 0
            model.save_pretrained(os.path.join(OUTPUT_PATH, f"epoch_{epoch + 1}"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping")
                break

# 모델 평가 함수 정의
def evaluate_model(data_loader, model, tokenizer, device=DEVICE):
    model.eval()
    preds, targets = [], []

    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            summary_ids = model.generate(input_ids, num_beams=NUM_BEAMS, max_length=MAX_OUTPUT_LENGTH, early_stopping=True)

        # 결과 디코딩
        pred = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # For the targets, we need to handle padding properly
        target_ids = labels[0]  # Assuming batch size is 1
        target = tokenizer.decode(target_ids[target_ids != -100].cpu().tolist(), skip_special_tokens=True)  # -100 is used for padding in some datasets

        preds.append(pred)
        targets.append(target)

    # ROUGE 계산
    rouge_scores = compute_rouge(preds, targets)
    return rouge_scores

# 요약 추출 및 생성 함수 정의
def extract_nouns_and_generate_summary(dialogue, model, tokenizer, kkma, device=DEVICE):
    inputs = tokenizer(dialogue, max_length=MAX_INPUT_LENGTH, return_tensors='pt', truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs.input_ids, num_beams=NUM_BEAMS, max_length=MAX_OUTPUT_LENGTH, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# 메인 함수 정의
def main():
    train_data, tokenizer = load_data(TRAIN_PATH)
    valid_data, _ = load_data(VALID_PATH)
    test_data, _ = load_data(TEST_PATH)
    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=8)
    
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    train_model(train_loader, valid_loader, model, tokenizer, DEVICE)
    
    # 테스트 데이터로 요약 생성
    for dialogue in test_data['dialogue']:
        summary = extract_nouns_and_generate_summary(dialogue, model, tokenizer, None, DEVICE)
        print(f"Dialogue: {dialogue}")
        print(f"Summary: {summary}")

if __name__ == "__main__":
    main()
