# 설치해야 할 라이브러리:
# pip install transformers datasets torch scikit-learn konlpy

import os
import platform
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from datasets import load_dataset
import numpy as np
from konlpy.tag import Okt
import nltk
import re

# NLTK 데이터 다운로드
nltk.download('punkt')

# 전역 상수
os_name = platform.system()
if os_name == 'Windows':
    PRE_PATH = ''
elif os_name == 'Linux':
    PRE_PATH = '/kkh/'
elif os_name == 'Darwin': # 맥
    PRE_PATH = '/kkh/'
PRE_PATH = PRE_PATH + "new_model/"

# 전역 상수 정의
MODEL_NAME = "facebook/bart-base"  # 기본 BART 모델 이름
TOKENIZER_NAME = MODEL_NAME  # 기본 BART 토크나이저 이름
SAVE_DIRECTORY = PRE_PATH + "kobart_model/"  # 모델 저장 디렉토리
BATCH_SIZE = 32  # 배치 사이즈
VAL_SPLIT = 0.1  # 검증 데이터 비율
NUM_EPOCHS = 10  # 학습 에포크 수
LEARNING_RATE = 5e-5  # 학습률
EARLY_STOPPING_PATIENCE = 3  # 얼리스탑을 위한 인내 에포크 수

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1단계: 데이터 전처리
def preprocess_data():
    tokenizer = Okt()

    def process_text(text):
        sentences = re.split(r'[.!?]', text)
        tokenized_sentences = [tokenizer.morphs(sentence.strip()) for sentence in sentences if sentence.strip()]
        return [' '.join(tokens) for tokens in tokenized_sentences]

    # 데이터셋 로딩
    try:
        dataset = load_dataset('nsmc', 'nsmc', trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    processed_data = []
    for sample in dataset['train']:
        text = sample['document']
        processed_sentences = process_text(text)
        processed_data.extend(processed_sentences)
    
    return processed_data

# 2단계: 데이터셋 준비
class KoBartDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        decoder_input_ids = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids
        }

# 3단계: 모델 및 토크나이저 로드
def load_model_and_tokenizer():
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    tokenizer = BartTokenizer.from_pretrained(TOKENIZER_NAME)
    model.to(device)
    return model, tokenizer

# 4단계: 데이터 로더 준비
def prepare_dataloaders(train_dataset, val_dataset, batch_size=BATCH_SIZE):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# 5단계: 모델 학습
def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOPPING_PATIENCE):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_val_loss = float('inf')
    patience_counter = 0

    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                labels=batch["decoder_input_ids"]
            )
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        # 검증 단계
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    labels=batch["decoder_input_ids"]
                )
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_val_loss = total_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}")

        # 얼리스탑 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# 6단계: 모델 저장
def save_model(model, tokenizer, save_directory=SAVE_DIRECTORY):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

# 학습 및 검증 함수
def train_and_validate(processed_data, tokenizer, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, patience=EARLY_STOPPING_PATIENCE):
    if len(processed_data) == 0:
        print("No data available for training.")
        return

    # 데이터 분할
    split_index = int(len(processed_data) * (1 - VAL_SPLIT))
    train_data = processed_data[:split_index]
    val_data = processed_data[split_index:]

    train_dataset = KoBartDataset(train_data, tokenizer)
    val_dataset = KoBartDataset(val_data, tokenizer)

    train_loader, val_loader = prepare_dataloaders(train_dataset, val_dataset, batch_size=BATCH_SIZE)

    model, _ = load_model_and_tokenizer()
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=patience)
    save_model(model, tokenizer)

# Main 함수
def main():
    processed_data = preprocess_data()
    if processed_data:
        tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
        train_and_validate(processed_data, tokenizer)

if __name__ == "__main__":
    main()