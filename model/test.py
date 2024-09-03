import torch

from transformers import T5ForConditionalGeneration
from transformers import BartForConditionalGeneration

from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
import re

class Test():
    def __init__(self, config, dataset ,tokenizer, device):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = dataset

    def getBartModel(self, model_path):
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.resize_token_embeddings(len(self.tokenizer))
        return model.to(self.device)
    
    def getT5Model(self, model_path):
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model.resize_token_embeddings(len(self.tokenizer))
        return model.to(self.device)
    
    def testModel(self, model, model_name):
        # 데이터셋 
        dataloader = DataLoader(self.dataset, batch_size=self.config['test']['batch_size'])

        # 모델 예측
        summary = []
        text_ids = []
        with torch.no_grad():
            for item in tqdm(dataloader):
                text_ids.extend(item['ID'])
                
                generated_ids = model.generate(
                    input_ids = item['input_ids'].to(self.device),
                    no_repeat_ngram_size = self.config['test']['no_repeat_ngram_size'],
                    early_stopping = self.config['test']['early_stopping'],
                    max_length = self.config['test']['generate_max_length'],
                    num_beams = self.config['test']['num_beams']
                )

                for ids in generated_ids:
                    summarized_text = self.tokenizer.decode(ids)
                    summary.append(summarized_text) 

        # 불필요한 토큰 삭제
        remove_tokens = self.config['test']['remove_tokens']
        preprocessed_summary = summary.copy()

        for token in remove_tokens:
            preprocessed_summary = [sentence.replace(token, " ") for sentence in preprocessed_summary]

        # 결과 dataframe 형태로 반환
        output = pd.DataFrame(
            {
                "fname": text_ids,
                "summary": preprocessed_summary
            }
        )

        # 결과 csv 저장
        self.save_result(output, model_name)
    

    def post_process_summary(self, summary):
        # 1. '#'과 조사 사이의 띄어쓰기 제거
        summary = re.sub(r'#([A-Za-z0-9_]+)#\s+(은|는|이|가|을|를|에|에게|의|로|으로)', r'#\1#\g<2>', summary)
        
        # 2. 의미 없는 큰 따옴표 제거
        summary = summary.replace('"', '')
        
        # 3. 문장 맨 앞의 들여쓰기 제거
        summary = summary.strip()
        return summary
    
    
    def save_result(self, result_df, model_name):
        # 전처리
        result_df['summary'] = result_df['summary'].apply(self.post_process_summary)

        # 파일 저장
        file_cnt = 0
        model_name = model_name.split('/')[-1]
        file_name = os.path.join(self.config['path']['submit_dir'], model_name)
        
        while os.path.exists(file_name + '.csv'):
            file_cnt += 1
            file_name = f"{file_name}({file_cnt})"


        file_name += '.csv'
        result_df.to_csv(file_name, index=False)

    
