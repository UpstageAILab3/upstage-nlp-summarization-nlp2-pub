from typing import Any
import torch
from transformers import BartForConditionalGeneration
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

class Test():
    def __init__(self, config, dataset ,tokenizer, device):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = dataset

        self.best_model = self.get_best_model()

    def get_best_model(self):
        model = BartForConditionalGeneration.from_pretrained(self.model_path)
        model.resize_token_embeddings(len(self.tokenizer))
        return model.to(self.device)
    
    def __call__(self):
        # 데이터셋 
        dataloader = DataLoader(self.dataset, batch_size=self.config['test']['batch_size'])

        # 모델 예측
        summary = []
        text_ids = []
        with torch.no_grad():
            for item in tqdm(dataloader):
                text_ids.extend(item['ID'])
                generated_ids = self.best_model.generate(input_ids = item['input_ids'].to(self.device),
                                                        no_repeat_ngram_size = self.config['test']['no_repeat_ngram_size'],
                                                        early_stopping = self.config['test']['early_stopping'],
                                                        max_length = self.config['test']['generate_max_length'],
                                                        num_beams = self.config['test']['num_beams'])
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

        return output