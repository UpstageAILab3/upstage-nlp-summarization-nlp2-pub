from .test import Test
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from rouge import Rouge
import os
import re

from transformers import BartForConditionalGeneration

class ModelAnalyze():
    def __init__(self, config, tokenizer, device):
        self.config = config
        self.tokenizer = tokenizer
        self.device = device


    def get_result(self, dataset, model_path):
        """
        INPUT:
            dataset : ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels']

        OUTPUT:
            DataFrame : ['input_text', 'generated_text', 'label'] 
        """

        dataloader = DataLoader(dataset, self.config['test']['batch_size'])
        model = self.model_path_to_model(model_path)

        input_text = []
        generated_text = []
        label = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_text = self.id2text(batch['input_ids'])
                generated_ids = model.generate(input_ids = batch['input_ids'].to(self.device),
                                               no_repeat_ngram_size = self.config['test']['no_repeat_ngram_size'],
                                               early_stopping = self.config['test']['early_stopping'],
                                               max_length = self.config['test']['generate_max_length'],
                                               num_beams = self.config['test']['num_beams'])
                generated_batch_text = self.id2text(generated_ids)
                batch_label = self.id2text(batch['labels'])
                
                for item_text, generated_item_text, item_label in zip(batch_text, generated_batch_text, batch_label):
                    input_text.append(item_text)
                    generated_text.append(generated_item_text)
                    label.append(item_label)

        output = pd.DataFrame({"input_text" : input_text,
                               "generated_text" : generated_text,
                               "label" : label})
        
        # 평가 지표
        output = self.compute_metric(output)

        # 저장
        model_name = self.config['model']['select_model']
        self.save_file(output, model_name)

        return output
    
    def compute_metric(self, output_df):
        df = output_df.copy()
        rouge = Rouge()

        predictions = df['generated_text']
        label = df['label']

        # 개별 데이터 점수
        results = rouge.get_scores(predictions, label, avg = False)
        results_data = []
        for score in results:
            rouge_1 = score['rouge-1']['f']
            rouge_2 = score['rouge-2']['f']
            rouge_l = score['rouge-l']['f']
            rouge_mean = (rouge_1 + rouge_2 + rouge_l) / 3

            results_data.append({'rouge_1' : f"{rouge_1:.4f}", 'rouge_2': f"{rouge_2:.4f}", 'rouge_l' : f"{rouge_l:.4f}", 'rouge_mean' : f'{rouge_mean:.4f}'})

        results_df = pd.DataFrame(results_data)

        # 데이터 + 개별 데이터 평가
        df = pd.concat([df, results_df], axis=1)
        df = df.sort_values(by = 'rouge_mean')

        # 평균 데이터 추가
        avg_results = rouge.get_scores(predictions, label, avg = True)
        rouge_mean = (avg_results['rouge-1']['f'] + avg_results['rouge-2']['f'] + avg_results['rouge-l']['f']) / 3
        avg_row = {
            'input_text': 'AVERAGE',
            'generated_text': 'AVERAGE',
            'label': 'AVERAGE',
            'rouge_1': avg_results['rouge-1']['f'],
            'rouge_2': avg_results['rouge-2']['f'],
            'rouge_l': avg_results['rouge-l']['f'],
            'rouge_mean': rouge_mean
        }
        df = pd.concat([df, pd.DataFrame([avg_row])], axis=0)

        return df

    def model_path_to_model(self, model_path):
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.resize_token_embeddings(len(self.tokenizer))
        return model.to(self.device)
        

    def id2text(self, id_batch):
        batch_text = []
        remove_tokens = self.config['test']['remove_tokens']

        id_batch[id_batch == -100] = self.tokenizer.pad_token_id
        decoded_batch = self.tokenizer.batch_decode(id_batch, clean_up_tokenization_spaces = True)

        for token in remove_tokens:
            batch_text = [sentence.replace(token, ' ') for sentence in decoded_batch]

        return batch_text
    
    def save_file(self, result_df, model_name):
        # 전처리
        result_df['input_text'] = result_df['input_text'].apply(self.post_process)
        result_df['generated_text'] = result_df['generated_text'].apply(self.post_process)
        result_df['label'] = result_df['label'].apply(self.post_process)

        # 파일 저장
        file_cnt = 0
        model_name = model_name.split('/')[-1]
        file_name = os.path.join(self.config['path']['valid_dir'], model_name)
        
        while os.path.exists(file_name + '.csv'):
            file_name = re.sub(r'\(\d+\)', '', file_name)
            file_cnt += 1
            file_name = f"{file_name}({file_cnt})"

        file_name += '.csv'
        result_df.to_csv(file_name, index=False)

    def post_process(self, summary):
        # 1. '#'과 조사 사이의 띄어쓰기 제거
        summary = re.sub(r'#([A-Za-z0-9_]+)#\s+(은|는|이|가|을|를|에|에게|의|로|으로)', r'#\1#\g<2>', summary)
        
        # 2. 의미 없는 큰 따옴표 제거
        summary = summary.replace('"', '')
        
        # 3. 문장 맨 앞의 들여쓰기 제거
        summary = summary.strip()
        return summary
    


        

    