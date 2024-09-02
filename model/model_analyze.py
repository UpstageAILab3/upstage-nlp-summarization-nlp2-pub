from .test import Test
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd
from rouge import Rouge

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


        

    