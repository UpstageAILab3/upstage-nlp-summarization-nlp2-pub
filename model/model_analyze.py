from .test import Test
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd

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
            for item in tqdm(dataloader):
                item_text = self.id2text(item['input_ids'])
                generated_ids = model.generate(input_ids = item['input_ids'],
                                               no_repeat_ngram_size = self.config['test']['no_repeat_ngram_size'],
                                               early_stopping = self.config['test']['early_stopping'],
                                               max_length = self.config['test']['generate_max_length'],
                                               num_beams = self.config['test']['num_beams'])
                item_generated_text = self.id2text(generated_ids)
                item_label = item['labels'] 

                input_text.append(item_text)
                generated_text.append(item_generated_text)
                label.append(item_label)

        output = pd.DataFrame({"input_text" : input_text,
                               "generated_text" : generated_text,
                               "label" : label})
        return output
    
    def compute_metric(self):
        pass

    def model_path_to_model(self, model_path):
        model = BartForConditionalGeneration.from_pretrained(model_path)
        model.resize_token_embeddings(len(self.tokenizer))
        return model.to(self.device)
        

    def id2text(self, id_data):
        summary_text = self.tokenizer.decode(id_data)
        remove_tokens = self.config['test']['remove_tokens']

        preprocessed_text = [word for word in summary_text.split() 
                             if word not in remove_tokens]

        return " ".join(preprocessed_text)


        

    