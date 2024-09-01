from .test import Test
from torch.util.data import DataLoader
import torch
from tqdm import tqdm
import pandas as pd

class Valid():
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer


    def get_result(self, dataset, model):
        """
        INPUT:
            dataset : ['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'labels']

        OUTPUT:
            DataFrame : ['input_text', 'generated_text', 'label'] 
        """
        dataloader = DataLoader(dataset, self.config['test']['batch_size'])

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
        

    def id2text(self, id_data):
        summary_text = self.tokenizer.decoce(id_data)
        remove_tokens = self.config['test']['remove_tokens']

        preprocessed_text = [word for word in summary_text.split() 
                             if word not in remove_tokens]

        return " ".join(preprocessed_text)


        

    