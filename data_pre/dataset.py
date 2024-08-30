from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class TrainValidDataset(Dataset):
    def __init__(self, config, tokenizer, is_train):
        self.config = config
        self.tokenizer = tokenizer

        if is_train:
            self.df = pd.read_csv(self.config['path']['train_csv'])
        else:
            self.df = pd.read_csv(self.config['path']['dev_csv'])

        self.encoder_input = self.df['dialogue'].tolist()
        self.decoder_input = self.df['summary'].apply(lambda x : config['tokenizer']['start_token'] + str(x)).tolist()
        self.decoder_output = self.df['summary'].apply(lambda x : str(x) + config['tokenizer']['end_token']).tolist()
                           
        self.tokenized_encoder_input = self.get_tokenized_data(self.encoder_input)
        self.tokenized_decoder_input = self.get_tokenized_data(self.decoder_input)
        self.tokenized_decoder_output = self.get_tokenized_data(self.decoder_output)

    def get_tokenized_data(self, list_data):
        return self.tokenizer(list_data,
                            return_tensors = "pt",
                            padding = True,
                            add_special_tokens = True,
                            truncation = True,
                            max_length = self.config['tokenizer']['max_length'],
                            return_token_type_ids = False)


    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_encoder_input.items()}
        item2 = {key : val[idx].clone().detach() for key, val in self.tokenized_decoder_input.items()}

        item['decoder_input_ids'] = item2['input_ids']
        item['decoder_attention_mask'] = item2['attention_mask']

        item['labels'] = self.tokenized_decoder_output['input_ids'][idx]

        return item
    
    def __len__(self):
        return len(self.df)

class TestDataset(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.df = pd.read_csv(config['path']['test_csv'])
        
        self.encoder_input = self.df['dialogue'].tolist()
        self.id =  self.df['fname'].tolist()

        self.tokenized_encoder_input = self.get_tokenized_data(self.encoder_input)

    def get_tokenized_data(self, list_data):
        return self.tokenizer(list_data,
                            return_tensors = "pt",
                            padding = True,
                            add_special_tokens = True,
                            truncation = True,
                            max_length = self.config['tokenizer']['max_length'],
                            return_token_type_ids = False)
    
    def __getitem__(self, idx):
        item = {key : val[idx].clone().detach() for key, val in self.tokenized_encoder_input.items()}
        item['ID'] = self.id[idx]
        return item
    
    def __len__(self):
        return len(self.df)
