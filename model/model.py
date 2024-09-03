from transformers import T5ForConditionalGeneration
from transformers import BartConfig, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoConfig

class Model():
    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.special_tokens_dict = {'additional_special_tokens' : self.config['tokenizer']['special_tokens']}

    def getBartTokenizer(self):
        model_name = self.config['model']['bart']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens(self.special_tokens_dict)
        
        return tokenizer
    
    def getT5Tokenizer(self):
        model_name = self.config['model']['t5']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens(self.special_tokens_dict)

        return tokenizer

    def getBartModel(self, tokenizer):
        model_name = self.config['model']['bart']

        bart_config = BartConfig().from_pretrained(model_name)
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_name, config = bart_config)
        self.bart_model.resize_token_embeddings(len(tokenizer))
        
        return self.bart_model.to(self.device)
    
    def getT5Model(self, tokenizer):
        model_name = self.config['model']['t5']

        t5_config = AutoConfig.from_pretrained(model_name)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name, config = t5_config)
        self.t5_model.resize_token_embeddings(len(tokenizer))

        return self.t5_model.to(self.device)
        


    