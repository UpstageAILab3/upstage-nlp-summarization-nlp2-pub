from transformers import BartConfig, BartForConditionalGeneration, AutoTokenizer

class Model():
    def __init__(self, config):
        model_name = config['model']['name']
        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.special_tokens_dict = {'additional_special_tokens' : config['tokenizer']['special_tokens']}
        self.tokenizer.add_special_tokens(self.special_tokens_dict)

        # model
        bart_config = BartConfig().from_pretrained(model_name)
        
        bart_model = BartForConditionalGeneration.from_pretrained(model_name, config = bart_config)
        self.model = bart_model.resize_token_embeddings(len(self.tokenizer))

    def getModel(self):
        return self.model
    
    def getTokenizer(self):
        return self.tokenizer
