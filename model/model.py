from transformers import BartConfig, BartForConditionalGeneration, AutoTokenizer

class Model():
    def __init__(self, config, device):
        self.device = device
        model_name = config['model']['name']

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.special_tokens_dict = {'additional_special_tokens' : config['tokenizer']['special_tokens']}
        self.tokenizer.add_special_tokens(self.special_tokens_dict)

        # model
        bart_config = BartConfig().from_pretrained(model_name)
        
        self.bart_model = BartForConditionalGeneration.from_pretrained(model_name, config = bart_config)
        self.bart_model.resize_token_embeddings(len(self.tokenizer))

    def getModel(self):
        return self.bart_model.to(self.device)
    
    def getTokenizer(self):
        return self.tokenizer
