import yaml
import torch

from model.model import Model

from model.trainer import Trainer
from data_pre.dataset import TrainValidDataset, TestDataset


class Main:
    def __init__(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        with open('/root/upstage-nlp-summarization-nlp2/config.yaml', "r") as file:
            self.config = yaml.safe_load(file)

        # Model & Tokenizer
        self.model_instance = Model(self.config)
        self.model = self.model_instance.getModel()
        self.tokenizer = self.model_instance.getTokenizer()

        # dataset
        self.train_dataset = TrainValidDataset(self.config, self.tokenizer, is_train=True)
        self.val_dataset = TrainValidDataset(self.config, self.tokenizer, is_train=False)
        self.test_dataset = TestDataset(self.config)

        # trainer
        self.trainer = Trainer(self.config, self.model, self.train_dataset, self.val_dataset, self.tokenizer).get_trainer()
        self.trainer.train()

    def __call__(self):
        pass

        

if __name__ == "__main__":
    Main()