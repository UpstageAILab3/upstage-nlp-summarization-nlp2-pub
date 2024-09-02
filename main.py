import yaml
import torch
import os

from data_pre.dataset import TrainValidDataset, TestDataset
from model.model import Model
from model.trainer import Trainer
from model.test import Test


class Main:
    def __init__(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        with open('/root/upstage-nlp-summarization-nlp2/config.yaml', "r") as file:
            self.config = yaml.safe_load(file)

        # Model & Tokenizer
        self.model_instance = Model(self.config, self.device)
        self.model = self.model_instance.getModel()
        self.tokenizer = self.model_instance.getTokenizer()

        # dataset
        self.train_dataset = TrainValidDataset(self.config, self.tokenizer, is_train=True)
        self.val_dataset = TrainValidDataset(self.config, self.tokenizer, is_train=False)
        self.test_dataset = TestDataset(self.config, self.tokenizer)

    def trainer(self):
        self.trainer = Trainer(self.config, self.model, self.train_dataset, self.val_dataset, self.tokenizer).get_trainer()
        return self.trainer


    def test(self, model_path):
        test_inst = Test(self.config, self.test_dataset, self.tokenizer, model_path, self.device)
        result_df = test_inst()

        return result_df


if __name__ == "__main__":
    # Model / Tokenizer / dataset / config 초기화
    main = Main()

    # 학습
    trainer = main.trainer()
    trainer.train()

    # 평가
    model_path = '/root/upstage-nlp-summarization-nlp2/results/checkpoint-2340'
    result = main.test(model_path)

