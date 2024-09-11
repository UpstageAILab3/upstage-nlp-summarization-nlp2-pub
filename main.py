import yaml
import torch
import gc
import os
import random

from model.model import Model
from model.trainer import Trainer
from model.test import Test
from model.model_analyze import ModelAnalyze

from data_pre.dataset import TrainValidDataset, TestDataset, T5TrainValidDataset, T5TestDataset

import pandas as pd
pd.set_option('display.max_colwidth', None)

import warnings
warnings.filterwarnings('ignore')

class Main:
    def __init__(self) -> None:
        self.clear_cuda_memory()
        self.set_seed()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        with open('/root/upstage-nlp-summarization-nlp2/config.yaml', "r") as file:
            self.config = yaml.safe_load(file)

        self.model_select = self.config['model']['select_model']

        # model / tokenizer / dataset 생성
        self.tokenizer, self.model, self.train_dataset, self.valid_dataset, self.test_dataset = self.get_model_tokenizer_datasets()

        # trainer 생성 & 실행
        self.trainer = self.get_trainer()
        self.trainer.train()

        # test 추론
        self.inference()

        # 모델 성능 분석
        self.model_eval()


    # model & tokenizer & dataset 생성
    def get_model_tokenizer_datasets(self):
        print("\n","-" * 30, " (모델 / 토크나이저 / 데이터 셋)을 생성합니다", "-" * 30)
        model_utils = Model(self.config, self.device)

        if self.model_select == 't5':
            tokenizer = model_utils.getT5Tokenizer()
            model = model_utils.getT5Model(tokenizer)

            train_dataset = T5TrainValidDataset(self.config, tokenizer, is_train=True)
            valid_dataset = T5TrainValidDataset(self.config, tokenizer, is_train=False)
            test_dataset = T5TestDataset(self.config, tokenizer)

        elif self.model_select == 'bart':
            tokenizer = model_utils.getBartTokenizer()
            model = model_utils.getBartModel(tokenizer)

            train_dataset = TrainValidDataset(self.config, tokenizer, is_train = True)
            valid_dataset = TrainValidDataset(self.config, tokenizer, is_train = False)
            test_dataset = TestDataset(self.config, tokenizer)

        return tokenizer, model, train_dataset, valid_dataset, test_dataset


    # trainer 생성
    def get_trainer(self):
        print("\n","-" * 30, "학습을 시작합니다", "-" * 30)
        trainer = Trainer(self.config, self.model, self.train_dataset, self.valid_dataset, self.tokenizer).get_trainer()
        return trainer
    
    # Test 추론 & submission 파일 저장
    def inference(self):
        print("\n", "-" * 30, "test 추론을 시작합니다", "-" * 30)
        test_utils = Test(self.config, self.test_dataset, self.tokenizer, self.device)
        best_model_path = self.trainer.state.best_model_checkpoint

        if self.model_select == 't5':
            model = test_utils.getT5Model(best_model_path)
            model_name = self.config['model']['t5']

        elif self.model_select == 'bart':
            model = test_utils.getBartModel(best_model_path)
            model_name = self.config['model']['bart']

        test_utils.testModel(model, model_name)

    # 모델 성능 분석
    def model_eval(self):
        print("\n", "-"*30, "모델 성능 분석", "-" * 30)
        model_path = self.trainer.state.best_model_checkpoint
        model_analyze = ModelAnalyze(self.config, self.tokenizer, self.device)
        valid_df = model_analyze.get_result(self.valid_dataset, model_path)

    # CUDA 내용물 비우기
    def clear_cuda_memory(self):
        torch.cuda.empty_cache()
        gc.collect()

    # 시드 고정
    def set_seed(self, seed = 42):
        random.seed(seed)
        
        # PyTorch의 시드 설정
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티-GPU 사용 시
        
        # CuDNN의 결정론적 작동 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 환경 변수 설정 (일부 PyTorch 작업의 랜덤성 제어)
        os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':
    Main()