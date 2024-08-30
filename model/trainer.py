from transformers import Seq2SeqTrainingArguments, EarlyStoppingCallback, Seq2SeqTrainer
from rouge import Rouge

class Trainer():
    def __init__(self, config, model, train_dataset, val_dataset, tokenizer):
        self.config = config

        # 파라미터 설정
        train_config = config['train']
        valid_config = config['valid']

        self.train_args = Seq2SeqTrainingArguments(
            # 학습에 필요한 필수 파라미터
            num_train_epochs = train_config['epochs'],
            learning_rate = train_config['learning_rate'],
            optim = train_config['optim'],
            per_device_train_batch_size = train_config['batch_size'],
            generation_max_length = train_config['generation_max_length'],
            seed = train_config['seed'],

            # 평가와 관련된 파라미터
            per_device_eval_batch_size = valid_config['batch_size'],
            evaluation_strategy = valid_config['evaluation_strategy'],
            predict_with_generate = train_config['predict_with_generate'],
            do_train = train_config['do_train'],
            do_eval = train_config['do_eval'],

            # 학습 성능을 올려주는 파라미터
            lr_scheduler_type = train_config['lr_scheduler_type'],
            warmup_ratio = train_config['warmup_ratio'],
            weight_decay = train_config['weight_decay'],
            gradient_accumulation_steps = train_config['gradient_accumulation_steps'],
            fp16 = train_config['fp16'],

            # save 와 관련된 파라미터
            output_dir = self.config['path']['output_dir'],
            overwrite_output_dir = train_config['overwrite_output_dir'],
            save_strategy = train_config['save_strategy'],
            save_total_limit = train_config['save_total_limit'],
            load_best_model_at_end = train_config['load_best_model_at_end'],
        )

        # callback 설정
        self.callback = EarlyStoppingCallback(
            early_stopping_patience = valid_config['early_stopping_patience'],
            early_stopping_threshold = valid_config['early_stopping_threshold']
        )

        # Trainer 생성
        self.trainer = Seq2SeqTrainer(
            model = model,
            args = self.train_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            compute_metrics = lambda pred: self.compute_metrics(config, tokenizer, pred),
            callbacks = [self.callback]
        )

    def compute_metrics(self, config, tokenizer, pred):
        rouge = Rouge()
        predictions = pred.predictions
        labels = pred.label_ids

        # 의미 없는 토큰(-100) -> pad 토큰
        pad_token_id = tokenizer.pad_token_id
        predictions[predictions == -100] = pad_token_id
        labels[labels == -100] = pad_token_id

        # token -> text 변환
        decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces = True)
        labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces = True)

        # remove token 제거
        replaced_predictions = decoded_preds.copy()
        replaced_labels = labels.copy()
        remove_tokens = config['test']['remove_tokens']

        for token in remove_tokens:
            replaced_predictions = [sentence.replace(token, " ") for sentence in replaced_predictions]
            replaced_labels = [sentence.replace(token, " ") for sentence in replaced_labels]

        results = rouge.get_scores(replaced_predictions, replaced_labels, avg = True)
        result = {key: value["f"] for key, value in results.items()}
        
        return result

    def get_trainer(self):
        return self.trainer