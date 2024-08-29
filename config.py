general = {
    "data_path": "../data/",
    "model_name": "digit82/kobart-summarization",
    "output_dir": "./"
}

tokenizer = {
    "encoder_max_len": 512,
    "decoder_max_len": 100,
    "bos_token": f"{tokenizer.bos_token}",
    "eos_token": f"{tokenizer.eos_token}",
    "special_tokens": ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
}

training = {
    "overwrite_output_dir": True,
    "num_train_epochs": 20,
    "learning_rate": 1e-5,
    "per_device_train_batch_size": 50,
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "lr_scheduler_type": 'cosine',
    "optim": 'adamw_torch',
    "gradient_accumulation_steps": 1,
    "evaluation_strategy": 'epoch',
    "save_strategy": 'epoch',
    "save_total_limit": 5,
    "fp16": True,
    "load_best_model_at_end": True,
    "seed": 42,
    "logging_dir": "./logs",
    "logging_strategy": "epoch",
    "predict_with_generate": True,
    "generation_max_length": 100,
    "do_train": True,
    "do_eval": True,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001,
    "report_to": "wandb"
}

wandb = {
    "entity": "wandb_repo",
    "project": "project_name",
    "name": "run_name"
}

inference = {
    "ckt_path": "model ckt path",
    "result_path": "./prediction/",
    "no_repeat_ngram_size": 2,
    "early_stopping": True,
    "generate_max_length": 100,
    "num_beams": 4,
    "batch_size": 32,
    "remove_tokens": ['<usr>', f"{tokenizer.bos_token}", f"{tokenizer.eos_token}", f"{tokenizer.pad_token}"]
}