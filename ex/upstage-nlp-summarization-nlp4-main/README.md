[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/3DbKuh4a)

# Dialogue Summarization | 일상 대화 요약

## Team NLP 4조

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|
| [강승현](https://github.com/UpstageAILab)                        | [김형수](https://github.com/UpstageAILab)                        | [이소영A](https://github.com/UpstageAILab)                       | [김창희](https://github.com/UpstageAILab)                        | [진성준](https://github.com/UpstageAILab)                        | [김하연](https://github.com/UpstageAILab)                        |
| 팀장                                                            | 팀원                                                            | 팀원                                                            | 팀원                                                            | 팀원                                                            | 팀원                                                            |

## 0. Overview

### Environment

- **(컴퓨팅 환경)** 인당 RTX 3090 서버를 VSCode와 SSH로 연결하여 사용
- **(협업 환경)** Github, Wandb
- **(의사 소통)** Slack, Zoom

## 1. Competiton Info

### Overview

- 이번 대회는 Dialogue Summurization 대회로 일상 대화로 이루어진 대화문의 요약을 생성하는 대회입니다.

### Timeline

- March 08, 2024 - Start Date
- March 20, 2024 - Final submission deadline

## 2. Components

### Directory

```
├── README.md
├── changhee
│   ├── DS-T5.ipynb
│   └── cleaning_data.ipynb
├── hayeon
│   ├── kobart_gogamza_keyword.ipynb
│   ├── kot5_keyword.ipynb
│   └── test.csv
├── hyperparameter-tuning
│   ├── hp_tuning_ver_2
│   ├── hp_tuning_ver_3.ipynb
│   └── optuna_hsk.ipynb
├── seongjun
│   └── t501_lora_inference.ipynb
├── seunghyun
│   └── kobert.ipynb
└── soyoung
    └── baseline_kot5.ipynb
```

## 3. Data descrption

### Dataset overview

- 이번 대회에 사용된 데이터는 모두 대화문 데이터이며 train data, validation data, test data는 각각 12,457개, 499개, 499개 입니다. 각각의 대화문은 최소 2명에서 최대 7명의 화자가 등장하며, 최소 2턴에서 최대 60턴까지 대화가 이어집니다. 대화문에서 발화자는 #Person"N"#으로 구분되어있습니다.
- 대화문에 존재하는 개인정보(예: 전화번호, 주소 등)는 다음과 같이 마스킹되어 있습니다.
  - 예) 전화번호 -> #PhoneNumber#

### EDA

- Train data, validation data, test data 에 존재하는 #..# 형식의 단어: 
  
  - `['#Address#', '#CarNumber#', '#CardNumber#', '#DateOfBirth#', '#Email#', '#PassportNumber#', '#Person#', '#Person1#', '#Person2#', '#Person3#', '#Person4#', '#Person5#', '#Person6#', '#Person7#', '#PhoneNumber#', '#SSN#']`

- Train data, validation data, test data를 `kobart-summarization`으로 tokenized 했을 때 각 대화문과 요약문의 길이 정보
  
  |         | **Train Data**                                                                   | **Validation Data**                                                              | **Test Data**                                                                     |
  | ------- |:-------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
  | **대화문** | mean: 158.597014<br/>min: 32<br/>25%: 105<br/>50%: 141<br/>75%: 197<br/>max: 950 | mean: 156.687375<br/>min: 46<br/>25%: 104<br/>50%: 139<br/>75%: 194<br/>max: 620 | mean: 163.146293<br/>min: 38<br/>25%: 103<br/>50%: 149<br/>75%: 204<br/>max: 1041 |
  | **요약문** | mean: 30.980734<br/>min: 7<br/>25%: 21<br/>50%: 29<br/>75%: 38<br/>max: 165      | mean: 28.795591<br/>min: 9<br/>25%: 19<br/>50%: 26<br/>75%: 35<br/>max: 89       |                                                                                   |

### Data Processing

- 오타 수정
  
  - '...#Person2#: 먼저, 이것은 19세기 초 배경ㅇ로 설정된 로맨스 소설이에요...' -> ‘으로’
  
  - '...#Person1#: 편집장이 제ㅏ 다른 잡지에서 편집자로 일했던 경험이 있다는 걸 듣고, 그가 도우미 편집자가 되고 싶냐고 물어봤어요....' -> ‘제가’
  
  - '...#Person1#: 이제 그만. 너는 아직ㅍ알맞는 사람을 만나지 못했을 뿐이고, 너는 너무 많이 일하는 것 같아. 너는 어떻게 즐기고 삶을 즐기는 법을 배워야 해....' -> ‘ 알맞는’
  
  - '...#Person1#: 아무것도 안 했어. 그는 결국 나갔어. 그런데 오늘 또 그를 봤어. 신발 가게 밖에서. 카페 근처에서. 나는 CD 가게에 들어가서 CD를 보는 척했ㄷ거든. 그런데 그도 들어왔어....' -> ‘ 거든'**

- 연속된 자음 대체
  
  - '...#Person1#: 속았어! ㅋㅋ.. 완전 속았어....' -> ‘웃기다’

- 마스킹 정보 추출 -> 토큰화 시 Special Token으로 추가

## 4. Modeling

**강승현**

- 사용한 모델  
  
  - bart 기반 한국어 사전학습 모델  
    
    - `digit82/kobart-summarization`
    
    - `EbanLee/kobart-summary-v2`
    
    - `gangyeolkim/kobart-korean-summarizer-v2`
    
    - `ainize/kobart-news`
  
  - gpt2 기반 한국어 사전학습 모델  
    
    - `digit82/kogpt2-summarization`

- 실험 결과
  
  - `digit82/kobart-summarization`
    
    - baseline 모델: eval_loss => 0.5629, 리더보드 점수 => 41.4015
    
    - Special Token 추가: eval_loss => 0.5623, 리더보드 점수 => 41.5824 (점수 상승)
    
    - 텍스트 정제(스탑워드 리스트에 포함된 단어 제거): eval_loss => 0.5619, 리더보드 점수 => 41.7505 (점수 상승)
  
  - `EbanLee/kobart-summary-v2`
    
    - eval/loss => 0.6062
  
  - `gogamza/kobart-summarization`
    
    - eval_loss => 0.5790, 리더보드 점수 => 40.2952 (점수 하락)
  
  - `ainize/kobart-news`
    
    - val_loss => 0.6404, 리더보드 점수 => 39.3084 (점수 하락)
  
  - `digit82/kogpt2-summarization`
    
    - 리더보드 점수 => 27.5233 (점수 하락)

**김창희**

- Hugging face의 모델 `psyche/KoT5-summarization`, `lcw99/t5-large-korean-text-summary`, `eenzeenee/t5-base-korean-summarization`을 사용하여 테스트

- `psyche/KoT5-summarization`을 사용했을 때 Train loss는 0, Validation Loss가 고정되는 문제가 있어 transformer의 trainer를 사용하지 않고 직접 training function을 구현하여 학습을 하였지만 baseline의 `digit82/kobart-summarization` 보다 성능이 좋지 못함

- `lcw99/t5-large-korean-text-summary`, `eenzeenee/t5-base-korean-summarization`를 사용하였을 때 Validation Loss가 가장 낮게 나오는 `lcw99/t5-large-korean-text-summary`모델을 선택

**김하연**

- 유총재(2023). Improvement of Dialogue Summarization Using Keyword Extractor 참고, 키워드 기반 생성 요약 진행

- pretrained model : gogamza/kobart-summarization

- Dialogue, Topic을 학습시킨 모델로 Test Topic 생성, Dialogue, Topic, Summary를 학습시킨 모델로 Test Summary 생성
  
  - 최종 결과: `final_result` => 41.8308, `rouge1` => 0.5135, `rouge2` => 0.3181, `rougeL` => 0.4233

**김형수**

- EDA 과정에서 데이터셋의 대화문과 요약문 최대 길이가 베이스라인 코드의 `encoder_max_len`와 `decoder_max_len`보다 긴 것을 확인하고, `encoder_max_len`, `decoder_max_len`, `generation_max_length` 값을 조절해가며 학습. 
  
  - 최초 `final_result` 41.6394에서 42.0135로 점수 소폭 상승

- 추가로 더 많은 `Seq2SeqTrainer`의 하이퍼파라미터를 조절하여 튜닝하기 위하여 `optuna` 라이브러리를 임포트하여 하이퍼파라미터 튜닝

- `Trainer` 클래스의 `hyperparameter_search` 메서드를 사용하여 하이퍼파라미터`learning_rate`, `per_device_train_batch_size`, `per_device_eval_batch_size`, `weight_decay`, `lr_scheduler_type`, `optim`를 튜닝.
  
  - 최종 결과: `final_result`=> 42.1127

**이소영**

- 사용한 모델  
  - bart 기반의 한국어 사전학습 모델  
    - `digit82/kobart-summarization`  
  - t5 기반의 한국어 사전학습 모델  
    - `paust/pko-t5-base`  
    - `lcw99/t5-base-korean-text-summary`  
    - `noahkim/KoT5_news_summarization`  
    - `paust/pko-t5-large`  
    - `lcw99/t5-large-korean-text-summary`
- Modeling Process
  - 가설1: eda 결과를 학습 및 평가 데이터의 토큰 개수를 고려하여 encoder_max_len, decoder_max_len, generation_max_length를 증가시키면 성능이 향상될 것
    - Model: digit82/kobart-summarization  
                      encoder_max_len: 1000  
      decoder_max_len: 200  
      per_device_train_batch_size: 16  
      -> 점수 향상, LB: 42.1735  
    - Model: digit82/kobart-summarization  
      encoder_max_len: 1000  
      decoder_max_len: 200  
      generation_max_length: 200  
      per_device_train_batch_size: 16  
      -> 점수 향상, LB: 42.4697
  - 가설2: learning rate를 줄이면 파라미터가 더 안정적으로 수렴하여 스코어가 향상될 것
    - Model: digit82/kobart-summarization  
      encoder_max_len: 1000  
      decoder_max_len: 200  
      generation_max_length: 200  
      per_device_train_batch_size: 16  
      learning_rate: 5-e6  
      -> 점수 하락, LB: 41.1145
  - 가설3: weight decay를 늘리면 오버피팅을 방지하여 모델의 일반화 성능이 향상될 것
    - Model: digit82/kobart-summarization  
      encoder_max_len: 1000  
      decoder_max_len: 200  
      generation_max_length: 200  
      per_device_train_batch_size: 16  
      weight_decay: 0.02  
      -> 점수 하락, LB: 41.5500
  - 가설4: Text-to-Text 방식으로 입력과 출력 사이의 상관 관계를 명시적으로 모델링하는 T5 기반 모델이 대화문 요약 task에서 더 높은 성능을 보일 것  
    - Model: paust/pko-t5-base  
      encoder_max_len: 1000  
      decoder_max_len: 200  
      generation_max_length: 200  
      per_device_train_batch_size: 6  
      -> 점수 상승, LB: 42.8954  
    - Model: lcw99/t5-base-korean-text-summary  
      encoder_max_len: 1000  
      decoder_max_len: 200  
      generation_max_length: 200  
      per_device_train_batch_size: 6  
      -> 점수 상승, LB: 42.9973  
    - Model: noahkim/KoT5_news_summarization  
      encoder_max_len: 1000  
      decoder_max_len: 200  
      generation_max_length: 200  
      per_device_train_batch_size: 6  
      -> 점수 상승, LB: 43.2997
  - 가설5: batch size를 늘리거나 gradient accumulation steps를 늘려 메모리 효율적인 방식으로 더 큰 배치 크기로 학습할 수 있도록 하면 수렴 속도가 향상될 것  
    - Model: noahkim/KoT5_news_summarization  
      encoder_max_len: 1000  
      decoder_max_len: 200  
      generation_max_length: 200  
      per_device_train_batch_size: 7  
      -> 점수 하락, LB: 42.5872  
    - Model: noahkim/KoT5_news_summarization  
      encoder_max_len: 1000  
      decoder_max_len: 200  
      generation_max_length: 200  
      per_device_train_batch_size: 7  
      gradient_accumulation_steps: 4  
      -> 점수 하락, LB: 42.4952
  - 가설6: 더 강력한 t5 모델인 t5-large 기반 사전 학습 모델을 사용하면 성능이 향상될 것  
    - Model: lcw99/t5-large-korean-text-summary  
      encoder_max_len: 1000  
      decoder_max_len: 200  
      generation_max_length: 200  
      per_device_train_batch_size: 1  
      -> 점수 상승, LB: 43.7724

**진성준**

- 이소영님 께서 공유해주신 모델 학습 후 lora 추가 학습 사용 모델
  
  - `noahkim/KoT5_news_summarization`
    
    - `"generation_max_length"`: 200,
    - `"encoder_max_len"`: 1000,  
    - `"decoder_max_len"`: 2000,  
  
  - `LoRA`
    
    ```python
    from peft import LoraConfig, TaskType, get_peft_model, PeftModellora_config = LoraConfig(  
    r=32,  
    target_modules=["q","v"],  
    bias="none",  
    task_type=TaskType.SEQ_2_SEQ_LM  
    )  
    lora_model = get_peft_model(generate_model, lora_config)  
    ```

- 최종 결과: `final_result` => 42.8141, `rouge2` => 0.3260,  `rougeL` => 0.4325

## 5. Result

### Leader Board

- **Leader Board**
  
  <p align="center">
  <img src = "https://github.com/UpstageAILab/upstage-nlp-summarization-nlp4/assets/73140315/2f04b6ec-6ecd-4f39-998f-da990752a622" width="90%" height="90%">
  </p>

- **final_result**: 41.4034
  
  **rouge1**: 0.5153
  
  **rouge2**: 0.3079
  
  **rougeL**: 0.4189

### Presentation

- [[패스트캠퍼스] Upstage AI Lab 1기_4조_NLP_경진대회 발표자료 - Google Slides](https://docs.google.com/presentation/d/1cHfb_lUnwTv2N0KXznXokDiXKVmHmT4JePSsrKm-LFI/edit?usp=sharing)

### Meeting Log

- [Meeting Log](https://quickest-asterisk-75d.notion.site/d2f34ff6d4d2415a905b9bd781f6a549?v=49d734caa9ce4a51b480219e5b99ba04&pvs=4)

### Reference

- [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Using hyperparameter-search in Trainer - #9 by sgugger - 🤗Transformers - Hugging Face Forums](https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/9?page=3)
- [Huggingface text classification optuna wandb | Kaggle](https://www.kaggle.com/code/mahmoudhamza/huggingface-text-classification-optuna-wandb)
