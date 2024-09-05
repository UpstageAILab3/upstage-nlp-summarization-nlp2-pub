from kkh_solar import *
import pandas as pd

# CSV 파일 읽기
data = pd.read_csv('data/test.csv')

# 요약을 생성하고 test 번호와 요약문을 출력하는 부분
for idx, row in data.iterrows():
    dialogue = row['dialogue']
    test_number = row['fname']
    
    # 요약 생성
    summary = generate_summary(dialogue)
    
    # 현재 진행 중인 test 번호와 요약문 출력
    print(f"Processing {test_number}...")
    print(f"Summary: {summary}\n")
    
    # 요약문을 데이터프레임에 저장
    data.at[idx, 'summary'] = summary

# 'dialogue' 컬럼 제거
data = data.drop(columns=['dialogue'])

# 결과를 CSV 파일로 저장
data.to_csv('prediction/llm_submit.csv', index=False)


53, 423