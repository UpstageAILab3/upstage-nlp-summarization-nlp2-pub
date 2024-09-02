import streamlit as st
import pandas as pd
import re

def highlight_common_words(row):
    generated = row['generated_text']
    label = row['label']
    
    # 단어 단위로 분리 (한글, 영문, 숫자 포함)
    generated_words = set(re.findall(r'\b[\w가-힣]+\b', generated))
    label_words = set(re.findall(r'\b[\w가-힣]+\b', label))
    
    # 공통 단어 찾기
    common_words = generated_words.intersection(label_words)
    
    # 공통 단어에 대해 하이라이트 적용
    for word in common_words:
        pattern = re.compile(r'\b' + re.escape(word) + r'\b')
        generated = pattern.sub(f'<span style="color: red;">{word}</span>', generated)
        label = pattern.sub(f'<span style="color: red;">{word}</span>', label)
    
    return pd.Series({'generated_text': generated, 'label': label})

# Streamlit 앱
st.title('결과 분석')

uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # 'generated_text'와 'label' 열만 선택
    display_data = data[['generated_text', 'label']]
    
    # 스타일 적용
    styled_data = display_data.apply(highlight_common_words, axis=1)

    data['generated_text'] = styled_data['generated_text']
    data['label'] = styled_data['label']
    
    # Streamlit에 표시
    st.write(data.to_html(escape=False), unsafe_allow_html=True)