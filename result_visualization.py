import streamlit as st
import pandas as pd
import re
import math

css = """
<style>
    .dataframe {
        width: 100%;
        max-width: 100%;
    }
    .dataframe td {
        overflow-x: scroll;
    }
    .dataframe td:nth-child(1){
        min-width: 600px
    }
    .dataframe td:nth-child(2),
    .dataframe td:nth-child(3) {
        min-width: 300px;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

def highlight_common_words(row):
    input_text = row['input_text']
    generated = row['generated_text']
    label = row['label']
    
    input_text = re.sub('\n', "<br>", input_text)
    generated_words = set(re.findall(r'\b[\w가-힣]+\b', generated))
    label_words = set(re.findall(r'\b[\w가-힣]+\b', label))
    
    common_words = generated_words.intersection(label_words)
    
    for word in common_words:
        pattern = re.compile(r'\b' + re.escape(word) + r'\b')
        generated = pattern.sub(f'<span style="color: red;">{word}</span>', generated)
        label = pattern.sub(f'<span style="color: red;">{word}</span>', label)
    
    return pd.Series({'input_text' : input_text, 'generated_text': generated, 'label': label})

def paginate_dataframe(dataframe, page_size, page_num):
    total_pages = math.ceil(len(dataframe) / page_size)
    start = (page_num - 1) * page_size
    end = start + page_size
    return dataframe.iloc[start:end], total_pages

# Streamlit 앱
st.title('결과 분석')

uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    avg_data = data.iloc[-1]
    data = data.iloc[:-1]

    # 평균 표시
    avg_data = avg_data[['rouge_1', 'rouge_2', 'rouge_l', 'rouge_mean']]
    avg_text = f"""
    <h1>total score : {avg_data['rouge_mean']*100:.4f}</h1>
    <p> 
        rouge_1 : {avg_data['rouge_1']:.4f}<br>
        rouge_2 : {avg_data['rouge_2']:.4f}<br>
        rouge_l : {avg_data['rouge_l']:.4f}
    </p>
    """
    st.markdown(avg_text, unsafe_allow_html = True)
    
    # 'generated_text'와 'label' 열만 선택
    display_data = data[['input_text', 'generated_text', 'label']]
    
    # 스타일 적용
    styled_data = display_data.apply(highlight_common_words, axis=1)

    data['input_text'] = styled_data['input_text']
    data['generated_text'] = styled_data['generated_text']
    data['label'] = styled_data['label']
    
    # 페이지네이션 설정
    page_size = 5
    total_pages = math.ceil(len(data) / page_size)

    if 'page_num' not in st.session_state:
        st.session_state.page_num = 1
      # 페이지 번호 입력
    page_num = st.number_input('Page number', min_value=1, max_value=total_pages, value=st.session_state.page_num)
    
    # 페이지 번호가 변경되면 세션 상태 업데이트
    if page_num != st.session_state.page_num:
        st.session_state.page_num = page_num
        st.rerun()

    # 데이터프레임 페이지네이션
    df_paginated, _ = paginate_dataframe(data, page_size, st.session_state.page_num)
    
    # 페이지 네비게이션 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button('First page'):
            st.session_state.page_num = 1
            st.rerun()
    with col2:
        if st.button('Last page'):
            st.session_state.page_num = total_pages
            st.rerun()
    
    # 페이지네이션된 데이터프레임 표시
    st.write(f"Page {page_num} of {total_pages}")
    st.write(df_paginated.to_html(escape=False, index=False), unsafe_allow_html=True)