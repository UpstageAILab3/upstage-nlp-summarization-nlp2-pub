from openai import OpenAI  # openai==1.2.0

# 요약 생성 함수
def generate_summary(text):

    # 파일에서 API 키를 읽는 함수
    def get_api_key(file_path):
        with open(file_path, "r") as file:
            api_key = file.readline().strip()
        return api_key

    # API 키 파일 경로
    api_key_file = "ex_key/solar.txt"
    api_key = get_api_key(api_key_file)

    # OpenAI 클라이언트 초기화
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.upstage.ai/v1/solar"
    )


    text2 = f"""{text}

    위 내용을 요약해줘
    - 요약문은 한 문장 또는 두 문장으로 적어줘
    - 요약문은 최대 90 글자 수로 적어줘
    - 요약문은 3인칭 관찰자가 설명하듯이 적어주고, 모두 한글로 작성해줘
    - 사람 이름 대신에, #Person1# 이나 #Person2# 와 같은 키워드 형태로 적어줘
    """

    # 요청 생성
    stream = client.chat.completions.create(
        model="solar-1-mini-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": text2
            }
        ],
        stream=True,
    )

    # 스트리밍으로 응답 출력
    summary = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            summary += chunk.choices[0].delta.content

    # 콤마(,)가 포함된 경우 쌍따옴표 추가
    if ',' in summary:
        summary = f'"{summary}"'

    return summary

# # 테스트용 호출
# dialogue = """
# #Person1#: 실례합니다, 열쇠 한 묶음 보셨나요?
# #Person2#: 어떤 종류의 열쇠인가요?
# #Person1#: 5개의 열쇠와 작은 발 장식이 있어요.
# #Person2#: 안타깝네요! 저는 보지 못했습니다.
# #Person1#: 그럼, 찾는 데 도와주실 수 있나요? 제가 여기는 처음이라서요.
# #Person2#: 물론입니다. 기꺼이 도와드리겠습니다. 사라진 열쇠를 찾는 데 도와드리겠습니다.
# #Person1#: 정말 친절하시네요.
# #Person2#: 별 말씀을요. 어, 찾았어요.
# #Person1#: 오, 하느님 감사합니다! 어떻게 감사의 말씀을 드려야 할지 모르겠네요.
# #Person2#: 천만에요.
# """

# summary = generate_summary(dialogue)
# print(summary)
