from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os
from dotenv import load_dotenv
import sqlite3
from langchain_google_genai import GoogleGenerativeAI
import sqlite3
import json
from sentence_transformers import SentenceTransformer, util
import sqlite3
import numpy as np
from langchain_core.prompts import PromptTemplate
import ast
import random

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="JEJU!",
    page_icon="🍊",
)

model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        print("llm start")

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        print("llm end")

    def on_llm_new_token(self, token, *arlgs, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
        print("llm new token")


conn = sqlite3.connect('jeju2.db')
cursor = conn.cursor()

print("conn")

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ])

""""""

dtemplate = """
당신은 제주도 맛집 데이터 분석 전문가입니다.
주어진 질문을 분석하여 신한카드 데이터 관련 질문인지 일반 맛집 추천 질문인지 판단해주세요.
다음과 같은 형식의 데이터 필드가 포함된 질문이 주어질 것입니다:

개설일자 (YYYY-MM-DD)
주소 (String)
가맹점명_포함_텍스트 (String)
이용_건수_상위 (Integer)
총_매출_상위 (Integer)
건당_이용_금액_상위 (Integer)
여행_요일 (String)
성별_선호 (String)
선호_나이대 (String)
현지인맛집 (Boolean)
분류 (String)

판단 기준:

질문이 다음과 같은 패턴을 포함하면 "신한카드"를 출력하세요:

[지역명]에 있는 [업종]중 [조건]은?
구체적인 데이터 필드(이용 건수, 매출, 선호 나이대 등)를 기반으로 한 검색성 질문
SQL로 쿼리 가능한 형태의 데이터 조회 질문

그 외의 모든 경우 "맛집추천"을 출력하세요.

주의사항:

반드시 "신한카드" 또는 "맛집추천" 중 하나만 출력해야 합니다.
다른 어떤 설명이나 부가 텍스트도 포함하지 마세요.

예시:
입력: "제주시 한림읍에 있는 카페 중 30대 이용 비중이 가장 높은 곳은?"
출력: 신한카드
입력: "제주 공항 근처 맛집 알려주세요"
출력: 맛집추천
입력: "제주시 노형동에 있는 단품요리전문점 중 이용건수가 상위 10%에 속하고 현지인 이용비중이 가장 높은 곳은?"
출력: 신한카드

질문: {question}
"""

prompt = PromptTemplate.from_template(dtemplate)

Divchain = prompt | llm
""""""


""""""


def convert_string_to_dict(input_string):
    """
    문자열을 딕셔너리로 변환하고 'Null' 값을 None으로 변경하는 함수

    Args:
        input_string (str): 변환할 문자열

    Returns:
        dict: 변환된 딕셔너리
    """
    # 작은따옴표를 큰따옴표로 변경하여 JSON 형식으로 만듦
    json_string = input_string.replace("'", '"')

    try:
        # JSON 파싱 시도
        result = json.loads(json_string)
    except json.JSONDecodeError:
        # JSON 파싱 실패시 ast.literal_eval 사용
        result = ast.literal_eval(input_string)

    # 'Null' 값을 None으로 변경
    for key, value in result.items():
        if value == 'Null':
            result[key] = None

    return result


def get_sales_rank_condition(percentile, Col):
    """
    백분위 수를 OR 조건으로 연결된 TEXT 형식으로 변환
    """
    conditions = []

    if 0 <= percentile <= 10:
        conditions.append(Col+" = '6_90% 초과'")
    elif 10 < percentile <= 25:
        conditions.append(Col+" = '5_75~90%'")
        conditions.append(Col+" = '6_90% 초과'")
    elif 25 < percentile <= 50:
        conditions.append(Col+" = '4_50~75%'")
        conditions.append(Col+" = '5_75~90%'")
        conditions.append(Col+" = '6_90% 초과'")
    elif 50 < percentile <= 75:
        conditions.append(Col+" = '3_25~50%'")
        conditions.append(Col+" = '4_50~75%'")
        conditions.append(Col+" = '5_75~90%'")
        conditions.append(Col+" = '6_90% 초과'")
    elif 75 < percentile <= 90:
        conditions.append(Col+" = '2_10~25%'")
        conditions.append(Col+" = '3_25~50%'")
        conditions.append(Col+" = '4_50~75%'")
        conditions.append(Col+" = '5_75~90%'")
        conditions.append(Col+" = '6_90% 초과'")
    elif 90 < percentile <= 100:
        conditions.append(Col+" = '1_상위 10% 이하'")
        conditions.append(Col+" = '2_10~25%'")
        conditions.append(Col+" = '3_25~50%'")
        conditions.append(Col+" = '4_50~75%'")
        conditions.append(Col+" = '5_75~90%'")
        conditions.append(Col+" = '6_90% 초과'")

    if conditions:
        return "(" + " OR ".join(conditions) + ")"
    return None


def build_query(params):
    conditions = []
    order_by = ""

    # 기본 쿼리 시작
    base_query = "SELECT MCT_NAVER_NAME, NAVER_ADDR, WT, AMENITY, PHONE, PAYMENT, BOSS_TIP, original_type FROM Information WHERE 1=1"

    # 주소 조건
    if params["주소"] is not None:
        conditions.append(f"AND ADDR LIKE '%{params['주소']}%'")

    # 가맹점명 포함 텍스트 조건
    if params["가맹점명_포함_텍스트"] is not None:
        conditions.append(f"AND MCT_NM LIKE '%{params['가맹점명_포함_텍스트']}%'")

    # 이용 건수 상위% 조건
    if params["이용_건수_상위"] is not None:
        rank_condition = get_sales_rank_condition(
            params["이용_건수_상위"], "UE_CNT_GRP")
        if rank_condition:
            conditions.append(f"AND {rank_condition}")

    # 총 매출 상위% 조건 - 백분위를 TEXT 구간으로 변환
    if params["총_매출_상위"] is not None:
        rank_condition = get_sales_rank_condition(
            params["총_매출_상위"], "UE_AMT_GRP")
        if rank_condition:
            conditions.append(f"AND {rank_condition}")

    # if params["건당 이용 금액(가격대) 상위%"] is not None:
    #     rank_condition = get_sales_rank_condition(params["건당 이용 금액(가격대) 상위%"], "RC_M12_TOT_AMT_RANK")
    #     if rank_condition:
    #         conditions.append(f"OR {rank_condition}")

        # 분류 조건
    if params["분류"] is not None:
        conditions.append(f"AND original_type = '{params['분류']}'")

    # 선호 나이대 조건
    if params["선호_나이대"] is not None:
        age_conditions = {
            "20": "RC_M12_AGE_UND_20_CUS_CNT_RAT",
            "30": "RC_M12_AGE_30_CUS_CNT_RAT",
            "40": "RC_M12_AGE_40_CUS_CNT_RAT",
            "50": "RC_M12_AGE_50_CUS_CNT_RAT",
            "60": "RC_M12_AGE_OVR_60_CUS_CNT_RAT"
        }
        if params["선호_나이대"] in age_conditions:
            conditions.append(f"AND {age_conditions[params['선호_나이대']]} > 0.3")
            # 오름차순 정렬이 선호_나이대인 경우
            if params.get("오름차순") == "선호_나이대":
                order_by = f"ORDER BY {age_conditions[params['선호_나이대']]} DESC"

    # 성별 선호 조건
    if params["성별_선호"] is not None:
        if params["성별_선호"] == "남":
            conditions.append("ORDER BY RC_M12_MAL_CUS_CNT_RAT DESC")
        elif params["성별_선호"] == "여":
            conditions.append("ORDER BY RC_M12_MAL_CUS_CNT_RAT DESC")

    # 최종 쿼리 조합
    query = base_query + " " + " ".join(conditions)
    if order_by:
        query += " " + order_by

    return query


template22 = """
너가 만약 확실하게 모르는 정보는 절대 적지 말고, null 으로 채워넣어. 이거 기반으로 검색되는거라 오름차순 특징도 꼭 잡아야돼.
Generate a JSON output that includes the following fields:
1. "개설일자" (Date of establishment): a string in the format 'YYYY-MM-DD'
2. "주소" (Address): a string representing the address "ㅇㅇ시 ㅇㅇ동" 형태로 적어주세요
3. "가맹점명_포함_텍스트" (Store name text): a string related to the store name 확실히 포함될때만 
4. "이용_건수_상위" (Usage count percentile): an integer representing the top percentile for usage count
5. "총_매출_상위" (Total sales percentile): an integer representing the top percentile for total sales
6. "건당_이용_금액(가격대) 상위" (Amount per usage percentile): an integer representing the top percentile for amount spent per usage
7. "여행_요일" (Preferred days of travel): a string, choose from "월" or "화" or "수" or "목" or "금" or "토" or "일" -> "월,화,일" 이런 형태로
8. "성별_선호" (Gender preference): a string, choose from "남", "여"
9. "선호_나이대" (Preferred age group): a string, choose from "20", "30", "40", "50", "60"
10. "현지인맛집" (Local favorite restaurant): a boolean (true/false)
11. "분류" (Sort of restaurant): 다음 중 하나를 고르시오 모를 경우 고르지 말 것 - '패밀리 레스토랑', '단품요리 전문', '구내식당/푸드코트', '가정식', '베이커리', '차', '커피', '피자',
       '중식', '맥주/요리주점', '치킨', '샌드위치/토스트', '일식', '양식', '분식', '꼬치구이', '햄버거',
       '도너츠', '포장마차', '떡/한과', '주스', '아이스크림/빙수', '기사식당', '부페', '야식', '도시락'


Example query: "Find a restaurant in Jeju-si with high usage count and local user preference."
Output must strictly follow the structure:
{{
  "개설일자": "YYYY-MM-DD",
  "주소": String,
  "가맹점명_포함_텍스트": String,
  "이용_건수_상위": Integer,
  "총_매출_상위": Integer,
  "건당_이용_금액_상위": Integer,
  "여행_요일": String,
  "성별_선호": String,
  "선호_나이대": String,
  "현지인맛집": Boolean,
  "분류": String,
}}

질문: {question}
"""

prompt123 = PromptTemplate.from_template(template22)

SQLchain = prompt123 | llm
""""""


def recommend(user_question, address=None, merchant_name=None, pref_price=None, want_popular=False, day=None, pref_gender=None, want_local_matjip=False, restaurant_type=None, return_num=5):
    conn = sqlite3.connect('./jeju2.db')
    cursor = conn.cursor()

    query = """
WITH MaxVisitCounts AS (
    SELECT
        r.placeID,
        MAX(r.visit_num) AS max_visit_num
    FROM
        Review r
    WHERE
        r.visit_num >= 2
    GROUP BY
        r.placeID
),
TotalReviewCounts AS (
    SELECT
        r.placeID,
        COUNT(*) AS total_review_count
    FROM
        Review r
    GROUP BY
        r.placeID
),
AggregatedData AS (
    SELECT
        i.placeID,
        i.MCT_NM,
        i.UE_AMT_GRP,
        COALESCE(SUM(m.max_visit_num), 0) AS total_visit_num,
        COALESCE(trc.total_review_count, 0) AS total_review_count,
        i.MCT_TYPE,
        i.ADDR,
        i.CD,
        i.RC_M12_MAL_CUS_CNT_RAT,
        i.RC_M12_FME_CUS_CNT_RAT,
        i.LOCAL_UE_CNT_RAT,
        i.UE_CNT_GRP,
        i.keywords_embeddings
    FROM
        Information i
    LEFT JOIN
        MaxVisitCounts m ON i.placeID = m.placeID
    LEFT JOIN
        TotalReviewCounts trc ON i.placeID = trc.placeID
    GROUP BY
        i.placeID, i.MCT_NM, i.UE_AMT_GRP, i.MCT_TYPE, i.ADDR, i.CD, i.RC_M12_MAL_CUS_CNT_RAT, i.RC_M12_FME_CUS_CNT_RAT, i.LOCAL_UE_CNT_RAT, i.UE_CNT_GRP
)
SELECT
    a.placeID,
    a.MCT_NM,
    a.keywords_embeddings,
    (6 * (6 - a.UE_AMT_GRP) + 4 * (a.total_visit_num / NULLIF(a.total_review_count, 0))) AS score
FROM
    AggregatedData a
WHERE
    1=1
"""

    conditions = []
    params = []

    if restaurant_type:
        conditions.append("a.MCT_TYPE = ?")
        params.append(restaurant_type)

    if address:
        conditions.append("a.ADDR LIKE ?")
        params.append(f'%{address}%')

    if merchant_name:
        conditions.append("a.MCT_NM LIKE ?")
        params.append(f'%{merchant_name}%')

    if pref_price == "cheap":
        conditions.append(
            "a.UE_AMT_GRP IN ('1_상위 10% 이하', '2_10~25%', '3_25~50%')")
    elif pref_price == "expensive":
        conditions.append(
            "a.UE_AMT_GRP IN ('4_50~75%', '5_75~90%', '6_90% 초과')")

    if want_popular:
        conditions.append(
            "a.UE_CNT_GRP IN ('4_50~75%', '5_75~90%', '6_90% 초과')")

    if day:
        conditions.append("a.CD NOT LIKE ?")
        params.append(f'%{day}%')

    if conditions:
        query += " AND " + " AND ".join(conditions)

    query += " ORDER BY score DESC"

    if pref_gender == '남':
        query += ", FLOOR(a.RC_M12_MAL_CUS_CNT_RAT * 10) / 10 DESC"
    elif pref_gender == '여':
        query += ", FLOOR(a.RC_M12_FME_CUS_CNT_RAT * 10) / 10 DESC"

    if want_local_matjip:
        query += ", FLOOR(a.LOCAL_UE_CNT_RAT * 10) / 10 DESC"

    query += " LIMIT 300;"

    cursor.execute(query, params)
    results = cursor.fetchall()

    data = []
    for row in results:
        id_ = row[0]
        mct_nm = row[1]
        keywords_embeddings = row[2]
        if keywords_embeddings is None:
            continue
        embeddings = np.frombuffer(keywords_embeddings, dtype=np.float32)
        data.append((id_, mct_nm, embeddings))

    user_embedding = model.encode(user_question)

    cosine_scores = []
    for id_, mct_nm, embeddings in data:
        score = util.pytorch_cos_sim(user_embedding, embeddings)
        cosine_scores.append((id_, mct_nm, score.item()))

    cosine_scores.sort(key=lambda x: x[2], reverse=True)

    top_results = []
    for id_, mct_nm, score in cosine_scores[:return_num]:
        top_results.append(
            {'id': id_, 'restaurant_name': mct_nm, 'coss sim': score})

    cursor.close()
    conn.close()

    return top_results


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def reset_state():
    st.session_state["previous_response"] = None
    st.session_state["final_restaurant_lists"] = []


def send_message(message, role, save=False):

    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
너가 만약 모르는 정보는 절대 적지 말고, null 으로 채워넣어. 만약 아무런 정보도 없다고 판단되면, 아래 json 값들을 모두 null로 채워넣어.

Generate a JSON output that includes the following fields:
1. "address" (Address): a string representing the address. "노형동", "제주시", "납읍"처럼 지역 이름이여야 해. '제주공항 근처'와 같은 값은 들어가면 안돼.
2. "merchant_name" (Store name text): a string related to the store name
3. "pref_price" (Preffered Price): "cheap", "expensive"
4. "want_popular" (Preffered Population): a boolean (true / false)
5. "day" (Preferred days of travel): a string, choose from "월", "화", "수", "목", "금", "토", "일"
6. "pref_gender" (Gender preference): a string, choose from "남", "여"
7. "want_local_matjip" (Local favorite restaurant): a boolean (true / false)
8. "restaurant_type" (Sort of restaurant): '패밀리레스토랑', '호텔', '장례식장', '한식', '생선회', '국수', '중식당', '육류,고기요리', '카페', '베이커리', '24시뼈다귀탕', '찌개,전골', '맥주,호프', '치킨,닭강정', '피자', '햄버거', '돈가스', '브런치', '돼지고기구이', '가야밀면', '아귀찜,해물찜', '향토음식', '해물,생선요리', '닭갈비', '전복요리', '종합분식', '일식당', '샤브샤브', '해장국', '비빔밥', '칼국수,만두', '소고기구이', '낙지요리', '국밥', '분식', '장어,먹장어요리', '곱창,막창,양', '체험,홍보관', '족발,보쌈', '카페,디저트', '멕시코,남미음식', '베트남음식', '양식', '바(BAR)', '펜션', '라면', '정육식당', '매운탕,해물탕', '굴요리', '김밥', '곰탕,설렁탕', '브런치카페', '추어탕', '한정식', '와인', '오리요리', '양꼬치', '우동,소바', '요리주점', '이자카야', '백반,가정식', '생선구이', '종합도소매', '대게요리', '냉면', '복어요리', '이탈리아음식', '수련원,연수원', '프랜차이즈본사', '노래방', '떡볶이', '슈퍼,마트', '두부요리', '주류', '딤섬,중식만두', '떡,한과', '조개요리', '아이스크림', '프랑스음식', '백숙,삼계탕', '일본식라면', '술집', '감자탕', '닭볶음탕', '스파게티,파스타전문', '닭요리', '인테리어소품', '판촉,기념품', '뷔페', '떡카페', '인도음식', '쌈밥', '테이크아웃커피', '테마카페', '순대,순댓국', '죽', '막국수', '불닭', '닭발', '애견카페', '바닷가재요리', '종합생활용품', '도시락,컵밥', '일품순두부', '전세버스', '마라탕', '수산물', '태국음식', '미향해장국', '초밥,롤', '양갈비', '주꾸미요리', '한식뷔페', '떡류제조', '샌드위치', '스페인음식', '오뎅,꼬치', '만두', '포장마차', '찐빵', '전,빈대떡', '미용실', '덮밥', '전통,민속주점', '사철,영양탕', '퓨전음식'. "향토음식" is a traditional dish from Jeju.
9. "user_question": You need to put the user's question as a string!

Output must strictly follow the structure:
{{
  "address": "String",
  "merchant_name": "String",
  "pref_price": "String",
  "want_popular": Boolean,
  "day": "String",
  "pref_gender": "String",
  "want_local_matjip": Boolean,
  "restaurant_type": "String"
  "user_question": "String"
}}

Example query: "노형동에 몸국 가성비 있고, 인기 많은 곳 추천해줘."
output:
{{
  "address": "노형",
  "merchant_name": null,
  "pref_price": "cheap",
  "want_popular": true,
  "day": null,
  "pref_gender": null,
  "want_local_matjip": false,
  "restaurant_type": "향토음식"
  "user_question": "노형동에 몸국 가성비 있고, 인기 많은 곳 추천해줘."
}}

Example query: "제주공항 근처 맛집 알려줘! 도민들이 자주 가는 곳으로!"
output:
{{
  "address": "null",
  "merchant_name": null,
  "pref_price": null,
  "want_popular": false,
  "day": null,
  "pref_gender": null,
  "want_local_matjip": true,
  "restaurant_type": "null"
  "user_question": "null"
}}
            """,
        ),
        ("human", "{question}"),
    ]
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", """내가 제시한 음식점들을 각각 두 줄로 설명해.
            
            AMENITY와 BOSS_TIP, keywords는 최대한 포함시켜야 해. 다만, 'AMENITY', 'keywords'를 직접적으로 명시해서는 안돼. 해당 내용들을 자연스럽게 설명해야 해. 만약 None과 같이 제대로 된 데이터가 아니라면 해당 부분은 넘어가야 해.
            
            만약 아무런 값도 들어오지 않았다면, '해당 조건을 가진 음식점을 찾지 못했어요.. 다시 검색해주세요.'를 반환해야 해.
            
            """,
        ),
        ("human", "{question}"),
    ]
)

final_prompt_for_card = ChatPromptTemplate.from_messages(
    [
        (
            "system", "내가 제시한 음식점들을 각각 두 줄로 설명해. AMENITY와 BOSS_TIP, keywords는 최대한 포함시켜야 해. 다만, 'AMENITY', 'keywords'를 직접적으로 명시해서는 안돼. 해당 내용들을 자연스럽게 설명해야 해. 제목 아래에는 무조건 인용문으로 '(정량 데이터 검색 결과)'를 붙여야 해. 인용문 안에는 제목, 소제목 등을 절대 사용하면 안되고, 본문만 사용해야 해.",
        ),
        ("human", "{question}"),
    ]
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    print("init messages")

st.title("🍊 맛르방")


send_message("""
             혼저옵서예~\n
             **저에게 무엇이든 물어보세요!** \n
             '맛집 알려줘! 도민들이 자주 가고, 주차하기 편한 곳으로!', '20대 남자 넷이서 갈만한 횟집 있어?' 등 뭐든 편하게 물어보시면 정확하게 답해드릴게요!
             """, "ai", save=False)

# paint_history()

message = st.chat_input("질문을 입력해주세요!")

if message:
    reset_state()
    send_message(message, "human", save=False)

    divresponse = Divchain.invoke({"question": message})
    print(divresponse)

    if "신한카드" in divresponse:
        print("신한카드")

        response = SQLchain.invoke({"question": message})
        print(response)
        # 테스트

        if '```json' in response:
            start = response.find('json') + len('json\n')
            end = response.find('```', start)
            response = response[start:end].strip()
            print("json to dict parsing!")
            print(response)
        else:
            print(response)
        # test_string = "{'개설일자': 'Null', '주소': '제주시 한림읍', '가맹점명_포함_텍스트': 'Null', '이용_건수_상위': 'Null', '총_매출_상위': 'Null', '건당_이용_금액_상위': 'Null', '여행_요일': 'Null', '성별_선호': 'Null', '선호_나이대': '30', '현지인맛집': 'Null', '분류': '카페'}"
        converted_dict = convert_string_to_dict(response)
        print(converted_dict)

        query = build_query(converted_dict)
        print(query)

        query += " LIMIT 10"

        conn = sqlite3.connect('Jeju2.db')
        cursor = conn.cursor()

        cursor.execute(query)
        columns = cursor.fetchall()

        lists22 = []

        for column in columns:
            lists22.append(column)

        final_chain = (
            {
                "question": RunnablePassthrough(),
            }
            | final_prompt_for_card
            | llm
        )
        final_response = final_chain.invoke(f"{lists22}")

        send_message(final_response, "ai", save=False)

        conn.close()
    else:
        print("맛집추천")

        chain = (
            {
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        final_chain = (
            {
                "question": RunnablePassthrough(),
            }
            | final_prompt
            | llm
        )

        try:
            generatedResponse = chain.invoke(message)

            if '```json' in generatedResponse:
                start = generatedResponse.find('json') + len('json\n')
                end = generatedResponse.find('```', start)
                generatedResponse = generatedResponse[start:end].strip()

            try:
                generatedResponse_dict = json.loads(generatedResponse)
            except json.JSONDecodeError as json_err:
                send_message(f"JSON 파싱 오류: {json_err}", "ai", save=False)

            print(generatedResponse)

            results = recommend(
                user_question=generatedResponse_dict["user_question"],
                address=generatedResponse_dict["address"],
                merchant_name=generatedResponse_dict["merchant_name"],
                pref_price=generatedResponse_dict["pref_price"],
                want_popular=generatedResponse_dict["want_popular"],
                day=generatedResponse_dict["day"],
                pref_gender=generatedResponse_dict["pref_gender"],
                want_local_matjip=generatedResponse_dict["want_local_matjip"],
                restaurant_type=generatedResponse_dict["restaurant_type"],
                return_num=5
            )

            final_restaurant_lists = []

            for row in results:
                conn = sqlite3.connect('./jeju2.db')
                cursor = conn.cursor()

                placeid = row['id']

                cursor.execute("""
                  SELECT MCT_NAVER_NAME, NAVER_ADDR, WT, AMENITY, PHONE, PAYMENT, BOSS_TIP, MCT_TYPE, keywords, placeid 
                  FROM Information 
                  WHERE placeid = ?
              """, (placeid,))

                columns = [column[0] for column in cursor.description]
                for result in cursor.fetchall():
                    data_dict = dict(zip(columns, result))
                    final_restaurant_lists.append(data_dict)

            if final_restaurant_lists:
                for data_dict in final_restaurant_lists:
                    if data_dict['MCT_NAVER_NAME'] is not None:
                        st.sidebar.markdown(
                            f"### [{data_dict['MCT_NAVER_NAME']}]("
                            f"https://map.naver.com/p/search/{data_dict['MCT_NAVER_NAME'].replace(
                                ' ', '')}/place/{data_dict['placeID']}?c=15.00,0,0,0,dh&isCorrectAnswer=true)"
                        )
                    if data_dict['NAVER_ADDR'] is not None:
                        st.sidebar.write(f"주소: {data_dict['NAVER_ADDR']}")
                    if data_dict['PHONE'] is not None:
                        st.sidebar.write(f"전화번호: {data_dict['PHONE']}")
                    if data_dict['AMENITY'] is not None:
                        amenities = data_dict['AMENITY'].replace("|", " | ")
                        st.sidebar.write(f"편의시설: {amenities}")
                    if data_dict['PAYMENT'] is not None:
                        st.sidebar.write(f"추가 결제 수단: {data_dict['PAYMENT']}")
                    if data_dict['BOSS_TIP'] is not None:
                        st.sidebar.write(f"사장님 팁: {data_dict['BOSS_TIP']}")
                    if data_dict['MCT_NAVER_NAME'] is not None:
                        st.sidebar.write("---")

            final_restaurant_result_string = '\n\n'.join(
                map(str, final_restaurant_lists))

            print(final_restaurant_result_string)

            final_response = final_chain.invoke(
                final_restaurant_result_string + "만약 내가 앞에 아무런 말도 하지 않았다면, '해당 조건에 맞는 음식점을 찾지 못했어요. 조금 더 범위를 넓혀서 질문해주세요!'를 답해야 해.")

            send_message(final_response, "ai", save=False)

            conn.close()

        except Exception as e:
            send_message(f"이상한 오류가 발생했수다. 새로고침 해줍서 {e}", "ai", save=False)


else:
    st.session_state["messages"] = []
