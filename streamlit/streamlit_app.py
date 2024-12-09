### 1. 환경 설정 ----------------------------------------------------------------------

#라이브러리 import
import pandas as pd
import numpy as np

import pyupbit
import sqlite3 #db 연결

import streamlit as st
import plotly.express as px
from PIL import Image

### 2. streamlit 페이지 스타일 설정 -----------------------------------------------------

#페이지 설정
st.set_page_config(page_title="Bitcoin Trading Portfolio", layout="wide") #페이지 제목과 화면 넓게 설정

#디자인 (CSS 스타일 코드)
header_block = """
<style>
.stAppHeader.st-emotion-cache-12fmjuu.ezrtsby2{
    background-color: #e7ebf1;
}
</style>
"""
st.markdown(header_block, unsafe_allow_html=True)

out_block = """
<style>
.stMainBlockContainer.block-container.st-emotion-cache-1jicfl2.ea3mdgi5{
    background-color: #e7ebf1;
}
</style>
"""
st.markdown(out_block, unsafe_allow_html=True)

upper_block = """
<style>
.stColumn.st-emotion-cache-1h4axjh.e1f1d6gn3{
    background-color: #01369f;
    color: #ffffff;
    padding: 20px;
    justify-content: center; /* 수평 중앙 정렬 */
    align-items: center; /* 수직 중앙 정렬 */
}
</style>
"""
st.markdown(upper_block, unsafe_allow_html=True)

main1_block = """
<style>
.stColumn.st-emotion-cache-1msb0ab.e1f1d6gn3{
    background-color: #ffffff;
    padding: 20px;
    justify-content: center; /* 수평 중앙 정렬 */
    align-items: center; /* 수직 중앙 정렬 */
}
</style>
"""
st.markdown(main1_block, unsafe_allow_html=True)

metric_block = """
<style>
.stMetric{
    padding: 15px;
}
</style>
"""
st.markdown(metric_block, unsafe_allow_html=True)

metriclabel_block = """
<style>
[data-testid="stMetricLabel"]{
    color: #67727b;
}
</style>
"""
st.markdown(metriclabel_block, unsafe_allow_html=True)

metricvalue_block = """
<style>
[data-testid="stMetricValue"]{
    font-size: 30px;
}
</style>
"""
st.markdown(metricvalue_block, unsafe_allow_html=True)

main2_block = """
<style>
.stColumn.st-emotion-cache-136ne36.e1f1d6gn3{
    background-color: #ffffff;
    padding: 20px;
    justify-content: center; /* 수평 중앙 정렬 */
    align-items: center; /* 수직 중앙 정렬 */
}
</style>
"""
st.markdown(main2_block, unsafe_allow_html=True)

### 3. 데이터베이스 연결 ----------------------------------------------------------------

#데이터베이스 연결 함수
def get_connection():
    return sqlite3.connect("bitcoin_trades.db")

#데이터 로드 함수
@st.cache_data(ttl=600) #캐시를 사용하여 데이터 로딩 시간을 줄임
def load_data():
    conn = get_connection()
    query = "SELECT * FROM trades"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

#데이터 불러오기
df = load_data()

#df 데이터프레임 가독성 좋게 변경
df = df[['id', 'timestamp', 'decision', 'percentage', 'reason', 'btc_balance', 'krw_balance', 'btc_avg_buy_price', 'btc_krw_price', 'reflection']] #열 순서 지정
df = df.drop(["btc_avg_buy_price", "reflection"], axis=1) #불필요한 열 제거
df = df.sort_values(by='timestamp', ascending=False) #시간 기준으로 내림차순 정렬
df['id'] = range(1, len(df) + 1) #id열 다시 생성
df['timestamp'] = pd.to_datetime(df['timestamp']) #timestamp를 datetime 형식으로 변환
df["percentage"] = (df["percentage"]).astype(int)
df["krw_balance"] = round(df["krw_balance"])

for i in range(len(df)):
    if df["decision"][i] == "buy":
        df["decision"][i] = "매수"
    elif df["decision"][i] == "sell":
        df["decision"][i] = "매도"
    elif df["decision"][i] == "hold":
        df["decision"][i] = "보유"


### 4. 업비트 연동 ----------------------------------------------------------------

access_key = "***"
secret_key = "***"
upbit = pyupbit.Upbit(access_key, secret_key)

krw_balance = upbit.get_balance(ticker="KRW") #보유 원화 잔액
total_price = upbit.get_amount("ALL") #총 매수 금액

# #대시보드에 표시되는 수치
# 현재 비트코인 보유량 가져오기
btc_balance = upbit.get_balance(ticker="KRW-BTC")  # 보유 비트코인 양

# 현재 비트코인 가격 가져오기
btc_price = pyupbit.get_current_price(ticker="KRW-BTC")  # 현재 비트코인 가격

# 총 자산 계산
total_asset = krw_balance + (btc_balance * btc_price)  # 총 보유 자산

input_price = 100000 #원금 (10만원)
profit_percent = ((total_asset - input_price) / input_price) * 100 #수익률(%)
trading_percent = {"min":30, "max": 50}#설정된 최저/최대 매매 비율


### 5. 화면 레이아웃 ----------------------------------------------------------------

empty1, con1, empty2 = st.columns([0.01, 1.0, 0.01]) #[1] 제목
empty1, con21, con22, empty2 = st.columns([0.01, 0.5, 0.5, 0.01]) #[2] 투자정보(수치) + [4] 투자정보(표)
empty1, con31, con32, empty2 = st.columns([0.01, 0.5, 0.5, 0.01]) #[3] 투자정보(시각화) + [5] 자산정보
empty1, con41, con42, con43, empty2 = st.columns([0.01, 0.1, 0.8, 0.1, 0.01]) #[6] 비트코인 정보
empty1, con5, empty2 = st.columns([0.01, 1.0, 0.01]) #[7] 하단

### 6. 대시보드 구성 ----------------------------------------------------------------

#[1] 제목
with con1:
    st.title('**:chart_with_upwards_trend:비트코인 투자 현황 대시보드**')

    
#[2] 투자정보(수치)
with con21:
    st.markdown(f'<h1 style="font-size:15px;">{"기본 정보"}</h1>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("총 자산", f'￦{total_asset:,.0f}')  # 1000단위마다 콤마 표시
    c2.metric("수익률", f'{profit_percent:.2f}%')
    
    c3, c4 = st.columns(2)
    c3.metric("투자 원금", f'￦{input_price:,.0f}')
    c4.metric("총 거래 횟수", f'{len(df)}번')
    
    #DB에서 날짜 형태 변경
    dt = pd.to_datetime(df["timestamp"]) #timestamp 값을 datetime 형태로 변환
    first_trade_date = dt.min() #거래 시작일
    last_trade_date = dt.max() #최근 거래일
    
    c5, c6 = st.columns(2)
    c5.metric("최저 매매비율", f'{int(trading_percent["min"])}%')
    c6.metric("거래 시작일", f'{first_trade_date.strftime("%Y.%m.%d")}\n{first_trade_date.strftime("%H:%M")}')
    
    c7, c8 = st.columns(2)
    c7.metric("최대 매매비율", f'{int(trading_percent["max"])}%')
    c8.metric("최근 거래일", f'{last_trade_date.strftime("%Y.%m.%d")}\n{last_trade_date.strftime("%H:%M")}')

#[4] 투자정보 (표)
with con22:
    new_columns = { #표에서 표시되는 열 이름
    'id': '번호',
    'timestamp': '매매 시간',
    'decision': '매매 종류',
    'percentage': '매매 비율(%)',
    'reason': 'AI의 판단 이유',
    'btc_balance': '보유 비트코인 수량(BTC)',
    'krw_balance': '보유 원화 잔액(원)',
    'btc_krw_price': '비트코인 가격(원)'
    }
    
    st.markdown(f'<h1 style="font-size:15px;">{"거래 내역"}</h1>', help="거래 내역은 최신순으로 정렬되어 있고, 이중 하나를 선택하시면 AI의 판단 이유를 볼 수 있습니다.", unsafe_allow_html=True)
        
    # 데이터프레임 열 이름 변경 및 'reason' 열 제거
    display_df = df.drop(columns=['reason']).rename(columns=new_columns)
    
    
    # 데이터프레임 출력 및 선택 기능 활성화
    try:
        pick = st.dataframe(
            display_df,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            use_container_width=True
        )

        # 선택된 행 확인
        if pick and "rows" in pick.selection:
            selected_row_idx = pick.selection["rows"]
            if selected_row_idx:  # 선택된 행이 있을 경우
                selected_reason = df.iloc[selected_row_idx]['reason'].values[0]
            else:
                selected_reason = "거래 내역을 선택해주세요."  # 기본 메시지
        else:
            selected_reason = "거래 내역을 선택해주세요."  # 기본 메시지
    except Exception as e:
        # 선택되지 않은 경우 오류 없이 기본 데이터프레임만 출력
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )
        selected_reason = "거래 내역을 선택해주세요."  # 기본 메시지

    # 항상 expander 표시
    with st.expander("AI의 판단 이유 💬", expanded=True):
        st.write(selected_reason)
    
    # st.markdown(f'<h1 style="font-size:15px;">{"거래 내역"}</h1>', help="거래 내역은 최신순으로 정렬되어 있습니다.", unsafe_allow_html=True)
    # pick = st.dataframe(df.rename(columns=new_columns), hide_index=True, on_select="rerun", selection_mode = "single-row", use_container_width=True) #표가 웹페이지의 전체 너비를 사용
    # st.text(pick["selection"]["rows"][0])
  
    
#[3] 투자정보(시각화)
with con31:
    decision_counts = df["decision"].value_counts()
    
    color_map = {
        "매수": "#F6C6AD",
        "매도": "#A6CAEC",
        "보유": "#B4E5A2" 
    }

    fig = px.pie(
        values=decision_counts.values,
        names=decision_counts.index,
        title="매매 판단 비율",
        color=decision_counts.index,            # 색상을 설정할 기준
        color_discrete_map=color_map            # 색상 매핑
    )

    st.plotly_chart(fig)

#[5] 자산정보
with con32:
    
    tab1, tab2 = st.tabs(["보유 비트코인 수량", "보유 원화 잔액"])
    with tab1:
        fig = px.line(
            df, 
            x='timestamp', 
            y='btc_balance', 
            title='보유 비트코인 수량(BTC)',
            labels={
                'timestamp': '시간',
                'btc_balance': '비트코인 수량 (BTC)'
            }
        )
        fig.update_layout(
            yaxis=dict(
                tickformat=".4f", #소수점 고정
                title="비트코인 수량 (BTC)"
            )
        )
        st.plotly_chart(fig)
    
    with tab2:
        fig = px.line(
            df, 
            x='timestamp', 
            y='krw_balance', 
            title='보유 원화 잔액(원)',
            labels={
                'timestamp': '시간',
                'krw_balance': '원화 잔액 (원)'
            }
        )
        fig.update_layout(
            yaxis=dict(
                tickformat=",", #컴마 표시
                title="원화 잔액 (원)"
            )
        )
        st.plotly_chart(fig)
    
#[6] 비트코인 정보
with con42:
    fig = px.line(
        df, 
        x='timestamp', 
        y='btc_krw_price', 
        title='비트코인 현재가(원)',
        labels={
            'timestamp': '시간',
            'btc_krw_price': '비트코인 현재가 (원)'
        }
    )
    fig.update_layout(
            yaxis=dict(
                tickformat=",", #컴마 표시
                title="원화 잔액 (원)"
            )
        )
    st.plotly_chart(fig)
    
#[7] 하단
with con5:
    con51, con52, con53, con54 = st.columns([0.3, 0.3, 0.2, 0.2])
    
    with con51: #팀 정보
        st.markdown(f'<h1 style="font-size:17px;">{"(주) InduStat"}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="font-size:13px;">{"프로젝트 등록: 2024-02-DSCD-05"}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="font-size:13px;">{"팀장: 한진솔  |  팀원: 김정훈, 양태원, 이예슬"}</h1>', unsafe_allow_html=True)

    with con52: #프로젝트 안내사항
        st.markdown(f'<h1 style="font-size:17px;">{"Project"}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="font-size:13px;">{"서비스 이용약관"}</h1>', unsafe_allow_html=True)
        st.markdown(f'<h1 style="font-size:13px;">{"개인정보 처리방침"}</h1>', unsafe_allow_html=True)

    #동일한 크기로 이미지 설정
    target_size = (200, 100)  #폭:200 높이:100

    with con53:  #Upbit 로고
        upbit_logo = Image.open("upbit.jpg").resize(target_size)
        st.image(upbit_logo)

    with con54:  #InduStat 로고
        industat_logo = Image.open("InduStat.PNG.png").resize(target_size)
        st.image(industat_logo)
