import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import json
from openai import OpenAI
import ta
from ta.utils import dropna
import time
import requests
import base64
from PIL import Image
import io
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, WebDriverException, NoSuchElementException
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
import sqlite3
from datetime import datetime, timedelta
import schedule

# .env 파일에 저장된 환경 변수를 불러오기 (API 키 등)
load_dotenv()

# 로깅 설정 - 로그 레벨을 INFO로 설정하여 중요 정보 출력
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Upbit 객체 생성
access = os.getenv("UPBIT_ACCESS_KEY")
secret = os.getenv("UPBIT_SECRET_KEY")
if not access or not secret:
    logger.error("API keys not found. Please check your .env file.")
    raise ValueError("Missing API keys. Please check your .env file.")
upbit = pyupbit.Upbit(access, secret)

# OpenAI 구조화된 출력 체크용 클래스
class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str

# SQLite 데이터베이스 초기화 함수 - 거래 내역을 저장할 테이블을 생성
def init_db():
    conn = sqlite3.connect('bitcoin_trades.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  decision TEXT,
                  percentage INTEGER,
                  reason TEXT,
                  btc_balance REAL,
                  krw_balance REAL,
                  btc_avg_buy_price REAL,
                  btc_krw_price REAL,
                  reflection TEXT)''')
    conn.commit()
    return conn

# 거래 기록을 DB에 저장하는 함수
def log_trade(conn, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection=''):
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("""INSERT INTO trades 
                 (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price, reflection))
    conn.commit()

# 최근 투자 기록 조회
def get_recent_trades(conn, days=7):
    c = conn.cursor()
    seven_days_ago = (datetime.now() - timedelta(days=days)).isoformat()
    c.execute("SELECT * FROM trades WHERE timestamp > ? ORDER BY timestamp DESC", (seven_days_ago,))
    columns = [column[0] for column in c.description]
    return pd.DataFrame.from_records(data=c.fetchall(), columns=columns)

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    if trades_df.empty:
        return 0 # 기록이 없을 경우 0%로 설정
    # 초기 잔고 계산 (KRW + BTC * 현재 가격)
    initial_balance = trades_df.iloc[-1]['krw_balance'] + trades_df.iloc[-1]['btc_balance'] * trades_df.iloc[-1]['btc_krw_price']
    # 최종 잔고 계산
    final_balance = trades_df.iloc[0]['krw_balance'] + trades_df.iloc[0]['btc_balance'] * trades_df.iloc[0]['btc_krw_price']
    return (final_balance - initial_balance) / initial_balance * 100

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df) # 투자 퍼포먼스 계산
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None
    
    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df)
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an AI trading assistant tasked with analyzing recent trading performance and current market conditions to generate insights, probabilities, and improvements for future trading decisions."
            },
            {
                "role": "user",
                "content": f"""
                Recent trading data:
                {trades_df.to_json(orient='records')}
                
                Current market data:
                {current_market_data}
                
                Overall performance in the last 7 days: {performance:.2f}%
                
                Please analyze this data and provide:
                1. A brief reflection on the recent trading decisions.
                2. Probabilities for price movement:
                   - Probability of price increasing in the next 24 hours (%).
                   - Probability of price decreasing in the next 24 hours (%).
                3. Suggested decision (buy, sell, or hold) and percentage with reasoning.

                Ensure your response includes the probabilities as part of your reasoning and decision-making.
                Limit your response to 250 words or less.
                """
            }
        ]
    )


    try:
        response_content = response.choices[0].message.content
        return response_content
    except (IndexError, AttributeError) as e:
        logger.error(f"Error extracting response content: {e}")
        return None

# 데이터프레임에 보조 지표를 추가하는 함수
def add_indicators(df):
    # 볼린저 밴드 추가
    indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # RSI (Relative Strength Index) 추가
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # MACD (Moving Average Convergence Divergence) 추가
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # 이동평균선 (단기, 장기)
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    
    return df

# 공포 탐욕 지수 조회
def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['data'][0]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Fear and Greed Index: {e}")
        return None

# 뉴스 데이터 가져오기
def get_bitcoin_news():
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        logger.error("SERPAPI API key is missing.")
        return None  # 또는 함수 종료
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_news",
        "q": "btc",
        "api_key": serpapi_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        news_results = data.get("news_results", [])
        headlines = []
        for item in news_results:
            headlines.append({
                "title": item.get("title", ""),
                "date": item.get("date", "")
            })
        
        return headlines[:5]
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return []
        
# 유튜브 자막 데이터 가져오기
def get_combined_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
        combined_text = ' '.join(entry['text'] for entry in transcript)
        return combined_text
    except Exception as e:
        logger.error(f"Error fetching YouTube transcript: {e}")
        return ""

#### Selenium 관련 함수
def create_driver():
    env = os.getenv("ENVIRONMENT")
    logger.info("ChromeDriver 설정 중...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    try:
        if env == "local":
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
        elif env == "ec2":
            service = Service('/usr/bin/chromedriver')
        else:
            raise ValueError(f"Unsupported environment. Only local or ec2: {env}")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        logger.error(f"ChromeDriver 생성 중 오류 발생: {e}")
        raise

# XPath로 Element 찾기
def click_element_by_xpath(driver, xpath, element_name, wait_time=10):
    try:
        element = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        # 요소가 뷰포트에 보일 때까지 스크롤
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        # 요소가 클릭 가능할 때까지 대기
        element = WebDriverWait(driver, wait_time).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        element.click()
        logger.info(f"{element_name} 클릭 완료")
        time.sleep(2)  # 클릭 후 잠시 대기
    except TimeoutException:
        logger.error(f"{element_name} 요소를 찾는 데 시간이 초과되었습니다.")
    except ElementClickInterceptedException:
        logger.error(f"{element_name} 요소를 클릭할 수 없습니다. 다른 요소에 가려져 있을 수 있습니다.")
    except NoSuchElementException:
        logger.error(f"{element_name} 요소를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"{element_name} 클릭 중 오류 발생: {e}")
# 차트 클릭하기
def perform_chart_actions(driver):
    # 시간 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]",
        "시간 메뉴"
    )
    # 1시간 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]",
        "1시간 옵션"
    )
    # 지표 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    # 볼린저 밴드 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[15]",
        "볼린저 밴드 옵션"
    )
# 스크린샷 캡쳐 및 base64 이미지 인코딩
def capture_and_encode_screenshot(driver):
    try:
        # 스크린샷 캡처
        png = driver.get_screenshot_as_png()
        # PIL Image로 변환
        img = Image.open(io.BytesIO(png))
        # 이미지가 클 경우 리사이즈 (OpenAI API 제한에 맞춤)
        img.thumbnail((2000, 2000))
        # 이미지를 바이트로 변환
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        # base64로 인코딩
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_image
    except Exception as e:
        logger.error(f"스크린샷 캡처 및 인코딩 중 오류 발생: {e}")
        return None


def save_results_to_txt(probability_increase, pred_now, current_price, file_path="prediction_results.txt"):
    try:
        with open(file_path, "a", encoding="utf-8") as file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간
            file.write(f"Timestamp: {timestamp}\n")
            file.write(f"Probability of Increase: {probability_increase}%\n")
            file.write(f"Predicted Increase by LSTM: {pred_now}%\n")
            file.write(f"Current Bitcoin Price: {current_price} KRW\n")
            file.write("-" * 50 + "\n")  # 구분선
        logger.info("Prediction results saved to TXT file successfully.")
    except Exception as e:
        logger.error(f"Error saving prediction results to TXT file: {e}")

### 메인 AI 트레이딩 로직
def ai_trading():
    global upbit
    
    ### 데이터 가져오기
    # 1. 현재 투자 상태 조회
    all_balances = upbit.get_balances()
    filtered_balances = [balance for balance in all_balances if balance['currency'] in ['BTC', 'KRW']]
    
    # 2. 오더북(호가 데이터) 조회
    orderbook = pyupbit.get_orderbook("KRW-BTC")
    
    # 3. 차트 데이터 조회 및 보조지표 추가
    df_daily = pyupbit.get_ohlcv("KRW-BTC", interval="day", count=30)
    df_daily = dropna(df_daily)
    df_daily = add_indicators(df_daily)
    
    df_hourly = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=24)
    df_hourly = dropna(df_hourly)
    df_hourly = add_indicators(df_hourly)

    # 4. 공포 탐욕 지수 가져오기
    fear_greed_index = get_fear_and_greed_index()

    # 5. 뉴스 헤드라인 가져오기
    news_headlines = get_bitcoin_news()

    # 6. YouTube 자막 데이터 가져오기
    # youtube_transcript = get_combined_transcript("3XbtEX3jUv4")
    f = open("strategy.txt", "r", encoding="utf-8")
    youtube_transcript = f.read()
    f.close()

    # 7. Selenium으로 차트 캡처
    driver = None
    try:
        driver = create_driver()
        driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-BTC")
        logger.info("페이지 로드 완료")
        time.sleep(30)  # 페이지 로딩 대기 시간 증가
        logger.info("차트 작업 시작")
        perform_chart_actions(driver)
        logger.info("차트 작업 완료")
        chart_image = capture_and_encode_screenshot(driver)
        logger.info(f"스크린샷 캡처 완료.")
    except WebDriverException as e:
        logger.error(f"캡쳐시 WebDriver 오류 발생: {e}")
        chart_image = None
    except Exception as e:
        logger.error(f"차트 캡처 중 오류 발생: {e}")
        chart_image = None
    finally:
        if driver:
            driver.quit()

    ### AI에게 데이터 제공하고 판단 받기
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None
    try:
        # 데이터베이스 연결
        with sqlite3.connect('bitcoin_trades.db') as conn:
            # 최근 거래 내역 가져오기
            recent_trades = get_recent_trades(conn)
            
            # 현재 시장 데이터 수집 (기존 코드에서 가져온 데이터 사용)
            current_market_data = {
                "fear_greed_index": fear_greed_index,
                "news_headlines": news_headlines,
                "orderbook": orderbook,
                "daily_ohlcv": df_daily.to_dict(),
                "hourly_ohlcv": df_hourly.to_dict()
            }
            
            # 반성 및 개선 내용 생성
            reflection = generate_reflection(recent_trades, current_market_data)
            
            # AI 모델에 반성 내용 제공
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert in Bitcoin investing. Analyze the provided data and determine whether to buy, sell, or hold at the current moment. Consider the following in your analysis:

                        - Technical indicators and market data
                        - Recent news headlines and their potential impact on Bitcoin price
                        - The Fear and Greed Index and its implications
                        - Overall market sentiment
                        - Patterns and trends visible in the chart image
                        - Recent trading performance and reflection

                        Recent trading reflection:
                        {reflection}

                        Particularly important is to always refer to the trading method of 'Wonyyotti', a legendary Korean investor, to assess the current situation and make trading decisions. Wonyyotti's trading method is as follows:

                        {youtube_transcript}

                        Based on this trading method, analyze the current market situation and make a judgment by synthesizing it with the provided data and recent performance reflection.

                        Response format:
                        1. Decision (buy, sell, or hold)
                        2. If the decision is 'buy', provide a percentage (1-100) of available KRW to use for buying.
                        If the decision is 'sell', provide a percentage (1-100) of held BTC to sell.
                        If the decision is 'hold', set the percentage to 0.
                        3. Reason for your decision
                        4. Probability of price increase (%)
                        5. Probability of price decrease (%)

                        Ensure that the percentage is an integer between 1 and 100 for buy/sell decisions, and exactly 0 for hold decisions.
                        Your percentage should reflect the strength of your conviction in the decision based on the analyzed data."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Current investment status: {json.dumps(filtered_balances)}
                Orderbook: {json.dumps(orderbook)}
                Daily OHLCV with indicators (30 days): {df_daily.to_json()}
                Hourly OHLCV with indicators (24 hours): {df_hourly.to_json()}
                Recent news headlines: {json.dumps(news_headlines)}
                Fear and Greed Index: {json.dumps(fear_greed_index)}"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{chart_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "trading_decision",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "decision": {"type": "string", "enum": ["buy", "sell", "hold"]},
                                "percentage": {"type": "integer"},
                                "reason": {"type": "string"},
                                "probability_increase": {"type": "integer"},
                                "probability_decrease": {"type": "integer"}
                            },
                            "required": ["decision", "percentage", "reason", "probability_increase", "probability_decrease"],
                            "additionalProperties": False
                        }
                    }
                },
                max_tokens=4095
            )


            # Pydantic을 사용하여 AI의 트레이딩 결정 구조를 정의
            try:
                result = TradingDecision.model_validate_json(response.choices[0].message.content)
                response_data = json.loads(response.choices[0].message.content)
                probability_increase = response_data.get("probability_increase", 0)
                probability_decrease = response_data.get("probability_decrease", 0)
            except Exception as e:
                logger.error(f"Error parsing AI response: {e}")
                return
            
            logger.info(f"AI Decision: {result.decision.upper()}")
            logger.info(f"Decision Reason: {result.reason}")
            logger.info(f"Probability of Increase: {probability_increase}% ")
            logger.info(f"Probability of Decrease: {probability_decrease}% ")

            order_executed = False

            pd.options.display.float_format = '{:.6f}'.format
            import numpy as np
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler
            # 모델 로드는 전역으로 한 번만 수행
            model = tf.keras.models.load_model('lstm_final.h5')

            # 예측 함수를 tf.function으로 최적화
            @tf.function(reduce_retracing=True)
            def predict_step(x):
                return model(x, training=False)

            def expand_orderbook_units(row, max_units=15):
                new_columns = {}
                for i in range(max_units):
                    try:
                        unit = row['orderbook_units'][i]
                        new_columns[f'{i+1}_ask_price'] = unit['ask_price']
                        new_columns[f'{i+1}_bid_price'] = unit['bid_price']
                        new_columns[f'{i+1}_ask_size'] = unit['ask_size']
                        new_columns[f'{i+1}_bid_size'] = unit['bid_size']
                    except IndexError:
                        new_columns[f'{i+1}_ask_price'] = None
                        new_columns[f'{i+1}_bid_price'] = None
                        new_columns[f'{i+1}_ask_size'] = None
                        new_columns[f'{i+1}_bid_size'] = None
                return pd.Series(new_columns)

            def preprocessing_and_predict():
                orderbook_data = []
                current_prices = []
                required_size = 20
                sequence_length = 20
                max_attempts = 300
                attempts = 0
            
                try:
                    while len(orderbook_data) < required_size and attempts < max_attempts:
                        attempts += 1
                    
                        orderbook = pyupbit.get_orderbook("KRW-BTC")
                    
                        if orderbook:
                            current_price = orderbook['orderbook_units'][0]['ask_price']
                            current_prices.append(current_price)
                        
                            total_ask = sum(unit['ask_size'] for unit in orderbook['orderbook_units'])
                            total_bid = sum(unit['bid_size'] for unit in orderbook['orderbook_units'])
                        
                            orderbook_data.append({
                                'timestamp': int(time.time()),
                                'orderbook': orderbook,
                                'total_ask_size': total_ask,
                                'total_bid_size': total_bid,
                                'trade_volume': total_ask + total_bid
                            })
                    
                        time.sleep(0.1)
                    
                        if len(orderbook_data) % 5 == 0:
                            print(f"Collected {len(orderbook_data)}/{required_size} samples...")
                
                    if len(orderbook_data) < required_size:
                        print(f"Warning: Could not collect {required_size} samples. Collected {len(orderbook_data)} samples.")
                        if len(orderbook_data) == 0:
                            return None
                    else:
                        print(f"Successfully collected {required_size} samples!")
                    
                    df = pd.DataFrame(orderbook_data)
                    expanded = pd.concat([df['timestamp'], 
                                        df['orderbook'].apply(expand_orderbook_units)], axis=1)
                
                    expanded['trade_price'] = current_prices
                    expanded['trade_volume'] = df['trade_volume']
                    expanded['total_ask_size'] = df['total_ask_size']
                    expanded['total_bid_size'] = df['total_bid_size']
                
                    expanded['spread'] = expanded[[f'{i+1}_ask_price' for i in range(5)]].min(axis=1) - expanded[[f'{i+1}_bid_price' for i in range(5)]].max(axis=1)
                    expanded['imbalance'] = (expanded['total_bid_size'] - expanded['total_ask_size']) / (expanded['total_bid_size'] + expanded['total_ask_size'])
                    expanded['totalSize_ratio'] = expanded['total_bid_size'] / expanded['total_ask_size']
                
                    cols_to_keep = ['timestamp', 'total_ask_size', 'total_bid_size', 
                                    '1_ask_size', '1_bid_size', '2_ask_size', '2_bid_size', '3_ask_size', '3_bid_size',
                                    '4_ask_size', '4_bid_size', '5_ask_size', '5_bid_size',
                                    'trade_price', 'trade_volume', 'spread', 'imbalance', 'totalSize_ratio']
                
                    df = expanded[cols_to_keep]
                
                    X = df.values
                
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                
                    sequences = []
                    for i in range(len(X_scaled) - sequence_length + 1):
                        sequences.append(X_scaled[i:i + sequence_length])
                
                    X_sequence = np.array(sequences)[-1:]
                
                    # numpy array를 tensorflow tensor로 변환
                    X_tensor = tf.convert_to_tensor(X_sequence, dtype=tf.float32)
                
                    # 최적화된 예측 함수 사용
                    pred = predict_step(X_tensor)
                
                    return float(pred[0][0])  # tensor를 python float로 변환
                
                except Exception as e:
                    print(f"Error during data collection: {e}")
                    return None
            pred_now = preprocessing_and_predict()

            current_price = pyupbit.get_current_price("KRW-BTC")
            print(f"### Current Bitcoin Price: {current_price} KRW ###")
            
            print(f"### Probability of Increase: {probability_increase}% ###")
            print(f"### Probability of Increase: {pred_now}% ###")
            save_results_to_txt(probability_increase, pred_now, current_price)  # 새로 추가된 부분
            probability_increase = probability_increase *0.01

            decision_value = probability_increase * 0.3 + pred_now * 0.7

            def calculate_percentage(decision_value, min_value, max_value):
                if decision_value >= 0.6:
                    percentage = min_value + (max_value - min_value) * ((decision_value - 0.6) / 0.4)
                    return percentage
                elif decision_value <= 0.4:
                    percentage = min_value + (max_value - min_value) * ((0.4 - decision_value) / 0.4)
                    return percentage
                else:
                    return 0
 
            # 매매 결정 및 실행
            if decision_value > 0.6:
                decision = "buy"
                percentage = calculate_percentage(decision_value, 0.3,0.5)
                percentage = percentage *100
            elif decision_value < 0.4:
                decision = "sell"
                percentage = calculate_percentage(decision_value, 0.3, 0.5)
                percentage = percentage *100
            else:
                decision = "hold"
                percentage = 0

            logger.info(f"Decision: {decision.upper()}, Percentage: {percentage}%")

            order_executed = False
            if decision == "buy":
                my_krw = upbit.get_balance("KRW")
                if my_krw is None:
                    logger.error("Failed to retrieve KRW balance.")
                else:
                    buy_amount = my_krw * (percentage / 100) * 0.9995  # 수수료 고려
                    if buy_amount > 5000:
                        logger.info(f"Buy Order Executed: {percentage}% of available KRW")
                        try:
                            order = upbit.buy_market_order("KRW-BTC", buy_amount)
                            if order:
                                logger.info(f"Buy order executed successfully: {order}")
                                order_executed = True
                            else:
                                logger.error("Buy order failed.")
                        except Exception as e:
                            logger.error(f"Error executing buy order: {e}")
                    else:
                        logger.warning("Buy Order Failed: Insufficient KRW (less than 5000 KRW)")

            elif decision == "sell":
                my_btc = upbit.get_balance("KRW-BTC")
                if my_btc is None:
                    logger.error("Failed to retrieve BTC balance.")
                else:
                    sell_amount = my_btc * (percentage / 100)
                    current_price = pyupbit.get_current_price("KRW-BTC")
                    if sell_amount * current_price > 5000:
                        logger.info(f"Sell Order Executed: {percentage}% of held BTC")
                        try:
                            order = upbit.sell_market_order("KRW-BTC", sell_amount)
                            if order:
                                logger.info(f"Sell order executed successfully: {order}")
                                order_executed = True
                            else:
                                logger.error("Sell order failed.")
                        except Exception as e:
                            logger.error(f"Error executing sell order: {e}")
                    else:
                        logger.warning("Sell Order Failed: Insufficient BTC (less than 5000 KRW worth)")

            
            # 거래 기록 저장
            time.sleep(2)  # API 호출 제한 고려
            balances = upbit.get_balances()
            btc_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'BTC'), 0)
            krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
            btc_avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'BTC'), 0)
            current_btc_price = pyupbit.get_current_price("KRW-BTC")

            log_trade(conn, decision, percentage if order_executed else 0, 
                    f"Decision based on value: {decision_value:.2f}", 
                    btc_balance, krw_balance, btc_avg_buy_price, current_btc_price)   
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        return

if __name__ == "__main__":
    # 데이터베이스 초기화
    init_db()

    # 중복 실행 방지를 위한 변수
    trading_in_progress = False

    # 트레이딩 작업을 수행하는 함수
    def job():
        global trading_in_progress
        if trading_in_progress:
            logger.warning("Trading job is already in progress, skipping this run.")
            return
        try:
            trading_in_progress = True
            ai_trading()
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            trading_in_progress = False

    ## 테스트용 바로 실행
    # job()

    ## 매일 특정 시간(예: 오전 9시, 오후 3시, 오후 9시)에 실행
    
    schedule.every().day.at("09:00").do(job)
    schedule.every().day.at("09:10").do(job)
    schedule.every().day.at("09:20").do(job)
    schedule.every().day.at("09:30").do(job)
    schedule.every().day.at("09:40").do(job)
    schedule.every().day.at("09:50").do(job)
    schedule.every().day.at("10:00").do(job)
    schedule.every().day.at("10:10").do(job)
    schedule.every().day.at("10:20").do(job)
    schedule.every().day.at("10:30").do(job)
    schedule.every().day.at("10:40").do(job)
    schedule.every().day.at("10:50").do(job)
    schedule.every().day.at("11:00").do(job)
    schedule.every().day.at("11:10").do(job)
    schedule.every().day.at("11:20").do(job)
    schedule.every().day.at("11:30").do(job)
    schedule.every().day.at("11:40").do(job)
    schedule.every().day.at("11:50").do(job)
    schedule.every().day.at("12:00").do(job)
    schedule.every().day.at("12:10").do(job)
    schedule.every().day.at("12:20").do(job)
    schedule.every().day.at("12:30").do(job)
    schedule.every().day.at("12:40").do(job)
    schedule.every().day.at("12:50").do(job)
    schedule.every().day.at("13:00").do(job)
    schedule.every().day.at("13:10").do(job)
    schedule.every().day.at("13:20").do(job)
    schedule.every().day.at("13:30").do(job)
    schedule.every().day.at("13:40").do(job)
    schedule.every().day.at("13:50").do(job)
    schedule.every().day.at("14:00").do(job)
    schedule.every().day.at("14:10").do(job)
    schedule.every().day.at("14:20").do(job)
    schedule.every().day.at("14:30").do(job)
    schedule.every().day.at("14:40").do(job)
    schedule.every().day.at("14:50").do(job)
    schedule.every().day.at("15:00").do(job)
    schedule.every().day.at("15:10").do(job)
    schedule.every().day.at("15:20").do(job)
    schedule.every().day.at("15:30").do(job)
    schedule.every().day.at("15:40").do(job)
    schedule.every().day.at("15:50").do(job)
    schedule.every().day.at("16:00").do(job)
    schedule.every().day.at("16:10").do(job)
    schedule.every().day.at("16:20").do(job)
    schedule.every().day.at("16:30").do(job)
    schedule.every().day.at("16:40").do(job)
    schedule.every().day.at("16:50").do(job)
    schedule.every().day.at("17:00").do(job)
    schedule.every().day.at("17:10").do(job)
    schedule.every().day.at("17:20").do(job)
    schedule.every().day.at("17:30").do(job)
    schedule.every().day.at("17:40").do(job)
    schedule.every().day.at("17:50").do(job)
    schedule.every().day.at("18:00").do(job)
    schedule.every().day.at("18:10").do(job)
    schedule.every().day.at("18:20").do(job)
    schedule.every().day.at("18:30").do(job)
    schedule.every().day.at("18:40").do(job)
    schedule.every().day.at("18:50").do(job)
    schedule.every().day.at("19:00").do(job)
    schedule.every().day.at("19:10").do(job)
    schedule.every().day.at("19:20").do(job)
    schedule.every().day.at("19:30").do(job)
    schedule.every().day.at("19:40").do(job)
    schedule.every().day.at("19:50").do(job)
    schedule.every().day.at("20:00").do(job)
    schedule.every().day.at("20:10").do(job)
    schedule.every().day.at("20:20").do(job)
    schedule.every().day.at("20:30").do(job)
    schedule.every().day.at("20:40").do(job)
    schedule.every().day.at("20:50").do(job)
    schedule.every().day.at("21:00").do(job)
    schedule.every().day.at("21:10").do(job)
    schedule.every().day.at("21:20").do(job)
    schedule.every().day.at("21:30").do(job)
    schedule.every().day.at("21:40").do(job)
    schedule.every().day.at("21:50").do(job)
    schedule.every().day.at("22:00").do(job)
    schedule.every().day.at("22:10").do(job)
    schedule.every().day.at("22:20").do(job)
    schedule.every().day.at("22:30").do(job)
    schedule.every().day.at("22:40").do(job)
    schedule.every().day.at("22:50").do(job)
    schedule.every().day.at("23:00").do(job)
    schedule.every().day.at("23:10").do(job)
    schedule.every().day.at("23:20").do(job)
    schedule.every().day.at("23:30").do(job)
    schedule.every().day.at("23:40").do(job)
    schedule.every().day.at("23:50").do(job)
    schedule.every().day.at("00:00").do(job)
    schedule.every().day.at("00:10").do(job)
    schedule.every().day.at("00:20").do(job)
    schedule.every().day.at("00:30").do(job)
    schedule.every().day.at("00:40").do(job)
    schedule.every().day.at("00:50").do(job)
    schedule.every().day.at("01:00").do(job)
    schedule.every().day.at("01:10").do(job)
    schedule.every().day.at("01:20").do(job)
    schedule.every().day.at("01:30").do(job)
    schedule.every().day.at("01:40").do(job)
    schedule.every().day.at("01:50").do(job)
    schedule.every().day.at("02:00").do(job)
    schedule.every().day.at("02:10").do(job)
    schedule.every().day.at("02:20").do(job)
    schedule.every().day.at("02:30").do(job)
    schedule.every().day.at("02:40").do(job)
    schedule.every().day.at("02:50").do(job)
    schedule.every().day.at("03:00").do(job)
    schedule.every().day.at("03:10").do(job)
    schedule.every().day.at("03:20").do(job)
    schedule.every().day.at("03:30").do(job)
    schedule.every().day.at("03:40").do(job)
    schedule.every().day.at("03:50").do(job)
    schedule.every().day.at("04:00").do(job)
    schedule.every().day.at("04:10").do(job)
    schedule.every().day.at("04:20").do(job)
    schedule.every().day.at("04:30").do(job)
    schedule.every().day.at("04:40").do(job)
    schedule.every().day.at("04:50").do(job)
    schedule.every().day.at("05:00").do(job)
    schedule.every().day.at("05:10").do(job)
    schedule.every().day.at("05:20").do(job)
    schedule.every().day.at("05:30").do(job)
    schedule.every().day.at("05:40").do(job)
    schedule.every().day.at("05:50").do(job)
    schedule.every().day.at("06:00").do(job)
    schedule.every().day.at("06:10").do(job)
    schedule.every().day.at("06:20").do(job)
    schedule.every().day.at("06:30").do(job)
    schedule.every().day.at("06:40").do(job)
    schedule.every().day.at("06:50").do(job)
    schedule.every().day.at("07:00").do(job)
    schedule.every().day.at("07:10").do(job)
    schedule.every().day.at("07:20").do(job)
    schedule.every().day.at("07:30").do(job)
    schedule.every().day.at("07:40").do(job)
    schedule.every().day.at("07:50").do(job)
    schedule.every().day.at("08:00").do(job)
    schedule.every().day.at("08:10").do(job)
    schedule.every().day.at("08:20").do(job)
    schedule.every().day.at("08:30").do(job)
    schedule.every().day.at("08:40").do(job)
    schedule.every().day.at("08:50").do(job)    
    while True:
         schedule.run_pending()
         time.sleep(1)
