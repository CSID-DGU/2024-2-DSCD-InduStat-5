import os
from dotenv import load_dotenv
load_dotenv()

def ai_trading():
    #1. 업비트 차트 데이터 가져오기
    import pyupbit
    df = pyupbit.get_ohlcv("KRW-BTC",count=30, interval='day')


    #2. ai에게 데이터 제공하고 판단받기
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "You are an expert in Bitcoin investing. Tell me whether I should buy, sell, or hold at the moment based on the chart data provided.\n\nResponse Example:\n{\"decision\" : \"buy\" , \"reason\" : \"some technical reason\"}\n{\"decision\" : \"sell\" , \"reason\" : \"some technical reason\"}\n{\"decision\" : \"hold\" , \"reason\" : \"some technical reason\"}\n"
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": df.to_json()
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "You are an expert in Bitcoin investing. Tell me whether I should buy, sell, or hold at the moment based on the chart data provided. response in json"
            }
        ]
        },
        {
        "role": "assistant",
        "content": [
            {
            "type": "text",
            "text": "{\"decision\": \"hold\", \"reason\": \"The chart data shows volatility with recent declines in both price and volume, suggesting uncertainty in market direction. A 'hold' decision is recommended until clearer trends develop.\"}"
            }
        ]
        }
    ],

    response_format={
        "type": "json_object"
    }
    )

    result = response.choices[0].message.content

    #3.AI 판단에 따라 실제로 자동매매 진행하기
    import json
    result = json.loads(result)

    import pyupbit
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    upbit =pyupbit.Upbit(access, secret)

    print("### AI Decision: ",result["decision"].upper(),"###")
    print(f"### Reason: {result['reason']} ###")

    if result["decision"] =="buy" :
        my_krw = upbit.get_balance("KRW")
        #매수
        if my_krw*0.9995 >5000:
            print(upbit.buy_market_order("KRW-XRP", my_krw*0.9995))
            print("buy:",result["reason"])
        else:
            print("실패 : krw 5000미만")
    elif result["decision"] =="sell":
        my_btc = upbit.get_balance("KRW-BTC")
        current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]
        if my_btc*current_price >5000:
        #매도
        
            print(upbit.sell_market_order("KRW-BTC", upbit.get_balance("KRW-BTC")))
            print("sell:",result["reason"])
        else:
            print("실패: btc 5000미만")
    elif  result["decision"] :
        #지나감
        print("hold:",result["reason"])

while True:
    import time
    time.sleep(10)
    ai_trading()

