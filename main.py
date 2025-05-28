import pandas as pd
import ta
import yfinance as yf
from datetime import datetime, timedelta
from math import floor
from typing import Tuple
from alpaca_trade_api import REST
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time

# Alpaca API credentials
# NOTE: Here were api keys, even the private... (;
BASE_URL = "https://paper-api.alpaca.markets/"

# Alpaca API initialization
api = REST(API_KEY, API_SECRET, BASE_URL)

# FinBERT initialization for sentiment analysis
device = "cuda" if torch.cuda.is_available() else "cpu" # i have no cuda
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert") # Tokenizer basically converts data/string to numeric form that a neural network understands called Token ID.
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device) # Use token IDs to make pridiction of sentiment with pretrained model.
labels = ["positive", "negative", "neutral"] # Conclusion labels for finBERT output

def estimate_sentiment(news: list) -> Tuple[float, str]:
    if news:
        # Tokenize list of strings into token IDs
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device) # Get tokens
        logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"] # Use tokenID to get logits which are scores for each sentiment class 
        summed = torch.sum(logits, dim=0) 
        probs = torch.nn.functional.softmax(summed, dim=-1) # Converts to probablities
        idx = torch.argmax(probs).item() # Selects the highest probability
        return probs[idx].item(), labels[idx]
    else:
        return 0.0, labels[-1] # else neutral

class Analytics: # Bunch of analytics (dont quote me!)
    def __init__(self, symbol: str, start_: str, end_: str, interval_: str = "1d"):
        self.dataF = yf.download(symbol, start=start_, end=end_, interval=interval_)
        if not self.dataF.empty:
            self.calculate_analytics()
        else:
            raise ValueError("No data retrieved, check your symbol and date range")

    def calculate_analytics(self):
        self.dataF['MA20'] = ta.trend.sma_indicator(self.dataF['Close'], window=20)
        self.dataF['MA50'] = ta.trend.sma_indicator(self.dataF['Close'], window=50)
        self.dataF['RSI'] = ta.momentum.rsi(self.dataF['Close'], window=14)
        self.dataF['MACD'] = ta.trend.macd(self.dataF['Close'])
        self.dataF['MACD_Signal'] = ta.trend.macd_signal(self.dataF['Close'])
        bb = ta.volatility.BollingerBands(self.dataF['Close'], window=20, window_dev=2)
        self.dataF['BB_Upper'] = bb.bollinger_hband()
        self.dataF['BB_Lower'] = bb.bollinger_lband()
        self.dataF['ATR'] = ta.volatility.average_true_range(self.dataF['High'], self.dataF['Low'], self.dataF['Close'], window=14)
        self.dataF['VWAP'] = ta.volume.volume_weighted_average_price(self.dataF['High'], self.dataF['Low'], self.dataF['Close'], self.dataF['Volume'])
        self.dataF['Momentum'] = ta.momentum.awesome_oscillator(self.dataF['High'], self.dataF['Low'])

    def get_valuation(self):
      decider = 0
      if (self.dataF['MA20'] > self.dataF['MA50']).sum() > len(self.dataF) / 2:
          decider += 1
      if 50 <= self.dataF['RSI'].iloc[-1] <= 70:
          decider += 1
      if (self.dataF['MACD'] > self.dataF['MACD_Signal']).iloc[-1] and (self.dataF['MACD'].shift(1) <= self.dataF['MACD_Signal'].shift(1)).iloc[-1]:
          decider += 1
      price = self.dataF['Close']
      lower_band = self.dataF['BB_Lower']
      upper_band = self.dataF['BB_Upper']
      above_lower_band_percentage = (price > lower_band).mean()
      touch_upper_band_percentage = (price >= upper_band).mean()
      if above_lower_band_percentage >= 0.80 or touch_upper_band_percentage >= 0.20:
          decider += 1
      if price.iloc[-1] > self.dataF['VWAP'].iloc[-1]:
          decider += 1
      if price.iloc[-1] > self.dataF['VWAP'].iloc[-1] and price.iloc[-1] > (price.iloc[-2] + self.dataF['ATR'].iloc[-2]):
          decider += 1
      if self.dataF['Momentum'].iloc[-1] > 0:          
        decider += 1
      return decider

class TradingBot:
    def __init__(self, symbol: str, take_profit_pct, stop_loss_pct, cash_at_risk: float = 0.1): 
        self.symbol = symbol # Company
        self.cash_at_risk = cash_at_risk # Money willing to lose
        self.last_trade = None
        self.analytics = Analytics(symbol, start_=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), end_=datetime.now().strftime('%Y-%m-%d')) # Use one year of info for analytics
        self.take_profit_pct = take_profit_pct # when to get out with a win
        self.stop_loss_pct = stop_loss_pct # when to get out with a loss

    def get_cash(self):
        account = api.get_account()
        return float(account.cash)


    # Get last price of company
    def get_last_price(self):
      last_trade = api.get_latest_trade(self.symbol)
      if last_trade:
          return float(last_trade.price)
      else:
          return None


    # Returns your cash, last trading price, the quantity you can buy of company with cash_at_risk in mind
    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price()
        quantity = floor(cash * self.cash_at_risk / last_price)

        if quantity > 0:
          if cash < last_price * quantity:
              quantity = floor(cash / last_price)


        return cash, last_price, quantity

    # Interval between NOW and 3 days prior
    def get_dates(self):
        today = datetime.now()
        prev_three_days = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), prev_three_days.strftime('%Y-%m-%d')

    # Get sentimate of three days
    def get_sentiment(self):
        today, prev_three_days = self.get_dates()
        news = api.get_news(symbol=self.symbol, start=prev_three_days, end=today)
        headlines = [ev.headline for ev in news]
        return estimate_sentiment(headlines)

    # Main trading function
    def trade(self):
        cash, last_price, quantity = self.position_sizing() 
        probability, sentiment = self.get_sentiment() # sentiment
        valuation = self.analytics.get_valuation() # Alanytics valuation
        print(f"The quantity: {quantity}")

        if quantity <= 0:
          print(f"Skipping trade for {self.symbol}: Insufficient cash or negative quantity.")
          return

        # Using diffrent weights to choose buying or selling
        if cash > last_price:
            if sentiment == "positive" and probability > 0.999 and valuation >= 3:
                if self.last_trade == "sell":
                    self.sell_all()
                self.buy(quantity, last_price)
            elif sentiment == "negative" and probability > 0.999:
                if self.last_trade == "buy":
                    self.sell_all()
                self.sell(quantity, last_price)
            elif sentiment == "neutral" and valuation >= 4:
                if self.last_trade == "sell":
                    self.sell_all()
                self.buy(quantity, last_price)
            elif sentiment == "neutral" and valuation <= 1:
                if self.last_trade == "buy":
                    self.sell_all()
                self.sell(quantity, last_price)

    def buy(self, quantity, last_price):
      
      cash = self.get_cash()

      take_profit_price = round(last_price * (1 + self.take_profit_pct), 2)
      stop_loss_price = round(last_price * (1 - self.stop_loss_pct), 2)


      if cash < last_price * quantity:
          quantity = floor(cash / last_price)

      api.submit_order(
          symbol=self.symbol,
          qty=quantity,
          side='buy',
          type='limit',
          time_in_force='gtc',
          limit_price=round(last_price, 2),
          order_class='bracket',
          take_profit=dict(limit_price=take_profit_price),
          stop_loss=dict(stop_price=stop_loss_price)
      )
      self.last_trade = "buy"
      print(f"Bought {self.symbol} at {last_price}, TP: {take_profit_price}, SL: {stop_loss_price}")


    def sell(self, quantity, last_price):
      api.submit_order(
          symbol=self.symbol,
          qty=quantity,
          side='sell',
          type='limit',
          time_in_force='gtc',
          limit_price=round(last_price, 2)
      )
      self.last_trade = "sell"
      print(f"Sold {self.symbol} at {last_price}")

    def sell_all(self):
        position = api.get_position(self.symbol)
        if position:
            quantity = int(position.qty)
            last_price = self.get_last_price()
            self.sell(quantity, last_price)


symbols = ["AAPL", "NVDA", "AMZN", "TSLA", "MSFT", "AMD", "INTC", "QCOM"]
win_prc = 0.1
loss_prc = 0.05


if __name__ == "__main__":
    sleep_time = 900*4
    win_prc = 0.07
    loss_prc = 0.03

    bots = [TradingBot(symbol=symbol_, take_profit_pct=win_prc, stop_loss_pct=loss_prc) for symbol_ in symbols]

    while True:
        for bot in bots:
            bot.trade()
        time.sleep(sleep_time)
