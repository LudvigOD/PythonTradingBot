import pandas as pd
import ta
import yfinance as yf
from datetime import datetime

class Analytics:
    crnt_date = datetime.now().strftime("%Y-%m-%d")

    def __init__(self, symbol: str, start_: str = crnt_date, end_: str = crnt_date, interval_="1d"):
        self.dataF: pd.DataFrame = yf.download(symbol, start=start_, end=end_, interval=interval_)
        if not self.dataF.empty:
            self.calculate_analytics()
        else:
            raise ValueError("No data retrieved, check your symbol and date range")
        
    

    def calculate_analytics(self):
        # Moving Averages
        self.dataF['MA20'] = ta.trend.sma_indicator(self.dataF['Close'], window=20)
        self.dataF['MA50'] = ta.trend.sma_indicator(self.dataF['Close'], window=50)
        
        # Relative Strength Index (RSI)
        self.dataF['RSI'] = ta.momentum.rsi(self.dataF['Close'], window=14)
        
        # Moving Average Convergence Divergence (MACD)
        self.dataF['MACD'] = ta.trend.macd(self.dataF['Close'])
        self.dataF['MACD_Signal'] = ta.trend.macd_signal(self.dataF['Close'])
        
        # Bollinger Bands (BBANDS)
        bb = ta.volatility.BollingerBands(self.dataF['Close'], window=20, window_dev=2)
        self.dataF['BB_Upper'] = bb.bollinger_hband()
        self.dataF['BB_Lower'] = bb.bollinger_lband()
        
        # Average True Range (ATR)
        self.dataF['ATR'] = ta.volatility.average_true_range(self.dataF['High'], self.dataF['Low'], self.dataF['Close'], window=14)
        
        # Volume Indicators (e.g., Volume Weighted Average Price)
        self.dataF['VWAP'] = ta.volume.volume_weighted_average_price(self.dataF['High'], self.dataF['Low'], self.dataF['Close'], self.dataF['Volume'])

    # ma20/ma50 dataframe in order to compare
    def eval_ma2050(self):
        cleaned_MA20MA50 = self.dataF[['MA20', 'MA50']].dropna()
        checked = (cleaned_MA20MA50['MA20'] > cleaned_MA20MA50['MA50']).sum()
        rows = cleaned_MA20MA50.shape[0]
        return checked > rows / 2

    # if rsi > 50 but > 70 -> buy
    def eval_rsi(self):
        cleaned_rsi = self.dataF['RSI'].dropna()
        return 50 <= cleaned_rsi.iloc[-1] <= 70
    
    def eval_macd_bullish_crossover(self):
        macd = self.dataF['MACD']
        signal = self.dataF['MACD_Signal']
        bullish_crossovers = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        return bullish_crossovers.iloc[-1]
        
    def eval_bbands(self):
        price = self.dataF['Close']
        lower_band = self.dataF['BB_Lower']
        upper_band = self.dataF['BB_Upper']
        
        above_lower_band_percentage = (price > lower_band).mean()
        touch_upper_band_percentage = (price >= upper_band).mean()
        
        # You can adjust these thresholds according to your strategy
        above_lower_threshold = 0.80
        touch_upper_threshold = 0.20
        
        above_lower_band = above_lower_band_percentage >= above_lower_threshold
        touch_upper_band = touch_upper_band_percentage >= touch_upper_threshold
        
        return above_lower_band or touch_upper_band

    def eval_vwap(self):
        price = self.dataF['Close']
        vwap = self.dataF['VWAP']
        return price.iloc[-1] > vwap.iloc[-1]
    
    def get_valuation(self):
        decider = 0
        if self.eval_ma2050():
            decider += 1
        if self.eval_rsi():
            decider += 1
        if self.eval_macd_bullish_crossover():
            decider += 1
        if self.eval_bbands():
            decider += 1
        if self.eval_vwap():
            decider += 1
        return decider

    def get_data(self):
        return self.dataF

if __name__ == "__main__":
    analytics = Analytics("AAPL", start_="2024-01-01")
    analytics.calculate_analytics()
    print(analytics.dataF[["RSI"]])
