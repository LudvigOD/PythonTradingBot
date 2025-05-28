from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from finbert_utils import estimate_sentiment
from math import floor
from typing import Tuple
from analytics import Analytics

#NOTE: Add api_keys
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

# Define the backtest period
start_date = datetime(2023, 5, 1)
end_date = datetime(2024, 6, 5)

# Set up the broker
broker = Alpaca(ALPACA_CREDS)

# Set up the strategy
symbol = "AMZN"

class MLTrader(Strategy):
    def initialize(self, symbol="NVDA", cash_at_risk: float = 0.5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        self.analytics = Analytics(self.symbol, start_=start_date.strftime('%Y-%m-%d'), end_=end_date.strftime('%Y-%m-%d'), interval_="1d")

    def position_sizing(self, symbol: str) -> Tuple[float, float, int]:
        cash = self.get_cash()
        last_price = self.get_last_price(symbol)
        quantity = floor(cash * self.cash_at_risk / last_price)
        return cash, last_price, quantity

    def get_dates(self) -> Tuple[str, str]:
        today = self.get_datetime()
        prev_three_days = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), prev_three_days.strftime('%Y-%m-%d')

    def get_sentiment(self, symbol: str) -> Tuple[float, str]:
        today, prev_three_days = self.get_dates()
        news = self.api.get_news(symbol=symbol, start=prev_three_days, end=today)
        headlines = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(headlines)
        return probability, sentiment

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing(self.symbol)
        probability, sentiment = self.get_sentiment(self.symbol)

        # Update analytics before each trading iteration
        self.analytics.calculate_analytics()
        self.analytics_valuation = self.analytics.get_valuation()

        if cash > last_price:
            if sentiment == "positive" and probability > 0.999 and self.analytics_valuation >= 3:
                if self.last_trade == "sell":
                    self.sell_all()

                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"

            elif sentiment == "negative" and probability > 0.999:
                if self.last_trade == "buy":
                    self.sell_all()

                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"

            elif sentiment == "neutral" and self.analytics_valuation >= 4:
                if self.last_trade == "sell":
                    self.sell_all()

                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"

strategy = MLTrader(name="mlstrat", broker=broker, parameters={"symbol": symbol, "cash_at_risk": 0.5})

# Perform the backtest
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters = {"symbol": symbol}
)
