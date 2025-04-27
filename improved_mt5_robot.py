import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),
        logging.StreamHandler()
    ]
)

class EnhancedTradingStrategy:
    def __init__(self, ativo, timeframe, lote):
        self.ativo = ativo
        self.timeframe = self.converter_timeframe(timeframe)
        self.lote = float(lote)
        self.operando = True
        self.ticket_atual = None
        self.logger = logging.getLogger(__name__)

        # Enhanced configuration parameters
        self.rsi_sobrecomprado = 70
        self.rsi_sobrevendido = 30
        self.bb_desvio = 2.0
        self.atr_period = 14
        self.stoch_period = 14
        self.volume_threshold = 1.5
        self.profit_factor = 2.5  # Risk:Reward ratio
        self.max_daily_trades = 5
        self.trades_today = 0
        self.last_trade_time = None
        self.min_trade_interval = 300  # 5 minutes in seconds

    def converter_timeframe(self, tf):
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return mapping.get(tf, mt5.TIMEFRAME_M5)

    def check_market_conditions(self):
        """Check if market conditions are suitable for trading"""
        try:
            # Get symbol information
            symbol_info = mt5.symbol_info(self.ativo)
            if symbol_info is None:
                self.logger.error(f"Failed to get symbol info for {self.ativo}")
                return False

            # Check if market is open
            if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                self.logger.warning("Market is closed or trading is restricted")
                return False

            # Check spread
            current_spread = symbol_info.spread
            max_allowed_spread = 20  # Maximum allowed spread in points
            if current_spread > max_allowed_spread:
                self.logger.warning(f"Spread too high: {current_spread} points")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {e}")
            return False

    def executar(self):
        self.logger.info("🚀 Starting Enhanced Trading Strategy")
        self.logger.info(f"Trading {self.ativo} on {self.timeframe} timeframe")

        while self.operando:
            try:
                # Reset daily trades at market open
                current_time = datetime.now()
                if current_time.hour == 0 and current_time.minute == 0:
                    self.trades_today = 0

                # Check if maximum daily trades reached
                if self.trades_today >= self.max_daily_trades:
                    self.logger.info("Maximum daily trades reached")
                    time.sleep(60)
                    continue

                # Check market conditions
                if not self.check_market_conditions():
                    time.sleep(60)
                    continue

                self.logger.info("📊 Analyzing market conditions...")
                self.analisar_e_operar()
                time.sleep(5)

            except Exception as e:
                self.logger.error(f"❌ Strategy error: {e}")
                time.sleep(10)

    def analisar_e_operar(self):
        # Market Analysis Phase
        self.logger.info("🔍 Starting market analysis...")
        
        barras = mt5.copy_rates_from_pos(self.ativo, self.timeframe, 0, 200)
        if barras is None or len(barras) < 100:
            self.logger.error(f"❌ Failed to load candles for {self.ativo}")
            return

        df = pd.DataFrame(barras)
        
        # Technical Analysis
        self.logger.info("📈 Calculating technical indicators...")
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values

        # Enhanced Indicators
        ema9 = self.ema(close, 9)
        ema21 = self.ema(close, 21)
        ema50 = self.ema(close, 50)
        
        macd_line, signal_line = self.macd(close)
        rsi_valores = self.rsi(close, 14)
        bb_superior, bb_medio, bb_inferior = self.bollinger_bands(close, 20, self.bb_desvio)
        stoch_k, stoch_d = self.stochastic(high, low, close, self.stoch_period)
        atr = self.atr(high, low, close, self.atr_period)
        
        # Volume Analysis
        volume_ma = np.mean(volume[-20:])
        volume_atual = volume[-1]
        volume_alto = volume_atual > (volume_ma * self.volume_threshold)
        
        # Market Conditions Analysis
        self.logger.info("🎯 Analyzing entry conditions...")
        
        tendencia_alta = (ema9[-1] > ema21[-1] > ema50[-1]) and (close[-1] > ema9[-1])
        tendencia_baixa = (ema9[-1] < ema21[-1] < ema50[-1]) and (close[-1] < ema9[-1])

        # Enhanced Entry Conditions
        sinal_compra = (
            tendencia_alta and
            rsi_valores[-1] < 40 and  # More conservative RSI levels
            stoch_k[-1] < 30 and
            volume_alto and
            close[-1] < bb_medio[-1] and
            macd_line[-1] > macd_line[-2]  # MACD momentum
        )

        sinal_venda = (
            tendencia_baixa and
            rsi_valores[-1] > 60 and
            stoch_k[-1] > 70 and
            volume_alto and
            close[-1] > bb_medio[-1] and
            macd_line[-1] < macd_line[-2]
        )

        # Trade Execution with Time Check
        current_time = time.time()
        if self.last_trade_time and (current_time - self.last_trade_time) < self.min_trade_interval:
            self.logger.info("⏳ Waiting for minimum trade interval...")
            return

        # Dynamic position sizing based on ATR
        atr_atual = atr[-1]
        
        if sinal_compra:
            self.logger.info("🔵 Strong BUY signal detected!")
            self.logger.info(f"RSI: {rsi_valores[-1]:.2f}, Stochastic: {stoch_k[-1]:.2f}")
            sl_distance = atr_atual * 1.5
            tp_distance = atr_atual * self.profit_factor
            self.abrir_ordem(mt5.ORDER_TYPE_BUY, sl_distance, tp_distance)
            
        elif sinal_venda:
            self.logger.info("🔴 Strong SELL signal detected!")
            self.logger.info(f"RSI: {rsi_valores[-1]:.2f}, Stochastic: {stoch_k[-1]:.2f}")
            sl_distance = atr_atual * 1.5
            tp_distance = atr_atual * self.profit_factor
            self.abrir_ordem(mt5.ORDER_TYPE_SELL, sl_distance, tp_distance)
        
        else:
            self.logger.info("⏸️ No clear trading signals at the moment")

    def abrir_ordem(self, tipo_ordem, sl_distance, tp_distance):
        """Enhanced order execution with additional safety checks"""
        try:
            # Get current market price
            tick = mt5.symbol_info_tick(self.ativo)
            if tick is None:
                self.logger.error("❌ Failed to get current price")
                return

            # Calculate entry price
            preco = tick.ask if tipo_ordem == mt5.ORDER_TYPE_BUY else tick.bid
            point = mt5.symbol_info(self.ativo).point

            # Calculate SL and TP levels
            sl = preco - sl_distance * point if tipo_ordem == mt5.ORDER_TYPE_BUY else preco + sl_distance * point
            tp = preco + tp_distance * point if tipo_ordem == mt5.ORDER_TYPE_BUY else preco - tp_distance * point

            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.ativo,
                "volume": self.lote,
                "type": tipo_ordem,
                "price": preco,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": 123456,
                "comment": "Enhanced MT5 Robot v3",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Execute trade
            resultado = mt5.order_send(request)

            if resultado.retcode != mt5.TRADE_RETCODE_DONE:
                self.logger.error(f"❌ Order failed: {resultado.comment}")
                return

            # Update trade tracking
            self.ticket_atual = resultado.order
            self.trades_today += 1
            self.last_trade_time = time.time()

            # Log trade details
            direcao = "COMPRA" if tipo_ordem == mt5.ORDER_TYPE_BUY else "VENDA"
            self.logger.info(f"✅ {direcao} order executed successfully!")
            self.logger.info(f"📊 Entry: {preco:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
            self.logger.info(f"🎫 Ticket: {self.ticket_atual}")
            self.logger.info(f"📈 Risk:Reward Ratio: 1:{self.profit_factor}")

        except Exception as e:
            self.logger.error(f"❌ Error executing order: {e}")

    # Technical Indicators (Same as before, kept for completeness)
    def ema(self, data, period):
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    def macd(self, data, short_period=12, long_period=26, signal_period=9):
        ema_short = self.ema(data, short_period)
        ema_long = self.ema(data, long_period)
        macd_line = ema_short - ema_long
        signal_line = self.ema(macd_line, signal_period)
        return macd_line, signal_line

    def rsi(self, data, period=14):
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.convolve(gain, np.ones(period) / period, mode='valid')
        avg_loss = np.convolve(loss, np.ones(period) / period, mode='valid')

        rs = avg_gain / np.where(avg_loss == 0, 0.000001, avg_loss)
        rsi = 100 - (100 / (1 + rs))

        return np.concatenate([np.full(period - 1, 50), rsi])

    def bollinger_bands(self, data, period=20, num_std=2):
        sma = pd.Series(data).rolling(window=period).mean()
        std = pd.Series(data).rolling(window=period).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper.values, sma.values, lower.values

    def stochastic(self, high, low, close, period=14, k_smooth=3, d_smooth=3):
        # Calculate %K
        low_min = pd.Series(low).rolling(window=period).min()
        high_max = pd.Series(high).rolling(window=period).max()
        k = 100 * ((pd.Series(close) - low_min) / (high_max - low_min))
        k = k.rolling(window=k_smooth).mean()
        
        # Calculate %D
        d = k.rolling(window=d_smooth).mean()
        return k.values, d.values

    def atr(self, high, low, close, period=14):
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.values

# Example usage
if __name__ == "__main__":
    # Initialize MT5 connection
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        mt5.shutdown()
    
    # Create strategy instance
    strategy = EnhancedTradingStrategy(
        ativo="EURUSD",     # Trading symbol
        timeframe="M5",     # 5-minute timeframe
        lote=0.1            # Trading volume
    )
    
    try:
        strategy.executar()
    except KeyboardInterrupt:
        print("🛑 Strategy stopped by user")
    finally:
        mt5.shutdown()
