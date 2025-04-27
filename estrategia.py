import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import time


class EstrategiaTrading:
    def __init__(self, ativo, timeframe, lote, log_system):
        self.ativo = ativo
        self.timeframe = self.converter_timeframe(timeframe)
        self.lote = float(lote)
        self.operando = True
        self.log_system = log_system
        self.ticket_atual = None

        # Parâmetros de configuração
        self.rsi_sobrecomprado = 70
        self.rsi_sobrevendido = 30
        self.bb_desvio = 2.0
        self.atr_period = 14
        self.stoch_period = 14
        self.volume_threshold = 1.5

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

    def executar(self):
        while self.operando:
            try:
                self.analisar_e_operar()
                time.sleep(5)  # Reduzido para 5 segundos para maior responsividade
            except Exception as e:
                self.log_system.logar(f"Erro na estratégia: {e}")

    def parar(self):
        self.operando = False

    def analisar_e_operar(self):
        # Carregar dados históricos
        self.log_system.logar("📊 Analisando mercado...")
        barras = mt5.copy_rates_from_pos(self.ativo, self.timeframe, 0, 200)
        if barras is None or len(barras) < 100:
            self.log_system.logar(f"❌ Erro: Não foi possível carregar velas de {self.ativo}")
            return

        df = pd.DataFrame(barras)

        # Cálculos básicos
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values

        # Indicadores principais
        self.log_system.logar("📈 Calculando indicadores...")
        ema9 = self.ema(close, 9)
        ema21 = self.ema(close, 21)
        ema50 = self.ema(close, 50)

        macd_line, signal_line = self.macd(close)
        rsi_valores = self.rsi(close, 14)

        # Indicadores adicionais
        bb_superior, bb_medio, bb_inferior = self.bollinger_bands(close, 20, self.bb_desvio)
        stoch_k, stoch_d = self.stochastic(high, low, close, self.stoch_period)
        atr = self.atr(high, low, close, self.atr_period)

        # Análise de Volume
        volume_ma = np.mean(volume[-20:])
        volume_atual = volume[-1]
        volume_alto = volume_atual > (volume_ma * self.volume_threshold)

        # Análise de Momentum
        momentum = self.momentum(close, 10)

        # Condições de entrada
        self.log_system.logar("🎯 Verificando sinais de entrada...")
        
        tendencia_alta = (ema9[-1] > ema21[-1] > ema50[-1]) and (close[-1] > ema9[-1])
        tendencia_baixa = (ema9[-1] < ema21[-1] < ema50[-1]) and (close[-1] < ema9[-1])

        # Confirmação MACD
        macd_compra = macd_line[-2] < signal_line[-2] and macd_line[-1] > signal_line[-1]
        macd_venda = macd_line[-2] > signal_line[-2] and macd_line[-1] < signal_line[-1]

        # Confirmação RSI
        rsi_compra = rsi_valores[-2] < self.rsi_sobrevendido and rsi_valores[-1] > self.rsi_sobrevendido
        rsi_venda = rsi_valores[-2] > self.rsi_sobrecomprado and rsi_valores[-1] < self.rsi_sobrecomprado

        # Confirmação Estocástico
        stoch_compra = stoch_k[-1] < 20 and stoch_d[-1] < 20 and stoch_k[-1] > stoch_d[-1]
        stoch_venda = stoch_k[-1] > 80 and stoch_d[-1] > 80 and stoch_k[-1] < stoch_d[-1]

        # Confirmação Momentum
        momentum_positivo = momentum[-1] > 0 and momentum[-2] < momentum[-1]
        momentum_negativo = momentum[-1] < 0 and momentum[-2] > momentum[-1]

        # Sinais de entrada combinados
        sinal_compra = (
                tendencia_alta and
                (macd_compra or rsi_compra) and
                stoch_compra and
                momentum_positivo and
                volume_alto and
                close[-1] < bb_superior
        )

        sinal_venda = (
                tendencia_baixa and
                (macd_venda or rsi_venda) and
                stoch_venda and
                momentum_negativo and
                volume_alto and
                close[-1] > bb_inferior
        )

        # Cálculo dinâmico de Stop Loss e Take Profit baseado no ATR
        atr_atual = atr[-1]

        if sinal_compra:
            self.log_system.logar("🔵 Sinal de COMPRA forte detectado!")
            self.log_system.logar(f"📊 RSI: {rsi_valores[-1]:.2f}")
            self.log_system.logar(f"📊 Estocástico K: {stoch_k[-1]:.2f}, D: {stoch_d[-1]:.2f}")
            self.log_system.logar(f"📊 Volume: {volume_atual:.2f} (MA: {volume_ma:.2f})")
            sl_distance = atr_atual * 1.5
            tp_distance = atr_atual * 2.5
            self.abrir_ordem(mt5.ORDER_TYPE_BUY, sl_distance, tp_distance)

        elif sinal_venda:
            self.log_system.logar("🔴 Sinal de VENDA forte detectado!")
            self.log_system.logar(f"📊 RSI: {rsi_valores[-1]:.2f}")
            self.log_system.logar(f"📊 Estocástico K: {stoch_k[-1]:.2f}, D: {stoch_d[-1]:.2f}")
            self.log_system.logar(f"📊 Volume: {volume_atual:.2f} (MA: {volume_ma:.2f})")
            sl_distance = atr_atual * 1.5
            tp_distance = atr_atual * 2.5
            self.abrir_ordem(mt5.ORDER_TYPE_SELL, sl_distance, tp_distance)
        else:
            self.log_system.logar("⏸️ Aguardando melhores condições de mercado...")

    def abrir_ordem(self, tipo_ordem, sl_distance, tp_distance):
        tick = mt5.symbol_info_tick(self.ativo)
        if tick is None:
            self.log_system.logar("❌ Erro ao obter cotação atual")
            return

        preco = tick.ask if tipo_ordem == mt5.ORDER_TYPE_BUY else tick.bid
        point = mt5.symbol_info(self.ativo).point

        # Stop Loss e Take Profit dinâmicos
        sl = preco - sl_distance * point if tipo_ordem == mt5.ORDER_TYPE_BUY else preco + sl_distance * point
        tp = preco + tp_distance * point if tipo_ordem == mt5.ORDER_TYPE_BUY else preco - tp_distance * point

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
            "comment": "Future MT5 Robo v2",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        resultado = mt5.order_send(request)

        if resultado.retcode != mt5.TRADE_RETCODE_DONE:
            self.log_system.logar(f"❌ Erro ao enviar ordem: {resultado.comment}")
        else:
            self.ticket_atual = resultado.order
            direcao = "COMPRA" if tipo_ordem == mt5.ORDER_TYPE_BUY else "VENDA"
            self.log_system.logar(f"✅ Ordem de {direcao} enviada com sucesso!")
            self.log_system.logar(f"🎫 Ticket: {self.ticket_atual}")
            self.log_system.logar(f"💰 Preço: {preco:.5f}")
            self.log_system.logar(f"🛡️ Stop Loss: {sl:.5f}")
            self.log_system.logar(f"🎯 Take Profit: {tp:.5f}")

    # --- Indicadores Técnicos ---
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

    def momentum(self, data, period=10):
        momentum = np.zeros_like(data)
        momentum[period:] = data[period:] - data[:-period]
        momentum[:period] = momentum[period]
        return momentum
