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

        # Parâmetros otimizados para maior assertividade
        self.rsi_sobrecomprado = 65  # Menos restritivo para vendas
        self.rsi_sobrevendido = 35   # Menos restritivo para compras
        self.bb_desvio = 2.0         # Bandas mais próximas
        self.atr_period = 14         # Período menor para mais sensibilidade
        self.stoch_period = 14
        self.volume_threshold = 1.5   # Volume menos restritivo
        
        # Parâmetros de gestão de risco balanceados
        self.max_daily_loss = 5.0    # Máximo de perda diária em %
        self.min_rr_ratio = 1.5      # Risk/Reward mais agressivo
        self.max_positions = 5        # Mais posições simultâneas
        self.trailing_stop = True     # Ativar trailing stop
        self.breakeven_level = 0.5    # Mais rápido para breakeven

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
                time.sleep(5)
            except Exception as e:
                self.log_system.logar(f"❌ Erro na estratégia: {str(e)}")
                time.sleep(10)

    def parar(self):
        self.operando = False

    def analisar_e_operar(self):
        try:
            # Carregar dados históricos
            if self.operando:
                self.log_system.logar("🔍 Iniciando análise de mercado...")
            
            barras = mt5.copy_rates_from_pos(self.ativo, self.timeframe, 0, 200)
            if barras is None or len(barras) < 100:
                self.log_system.logar(f"❌ Erro: Não foi possível carregar velas de {self.ativo}")
                return

            df = pd.DataFrame(barras)
            if df.isnull().any().any():
                self.log_system.logar("❌ Erro: Dados inválidos ou nulos detectados")
                return

            # Cálculos básicos
            try:
                close = df['close'].values
                high = df['high'].values
                low = df['low'].values
                volume = df['tick_volume'].values

                if len(close) < 50:
                    self.log_system.logar("❌ Erro: Dados insuficientes para análise")
                    return

                # Indicadores principais
                try:
                    ema9 = self.ema(close, 9)
                    ema21 = self.ema(close, 21)
                    ema50 = self.ema(close, 50)
                    
                    macd_line, signal_line = self.macd(close)
                    rsi_valores = self.rsi(close, 14)
                    bb_superior, bb_medio, bb_inferior = self.bollinger_bands(close, 20, self.bb_desvio)
                    stoch_k, stoch_d = self.stochastic(high, low, close, self.stoch_period)
                    atr = self.atr(high, low, close, self.atr_period)
                    momentum = self.momentum(close, 10)

                    # Verificar indicadores
                    if any(map(np.isnan, [ema9[-1], ema21[-1], ema50[-1], macd_line[-1], rsi_valores[-1]])):
                        self.log_system.logar("❌ Erro: Indicadores com valores inválidos")
                        return

                    # Volume analysis
                    volume_ma = float(np.mean(volume[-20:]))
                    volume_atual = float(volume[-1])
                    volume_alto = bool(volume_atual > (volume_ma * self.volume_threshold))

                    # Análise de sinais
                    try:
                        # Tendência
                        tendencia_alta = bool(np.all([
                            float(ema9[-1]) > float(ema21[-1]),
                            float(close[-1]) > float(ema9[-1]),
                            float(ema9[-1]) > float(ema9[-2])
                        ]))

                        tendencia_baixa = bool(np.all([
                            float(ema9[-1]) < float(ema21[-1]),
                            float(close[-1]) < float(ema9[-1]),
                            float(ema9[-1]) < float(ema9[-2])
                        ]))

                        # RSI
                        rsi_compra = bool(np.all([
                            float(rsi_valores[-1]) < self.rsi_sobrevendido,
                            float(rsi_valores[-1]) > float(rsi_valores[-2])
                        ]))

                        rsi_venda = bool(np.all([
                            float(rsi_valores[-1]) > self.rsi_sobrecomprado,
                            float(rsi_valores[-1]) < float(rsi_valores[-2])
                        ]))

                        # MACD
                        macd_compra = bool(np.all([
                            float(macd_line[-1]) > float(signal_line[-1]),
                            float(macd_line[-1]) > float(macd_line[-2])
                        ]))

                        macd_venda = bool(np.all([
                            float(macd_line[-1]) < float(signal_line[-1]),
                            float(macd_line[-1]) < float(macd_line[-2])
                        ]))

                        # Sinais finais
                        sinal_compra = bool(np.all([
                            tendencia_alta,
                            macd_compra or rsi_compra,
                            volume_alto,
                            float(close[-1]) < float(bb_superior[-1]),
                            self.verificar_horario_favoravel(),
                            self.verificar_risco_posicao()
                        ]))

                        sinal_venda = bool(np.all([
                            tendencia_baixa,
                            macd_venda or rsi_venda,
                            volume_alto,
                            float(close[-1]) > float(bb_inferior[-1]),
                            self.verificar_horario_favoravel(),
                            self.verificar_risco_posicao()
                        ]))

                        # Logs de sinais
                        if tendencia_alta and self.operando:
                            self.log_system.logar("📈 Tendência de ALTA detectada - Aguardando confirmação")
                            if macd_compra or rsi_compra:
                                self.log_system.logar("🎯 Confirmação técnica positiva")

                        if tendencia_baixa and self.operando:
                            self.log_system.logar("📉 Tendência de BAIXA detectada - Aguardando confirmação")
                            if macd_venda or rsi_venda:
                                self.log_system.logar("🎯 Confirmação técnica negativa")

                        # Execução
                        if sinal_compra:
                            self.log_system.logar("✅ SINAL DE COMPRA CONFIRMADO")
                            sl_distance = atr[-1] * 1.5
                            tp_distance = atr[-1] * self.min_rr_ratio * 1.5
                            self.abrir_ordem(mt5.ORDER_TYPE_BUY, sl_distance, tp_distance)

                        elif sinal_venda:
                            self.log_system.logar("✅ SINAL DE VENDA CONFIRMADO")
                            sl_distance = atr[-1] * 1.5
                            tp_distance = atr[-1] * self.min_rr_ratio * 1.5
                            self.abrir_ordem(mt5.ORDER_TYPE_SELL, sl_distance, tp_distance)

                    except Exception as e:
                        self.log_system.logar(f"❌ Erro no cálculo de sinais: {str(e)}")
                        return

                except Exception as e:
                    self.log_system.logar(f"❌ Erro no cálculo de indicadores: {str(e)}")
                    return

            except Exception as e:
                self.log_system.logar(f"❌ Erro nos cálculos básicos: {str(e)}")
                return

        except Exception as e:
            self.log_system.logar(f"❌ Erro na análise: {str(e)}")
            return

    def verificar_horario_favoravel(self):
        """Verifica se o horário atual é favorável para operar"""
        hora_atual = pd.Timestamp.now().time()
        # Evita horários de baixa liquidez e alta volatilidade
        if (hora_atual >= pd.Timestamp('09:30').time() and 
            hora_atual <= pd.Timestamp('16:30').time()):
            return True
        return False

    def verificar_risco_posicao(self):
        """Verifica se a posição atende aos critérios de risco"""
        # Verifica número máximo de posições
        posicoes = mt5.positions_total()
        if posicoes >= self.max_positions:
            if self.operando:
                self.log_system.logar("⚠️ Máximo de posições atingido")
            return False
            
        # Verifica drawdown diário
        saldo_inicial = mt5.account_info().balance
        saldo_atual = mt5.account_info().equity
        drawdown = (saldo_inicial - saldo_atual) / saldo_inicial * 100
        
        if drawdown > self.max_daily_loss:
            if self.operando:
                self.log_system.logar(f"⚠️ Máximo drawdown diário atingido: {drawdown:.2f}%")
            return False
            
        return True

    def abrir_ordem(self, tipo_ordem, sl_distance, tp_distance):
        # Verificar se AutoTrading está habilitado
        if not mt5.terminal_info().trade_allowed:
            self.log_system.logar("❌ AutoTrading está desabilitado no MT5!")
            self.log_system.logar("ℹ️ Para habilitar o AutoTrading:")
            self.log_system.logar("1. Clique no botão 'AutoTrading' no topo do MT5 (botão com ícone '⚡')")
            self.log_system.logar("2. Verifique se o AutoTrading está permitido nas configurações do Expert Advisor")
            self.log_system.logar("3. Certifique-se que o botão está verde")
            return

        tick = mt5.symbol_info_tick(self.ativo)
        if tick is None:
            if self.operando:
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
            if self.operando:
                self.log_system.logar(f"❌ Erro ao enviar ordem: {resultado.comment}")
        else:
            self.ticket_atual = resultado.order
            direcao = "COMPRA" if tipo_ordem == mt5.ORDER_TYPE_BUY else "VENDA"
            if self.operando:
                self.log_system.logar(f"✅ ORDEM DE {direcao} CONFIRMADA E EXECUTADA!")
                self.log_system.logar(f"📊 Detalhes da Ordem:")
                self.log_system.logar(f"  • Ticket: {self.ticket_atual}")
                self.log_system.logar(f"  • Preço: {preco:.5f}")
                self.log_system.logar(f"  • Stop Loss: {sl:.5f}")
                self.log_system.logar(f"  • Take Profit: {tp:.5f}")

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
