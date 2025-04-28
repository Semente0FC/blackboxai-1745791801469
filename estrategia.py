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

        # Parâmetros mais dinâmicos para capturar mais oportunidades
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
                time.sleep(5)  # Reduzido para 5 segundos para maior responsividade
            except Exception as e:
                self.log_system.logar(f"Erro na estratégia: {e}")

    def parar(self):
        self.operando = False

    def mostrar_analise(self, close, bb_superior, bb_medio, bb_inferior, rsi_valores, macd_line, signal_line):
        """Mostra a análise técnica através de logs detalhados"""
        try:
            # Análise das Bandas de Bollinger
            distancia_bb = ((close[-1] - bb_medio[-1]) / (bb_superior[-1] - bb_inferior[-1])) * 100
            posicao_bb = "MEIO"
            if distancia_bb > 80:
                posicao_bb = "TOPO"
            elif distancia_bb < 20:
                posicao_bb = "FUNDO"
            
            self.log_system.logar("\n📊 ANÁLISE TÉCNICA DETALHADA:")
            self.log_system.logar("---------------------------")
            self.log_system.logar("🎯 Bandas de Bollinger:")
            self.log_system.logar(f"  • Posição: {posicao_bb}")
            self.log_system.logar(f"  • Superior: {bb_superior[-1]:.5f}")
            self.log_system.logar(f"  • Média: {bb_medio[-1]:.5f}")
            self.log_system.logar(f"  • Inferior: {bb_inferior[-1]:.5f}")
            
            # Análise do RSI
            self.log_system.logar("\n📈 RSI:")
            self.log_system.logar(f"  • Valor Atual: {rsi_valores[-1]:.2f}")
            if rsi_valores[-1] > self.rsi_sobrecomprado:
                self.log_system.logar("  • ALERTA: Região de Sobrecompra!")
            elif rsi_valores[-1] < self.rsi_sobrevendido:
                self.log_system.logar("  • ALERTA: Região de Sobrevenda!")
            
            # Análise do MACD
            macd_status = "NEUTRO"
            if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
                macd_status = "CRUZAMENTO PARA CIMA ⬆️"
            elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
                macd_status = "CRUZAMENTO PARA BAIXO ⬇️"
            
            self.log_system.logar("\n🔄 MACD:")
            self.log_system.logar(f"  • Status: {macd_status}")
            self.log_system.logar(f"  • MACD: {macd_line[-1]:.5f}")
            self.log_system.logar(f"  • Signal: {signal_line[-1]:.5f}")
            
            # Resumo da Análise
            self.log_system.logar("\n📝 RESUMO:")
            if posicao_bb == "TOPO" and rsi_valores[-1] > self.rsi_sobrecomprado:
                self.log_system.logar("⚠️ Possível região de venda - Aguardando confirmação")
            elif posicao_bb == "FUNDO" and rsi_valores[-1] < self.rsi_sobrevendido:
                self.log_system.logar("⚠️ Possível região de compra - Aguardando confirmação")
            
            self.log_system.logar("---------------------------\n")
            
        except Exception as e:
            self.log_system.logar(f"Erro ao mostrar análise: {e}")

    def analisar_e_operar(self):
        # Carregar dados históricos
        if self.operando:
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

        # Análise avançada de tendência e momentum
        if self.operando:
            self.log_system.logar("🔍 Iniciando análise de mercado...")
        
        # Tendência simplificada
        tendencia_alta = (
            (ema9[-1] > ema21[-1]) and     # Tendência de curto prazo
            (close[-1] > ema9[-1]) and      # Preço acima da média curta
            (ema9[-1] > ema9[-2])          # Inclinação positiva
        )
        
        tendencia_baixa = (
            (ema9[-1] < ema21[-1]) and     # Tendência de curto prazo
            (close[-1] < ema9[-1]) and      # Preço abaixo da média curta
            (ema9[-1] < ema9[-2])          # Inclinação negativa
        )

        # Análise de momentum melhorada
        momentum_positivo = (
            momentum[-1] > 0 and
            momentum[-1] > momentum[-2] and
            np.mean(momentum[-5:]) > 0              # Momentum médio positivo
        )
        
        momentum_negativo = (
            momentum[-1] < 0 and
            momentum[-1] < momentum[-2] and
            np.mean(momentum[-5:]) < 0              # Momentum médio negativo
        )

        # Confirmações técnicas refinadas
        rsi_compra = (
            rsi_valores[-1] < self.rsi_sobrevendido and
            rsi_valores[-1] > rsi_valores[-2] and    # RSI subindo
            min(rsi_valores[-3:]) < self.rsi_sobrevendido  # Confirmação da zona
        )
        
        rsi_venda = (
            rsi_valores[-1] > self.rsi_sobrecomprado and
            rsi_valores[-1] < rsi_valores[-2] and    # RSI caindo
            max(rsi_valores[-3:]) > self.rsi_sobrecomprado  # Confirmação da zona
        )

        # Divergências e convergências
        macd_compra = (
            macd_line[-1] > signal_line[-1] and     # MACD acima da signal
            macd_line[-1] > macd_line[-2] and       # MACD subindo
            macd_line[-1] > 0                       # MACD positivo
        )
        
        macd_venda = (
            macd_line[-1] < signal_line[-1] and     # MACD abaixo da signal
            macd_line[-1] < macd_line[-2] and       # MACD caindo
            macd_line[-1] < 0                       # MACD negativo
        )

        # Confirmação por volume e preço
        stoch_compra = (
            stoch_k[-1] < 30 and                    # Mais conservador
            stoch_k[-1] > stoch_d[-1] and           # Cruzamento positivo
            stoch_k[-1] > stoch_k[-2]               # Estocástico subindo
        )
        
        stoch_venda = (
            stoch_k[-1] > 70 and                    # Mais conservador
            stoch_k[-1] < stoch_d[-1] and           # Cruzamento negativo
            stoch_k[-1] < stoch_k[-2]               # Estocástico caindo
        )

        # Análise de Fibonacci e Suporte/Resistência
        fib_retracement = self.calcular_fibonacci(high, low)
        
        # Sinais de entrada mais dinâmicos
        sinal_compra = (
            tendencia_alta and                  # Tendência de curto prazo
            (macd_compra or rsi_compra) and    # Apenas uma confirmação necessária
            (stoch_compra or momentum_positivo) and  # Flexibilidade na confirmação
            volume_alto and                     # Volume ainda importante
            close[-1] < bb_superior and        # Dentro das Bandas
            self.verificar_horario_favoravel() and  # Horário adequado
            self.verificar_risco_posicao()      # Gestão de risco ok
        )

        sinal_venda = (
            tendencia_baixa and                 # Tendência de curto prazo
            (macd_venda or rsi_venda) and      # Apenas uma confirmação necessária
            (stoch_venda or momentum_negativo) and  # Flexibilidade na confirmação
            volume_alto and                     # Volume ainda importante
            close[-1] > bb_inferior and        # Dentro das Bandas
            self.verificar_horario_favoravel() and  # Horário adequado
            self.verificar_risco_posicao()      # Gestão de risco ok
        )

        if self.operando:
            # Mostrar análise detalhada
            self.mostrar_analise(close, bb_superior, bb_medio, bb_inferior, rsi_valores, macd_line, signal_line)
        
        # Gestão de Risco e Execução
        atr_atual = atr[-1]
        
        if sinal_compra:
            # Primeiro aviso de possível compra
            if (tendencia_alta and macd_compra and rsi_compra):
                self.log_system.logar("🔵 Possível sinal de COMPRA detectado - Aguardando confirmação...")
                self.log_system.logar("📊 Condições favoráveis:")
                self.log_system.logar(f"  • Tendência: ALTA")
                self.log_system.logar(f"  • RSI: {rsi_valores[-1]:.2f}")
                self.log_system.logar(f"  • MACD: Positivo")
            
            # Stop Loss e Take Profit otimizados
            sl_distance = atr_atual * 1.5
            tp_distance = atr_atual * self.min_rr_ratio * 1.5
            
            if self.verificar_risco_recompensa(sl_distance, tp_distance):
                self.abrir_ordem(mt5.ORDER_TYPE_BUY, sl_distance, tp_distance)
            else:
                if self.operando:
                    self.log_system.logar("⚠️ Operação cancelada: Risk/Reward inadequado")

        elif sinal_venda:
            # Primeiro aviso de possível venda
            if (tendencia_baixa and macd_venda and rsi_venda):
                self.log_system.logar("🔴 Possível sinal de VENDA detectado - Aguardando confirmação...")
                self.log_system.logar("📊 Condições favoráveis:")
                self.log_system.logar(f"  • Tendência: BAIXA")
                self.log_system.logar(f"  • RSI: {rsi_valores[-1]:.2f}")
                self.log_system.logar(f"  • MACD: Negativo")
            
            # Stop Loss e Take Profit otimizados
            sl_distance = atr_atual * 1.5
            tp_distance = atr_atual * self.min_rr_ratio * 1.5
            
            if self.verificar_risco_recompensa(sl_distance, tp_distance):
                self.abrir_ordem(mt5.ORDER_TYPE_SELL, sl_distance, tp_distance)
            else:
                if self.operando:
                    self.log_system.logar("⚠️ Operação cancelada: Risk/Reward inadequado")
        elif not self.operando:
            return

    def calcular_fibonacci(self, high, low):
        """Calcula níveis de Fibonacci"""
        diff = high[-1] - low[-1]
        return {
            'nivel_236': low[-1] + diff * 0.236,
            'nivel_382': low[-1] + diff * 0.382,
            'nivel_500': low[-1] + diff * 0.500,
            'nivel_618': low[-1] + diff * 0.618,
            'nivel_786': low[-1] + diff * 0.786
        }

    def verificar_suporte(self, preco, fib_levels):
        """Verifica se o preço está próximo a um suporte de Fibonacci"""
        tolerancia = 0.001  # 0.1% de tolerância
        for nivel in fib_levels.values():
            if abs(preco - nivel) / preco < tolerancia:
                return True
        return False

    def verificar_resistencia(self, preco, fib_levels):
        """Verifica se o preço está próximo a uma resistência de Fibonacci"""
        tolerancia = 0.001  # 0.1% de tolerância
        for nivel in fib_levels.values():
            if abs(preco - nivel) / preco < tolerancia:
                return True
        return False

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

    def verificar_risco_recompensa(self, sl_distance, tp_distance):
        """Verifica se o trade atende ao mínimo de risk/reward"""
        rr_ratio = tp_distance / sl_distance
        return rr_ratio >= self.min_rr_ratio

    def abrir_ordem(self, tipo_ordem, sl_distance, tp_distance):
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
