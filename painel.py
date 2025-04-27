import tkinter as tk
from tkinter import ttk, messagebox
import MetaTrader5 as mt5
from utils import obter_saldo
from estrategia import EstrategiaTrading
from log_system import LogSystem
import threading
import time


class PainelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Future MT5 Trading Panel")

        # Modern dark theme colors with Spotify style
        self.colors = {
            'bg_dark': '#121212',  # Spotify background
            'bg_medium': '#181818',  # Spotify card color
            'accent': '#1DB954',  # Spotify green
            'accent_hover': '#1ed760',  # Lighter green
            'success': '#1DB954',  # Spotify green
            'danger': '#e74c3c',  # Keep original red
            'text': '#FFFFFF',  # White text
            'text_secondary': '#B3B3B3'  # Gray text
        }

        self.root.configure(bg=self.colors['bg_dark'])
        self.root.resizable(False, False)
        self.centralizar_janela(900, 650)

        self.ativo_selecionado = tk.StringVar()
        self.timeframe_selecionado = tk.StringVar()
        self.lote_selecionado = tk.StringVar(value="0.10")
        self.operando = False

        self.log_system = LogSystem()

        self.setup_styles()
        self.setup_ui()

    def centralizar_janela(self, largura, altura):
        largura_tela = self.root.winfo_screenwidth()
        altura_tela = self.root.winfo_screenheight()
        x = (largura_tela // 2) - (largura // 2)
        y = (altura_tela // 2) - (altura // 2)
        self.root.geometry(f"{largura}x{altura}+{x}+{y}")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Configure modern looking Combobox
        style.configure("TCombobox",
                        fieldbackground=self.colors['bg_medium'],
                        background=self.colors['bg_medium'],
                        foreground=self.colors['text'],
                        arrowcolor=self.colors['accent'],
                        selectbackground=self.colors['accent'],
                        selectforeground=self.colors['text'])

        # Configure modern looking Entry
        style.configure("TEntry",
                        fieldbackground=self.colors['bg_medium'],
                        foreground=self.colors['text'])

    def create_gradient_frame(self, parent, color1, color2):
        frame = tk.Frame(parent, bg=color1, height=2)
        return frame

    def setup_ui(self):
        # Header Section with gradient
        header_frame = tk.Frame(self.root, bg=self.colors['bg_dark'], pady=20)
        header_frame.pack(fill="x", padx=20)

        gradient = self.create_gradient_frame(header_frame, self.colors['accent'], self.colors['bg_dark'])
        gradient.pack(fill="x", pady=(0, 15))

        self.bem_vindo_label = tk.Label(
            header_frame,
            text="Future MT5 Trading Panel",
            font=("Helvetica", 24, "bold"),
            fg=self.colors['accent'],
            bg=self.colors['bg_dark']
        )
        self.bem_vindo_label.pack()

        self.saldo_label = tk.Label(
            header_frame,
            text="Saldo: ---",
            font=("Helvetica", 16),
            fg=self.colors['success'],
            bg=self.colors['bg_dark']
        )
        self.saldo_label.pack(pady=10)

        # Options Section
        options_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        options_frame.pack(pady=20, padx=30)

        # Create a modern looking container for options
        container = tk.Frame(options_frame, bg=self.colors['bg_medium'], padx=20, pady=20)
        container.pack(fill="x")

        # Trading Options
        self.combo_ativo = self.create_combobox(container, self.ativo_selecionado)
        self.add_labeled_widget(container, "Ativo Trading:", 0, self.combo_ativo)

        self.combo_timeframe = self.create_combobox(
            container,
            self.timeframe_selecionado,
            ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
        )
        self.add_labeled_widget(container, "Timeframe:", 1, self.combo_timeframe)

        self.entry_lote = self.create_entry(container, self.lote_selecionado)
        self.add_labeled_widget(container, "Tamanho do Lote:", 2, self.entry_lote)

        # Modern update button
        self.btn_atualizar_ativos = tk.Button(
            container,
            text="🔄 Atualizar Ativos",
            command=self.carregar_ativos,
            bg=self.colors['accent'],
            fg=self.colors['text'],
            font=("Helvetica", 10, "bold"),
            activebackground=self.colors['accent_hover'],
            activeforeground=self.colors['text'],
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2"
        )
        self.btn_atualizar_ativos.grid(row=0, column=2, padx=20, pady=5, sticky="w")

        # Control Buttons Section
        button_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        button_frame.pack(pady=20)

        self.btn_iniciar = tk.Button(
            button_frame,
            text="▶ Iniciar Análise com o Robô",
            command=self.iniciar_robô,
            bg=self.colors['accent'],
            fg=self.colors['text'],
            font=("Helvetica", 14, "bold"),
            activebackground=self.colors['accent_hover'],
            activeforeground=self.colors['text'],
            relief="flat",
            width=25,
            state="disabled",
            cursor="hand2",
            pady=12
        )
        self.btn_iniciar.grid(row=0, column=0, padx=10)

        self.btn_parar = tk.Button(
            button_frame,
            text="⏹ Parar Análise",
            command=self.parar_robô,
            bg=self.colors['danger'],
            fg=self.colors['text'],
            font=("Helvetica", 14, "bold"),
            activebackground=self.colors['danger'],
            activeforeground=self.colors['text'],
            relief="flat",
            width=20,
            cursor="hand2",
            pady=12
        )
        self.btn_parar.grid(row=0, column=1, padx=10)

        # Logs Section
        logs_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        logs_frame.pack(pady=20, fill="both", expand=True, padx=30)

        tk.Label(
            logs_frame,
            text="📋 Logs do Sistema",
            font=("Helvetica", 14, "bold"),
            fg=self.colors['text'],
            bg=self.colors['bg_dark']
        ).pack(anchor="w")

        text_frame = tk.Frame(logs_frame, bg=self.colors['bg_medium'])
        text_frame.pack(fill="both", expand=True)

        self.text_log = tk.Text(
            text_frame,
            height=15,
            width=90,
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief="flat",
            font=("Consolas", 11),
            padx=15,
            pady=15
        )
        self.text_log.pack(side="left", fill="both", expand=True)

        # Modern scrollbar
        self.scrollbar = tk.Scrollbar(
            text_frame,
            command=self.text_log.yview,
            width=10,
            relief="flat",
            bg=self.colors['bg_medium']
        )
        self.scrollbar.pack(side="right", fill="y", padx=2)

        self.text_log.config(yscrollcommand=self.scrollbar.set)
        self.log_system.conectar_interface(self.text_log)

        # Footer with gradient
        footer_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        footer_frame.pack(fill="x", pady=(0, 10))

        gradient = self.create_gradient_frame(footer_frame, self.colors['bg_dark'], self.colors['accent'])
        gradient.pack(fill="x")

        self.carregar_ativos()

        # Setup variable traces
        self.ativo_selecionado.trace_add("write", self.verificar_campos)
        self.timeframe_selecionado.trace_add("write", self.verificar_campos)
        self.lote_selecionado.trace_add("write", self.verificar_campos)

        # Start balance update thread
        threading.Thread(target=self.atualizar_saldo_loop, daemon=True).start()

    def add_labeled_widget(self, parent, label_text, row, widget):
        tk.Label(
            parent,
            text=label_text,
            font=("Helvetica", 12),
            fg=self.colors['text'],
            bg=self.colors['bg_medium']
        ).grid(row=row, column=0, padx=10, pady=10, sticky="e")
        widget.grid(row=row, column=1, pady=10, sticky="w")

    def create_combobox(self, parent, variable, values=None):
        combo = ttk.Combobox(
            parent,
            textvariable=variable,
            width=30,
            values=values or [],
            style="TCombobox"
        )
        if values:
            combo.current(1)
        return combo

    def create_entry(self, parent, variable):
        return tk.Entry(
            parent,
            textvariable=variable,
            font=("Helvetica", 12),
            bg=self.colors['bg_medium'],
            fg=self.colors['text'],
            relief="flat",
            width=25,
            insertbackground=self.colors['text']  # Cursor color
        )

    def carregar_ativos(self):
        try:
            symbols = mt5.symbols_get()
            lista_ativos = [symbol.name for symbol in symbols if symbol.visible]
            self.combo_ativo['values'] = lista_ativos
            if lista_ativos:
                self.combo_ativo.current(0)
            self.log_system.logar("✅ Ativos atualizados com sucesso!")
        except Exception as e:
            self.log_system.logar(f"❌ Erro ao carregar ativos: {e}")

    def atualizar_saldo_loop(self):
        while True:
            saldo = obter_saldo()
            self.saldo_label.config(text=f"Saldo: R$ {saldo:.2f}")
            time.sleep(5)

    def verificar_campos(self, *args):
        ativo = self.ativo_selecionado.get().strip()
        timeframe = self.timeframe_selecionado.get().strip()
        lote = self.lote_selecionado.get().strip()
        if ativo and timeframe and lote:
            self.btn_iniciar.config(state="normal")
        else:
            self.btn_iniciar.config(state="disabled")

    def iniciar_robô(self):
        ativo = self.ativo_selecionado.get().strip()
        timeframe = self.timeframe_selecionado.get().strip()
        lote = self.lote_selecionado.get().strip()

        if not ativo:
            self.log_system.logar("⚠️ Selecione um ativo para operar!")
            return
        if not timeframe:
            self.log_system.logar("⚠️ Selecione um timeframe para operar!")
            return
        if not lote:
            self.lote_selecionado.set("0.10")
            lote = "0.10"
            self.log_system.logar("⚠️ Lote vazio. Valor padrão 0.10 atribuído.")

        try:
            lote_float = round(float(lote), 2)
            if lote_float <= 0:
                self.log_system.logar("⚠️ O lote deve ser maior que zero.")
                return
        except ValueError:
            messagebox.showerror("Erro de Lote", "Valor de lote inválido! Informe um número válido como 0.10")
            self.log_system.logar("❌ Erro: Lote inválido informado.")
            return

        info = mt5.symbol_info(ativo)
        if info is None:
            self.log_system.logar(f"❌ Ativo {ativo} não encontrado no MetaTrader 5.")
            return
        if not info.visible:
            self.log_system.logar(f"⚠️ Ativo {ativo} não está visível no MT5. Abra o ativo no terminal!")
            return
        if info.trade_mode != mt5.SYMBOL_TRADE_MODE_FULL:
            self.log_system.logar(f"❌ Ativo {ativo} não está liberado para operar (modo inválido)!")
            return

        tick = mt5.symbol_info_tick(ativo)
        if tick is None:
            self.log_system.logar(f"❌ Não foi possível obter preços do ativo {ativo}.")
            return

        spread = (tick.ask - tick.bid) / info.point
        spread_maximo_aceito = 50

        if spread > spread_maximo_aceito:
            self.log_system.logar(
                f"⚠️ Spread do ativo {ativo} está muito alto ({spread:.1f} pontos). Análise bloqueada.")
            return

        if tick.bid == 0 or tick.ask == 0:
            self.log_system.logar(f"⚠️ Mercado para o ativo {ativo} está FECHADO. Análise bloqueada.")
            return
        else:
            self.log_system.logar(f"✅ Mercado para o ativo {ativo} está ABERTO.")

        self.operando = True
        self.log_system.logar(
            f"✅ Ambiente OK. Iniciando análise no ativo {ativo}, timeframe {timeframe}, lote {lote_float}. Spread atual: {spread:.1f} pontos.")
        self.estrategia = EstrategiaTrading(ativo, timeframe, lote_float, self.log_system)
        threading.Thread(target=self.estrategia.executar, daemon=True).start()

    def parar_robô(self):
        self.operando = False
        if hasattr(self, 'estrategia'):
            self.estrategia.parar()
        self.log_system.logar("🛑 Análise parada.")