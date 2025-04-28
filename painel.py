import tkinter as tk
from tkinter import ttk, messagebox
import MetaTrader5 as mt5
from utils import obter_saldo
from estrategia import EstrategiaTrading
from log_system import LogSystem
import threading
import time
from datetime import datetime

class PainelApp:  # Changed from EnhancedPainelApp to PainelApp to match imports
    def __init__(self, root):
        self.root = root
        self.root.title("Future MT5 Pro Trading")

        # Modern dark theme colors
        self.colors = {
            'bg_dark': '#0A0A0A',       # Darker background
            'bg_medium': '#1E1E1E',     # Medium background
            'bg_light': '#2D2D2D',      # Light background
            'accent': '#00C853',        # Vibrant green
            'accent_hover': '#00E676',  # Lighter green
            'warning': '#FFB300',       # Warning color
            'danger': '#FF3D00',        # Danger color
            'text': '#FFFFFF',          # White text
            'text_secondary': '#B3B3B3' # Gray text
        }

        self.root.configure(bg=self.colors['bg_dark'])
        self.root.resizable(False, False)
        self.centralizar_janela(1000, 700)

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

        # Combobox style
        style.configure("Custom.TCombobox",
                       fieldbackground=self.colors['bg_light'],
                       background=self.colors['bg_light'],
                       foreground=self.colors['text'],
                       arrowcolor=self.colors['accent'],
                       selectbackground=self.colors['accent'],
                       selectforeground=self.colors['text'])

    def setup_ui(self):
        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'], padx=20, pady=20)
        main_container.pack(fill="both", expand=True)

        # Header with logo and title
        self.setup_header(main_container)
        
        # Trading dashboard
        self.setup_dashboard(main_container)
        
        # Control panel
        self.setup_control_panel(main_container)
        
        # Enhanced log panel
        self.setup_log_panel(main_container)

        # Start data update threads
        self.start_update_threads()

    def setup_header(self, parent):
        header = tk.Frame(parent, bg=self.colors['bg_dark'])
        header.pack(fill="x", pady=(0, 20))

        # Logo and title container
        title_container = tk.Frame(header, bg=self.colors['bg_dark'])
        title_container.pack(side="left")

        logo_label = tk.Label(
            title_container,
            text="📈",
            font=("Helvetica", 32),
            fg=self.colors['accent'],
            bg=self.colors['bg_dark']
        )
        logo_label.pack(side="left", padx=(0, 10))

        title_label = tk.Label(
            title_container,
            text="FUTURE MT5 PRO",
            font=("Helvetica", 24, "bold"),
            fg=self.colors['text'],
            bg=self.colors['bg_dark']
        )
        title_label.pack(side="left")

        # Balance display
        self.saldo_frame = tk.Frame(header, bg=self.colors['bg_light'], padx=15, pady=10)
        self.saldo_frame.pack(side="right")

        tk.Label(
            self.saldo_frame,
            text="SALDO",
            font=("Helvetica", 10, "bold"),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_light']
        ).pack()

        self.saldo_label = tk.Label(
            self.saldo_frame,
            text="R$ 0.00",
            font=("Helvetica", 18, "bold"),
            fg=self.colors['accent'],
            bg=self.colors['bg_light']
        )
        self.saldo_label.pack()

    def setup_dashboard(self, parent):
        dashboard = tk.Frame(parent, bg=self.colors['bg_medium'], padx=20, pady=20)
        dashboard.pack(fill="x", pady=(0, 20))

        # Trading settings
        settings_frame = tk.Frame(dashboard, bg=self.colors['bg_medium'])
        settings_frame.pack(fill="x")

        # Asset selection
        asset_frame = self.create_input_group(settings_frame, "ATIVO")
        self.combo_ativo = ttk.Combobox(
            asset_frame,
            textvariable=self.ativo_selecionado,
            style="Custom.TCombobox",
            width=25
        )
        self.combo_ativo.pack(fill="x")

        # Timeframe selection
        timeframe_frame = self.create_input_group(settings_frame, "TIMEFRAME")
        self.combo_timeframe = ttk.Combobox(
            timeframe_frame,
            textvariable=self.timeframe_selecionado,
            values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
            style="Custom.TCombobox",
            width=25
        )
        self.combo_timeframe.pack(fill="x")
        self.combo_timeframe.current(1)

        # Lot size input
        lot_frame = self.create_input_group(settings_frame, "LOTE")
        self.entry_lote = tk.Entry(
            lot_frame,
            textvariable=self.lote_selecionado,
            font=("Helvetica", 12),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief="flat",
            width=25
        )
        self.entry_lote.pack(fill="x")

        # Organize frames horizontally
        asset_frame.pack(side="left", padx=(0, 10))
        timeframe_frame.pack(side="left", padx=10)
        lot_frame.pack(side="left", padx=(10, 0))

    def create_input_group(self, parent, label):
        frame = tk.Frame(parent, bg=self.colors['bg_medium'])
        
        tk.Label(
            frame,
            text=label,
            font=("Helvetica", 10, "bold"),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_medium']
        ).pack(anchor="w", pady=(0, 5))
        
        return frame

    def setup_control_panel(self, parent):
        control_panel = tk.Frame(parent, bg=self.colors['bg_medium'], padx=20, pady=20)
        control_panel.pack(fill="x", pady=(0, 20))

        # Status indicator
        self.status_label = tk.Label(
            control_panel,
            text="⭘ AGUARDANDO",
            font=("Helvetica", 12, "bold"),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_medium']
        )
        self.status_label.pack(side="left")

        # Control buttons
        buttons_frame = tk.Frame(control_panel, bg=self.colors['bg_medium'])
        buttons_frame.pack(side="right")

        self.btn_atualizar = self.create_button(
            buttons_frame,
            "🔄 Atualizar",
            self.carregar_ativos,
            self.colors['bg_light']
        )
        self.btn_atualizar.pack(side="left", padx=(0, 10))

        self.btn_iniciar = self.create_button(
            buttons_frame,
            "▶ Iniciar Robô",
            self.iniciar_robô,
            self.colors['accent']
        )
        self.btn_iniciar.pack(side="left", padx=(0, 10))

        self.btn_parar = self.create_button(
            buttons_frame,
            "⏹ Parar",
            self.parar_robô,
            self.colors['danger']
        )
        self.btn_parar.pack(side="left")

    def create_button(self, parent, text, command, color):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Helvetica", 11, "bold"),
            fg=self.colors['text'],
            bg=color,
            activebackground=self.colors['accent_hover'],
            activeforeground=self.colors['text'],
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2"
        )

    def setup_log_panel(self, parent):
        log_container = tk.Frame(parent, bg=self.colors['bg_medium'], padx=20, pady=20)
        log_container.pack(fill="both", expand=True)

        # Log header
        header_frame = tk.Frame(log_container, bg=self.colors['bg_medium'])
        header_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            header_frame,
            text="LOGS DO SISTEMA",
            font=("Helvetica", 10, "bold"),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_medium']
        ).pack(side="left")

        # Current time
        self.time_label = tk.Label(
            header_frame,
            text="",
            font=("Helvetica", 10),
            fg=self.colors['text_secondary'],
            bg=self.colors['bg_medium']
        )
        self.time_label.pack(side="right")

        # Log text area
        self.text_log = tk.Text(
            log_container,
            height=15,
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            insertbackground=self.colors['text'],
            relief="flat",
            font=("Consolas", 11),
            padx=15,
            pady=15
        )
        self.text_log.pack(side="left", fill="both", expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(log_container, command=self.text_log.yview)
        scrollbar.pack(side="right", fill="y")
        self.text_log.config(yscrollcommand=scrollbar.set)

        # Connect log system
        self.log_system.conectar_interface(self.text_log)

    def start_update_threads(self):
        # Update balance
        threading.Thread(target=self.atualizar_saldo_loop, daemon=True).start()
        # Update time
        threading.Thread(target=self.atualizar_hora_loop, daemon=True).start()
        # Load initial assets
        self.carregar_ativos()

    def atualizar_hora_loop(self):
        while True:
            current_time = datetime.now().strftime("%H:%M:%S")
            self.time_label.config(text=current_time)
            time.sleep(1)

    def atualizar_saldo_loop(self):
        while True:
            saldo = obter_saldo()
            self.saldo_label.config(text=f"R$ {saldo:.2f}")
            time.sleep(5)

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
        self.status_label.config(text="● OPERANDO", fg=self.colors['accent'])
        self.log_system.logar(
            f"✅ Ambiente OK. Iniciando análise no ativo {ativo}, timeframe {timeframe}, lote {lote_float}. Spread atual: {spread:.1f} pontos.")
        self.estrategia = EstrategiaTrading(ativo, timeframe, lote_float, self.log_system)
        threading.Thread(target=self.estrategia.executar, daemon=True).start()

    def parar_robô(self):
        self.operando = False
        self.status_label.config(text="⭘ AGUARDANDO", fg=self.colors['text_secondary'])
        if hasattr(self, 'estrategia'):
            self.estrategia.parar()
        self.log_system.logar("🛑 Análise parada.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PainelApp(root)
    root.mainloop()
