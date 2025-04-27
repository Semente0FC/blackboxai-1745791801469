import tkinter as tk
from splash_screen import exibir_splash
from login import LoginApp

if __name__ == "__main__":
    exibir_splash()  # Mostrar splash primeiro
    root = tk.Tk()
    app = LoginApp(root)
    root.mainloop()
