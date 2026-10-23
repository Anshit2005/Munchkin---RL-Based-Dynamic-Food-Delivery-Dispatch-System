"""Launch the delivery simulator interface."""

import tkinter as tk

from .ui import DeliveryGame


def main() -> None:
    root = tk.Tk()
    root.title("Munchkin - RL Based Dynamic Food Delivery Dispatch System")
    root.minsize(1050, 720)
    DeliveryGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
