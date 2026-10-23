"""Launch the delivery simulator interface."""

import tkinter as tk
from tkinter import simpledialog

from .ui import DeliveryGame


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    player1_name = simpledialog.askstring("Munchkin setup", "Enter Player 1's name:", parent=root)
    player2_name = simpledialog.askstring("Munchkin setup", "Enter Player 2's name:", parent=root)
    root.deiconify()
    root.title("Munchkin - Head-to-Head Dispatch")
    DeliveryGame(root, player1_name or "Player 1", player2_name or "Player 2")
    root.mainloop()


if __name__ == "__main__":
    main()
