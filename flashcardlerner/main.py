import tkinter as tk
import random

# Define FlashCard class
class FlashCard:
    def __init__(self, term, definition):
        self.term = term
        self.definition = definition

# Define FlashCard App class
class FlashCardApp:
    def __init__(self, master):
        self.master = master
        master.title("📚 FlashCard App")
        master.geometry("400x300")
        master.configure(bg="#f0f8ff")

        # Sample data
        self.cards = [
            FlashCard("OOP", "Object-Oriented Programming"),
            FlashCard("HTML", "HyperText Markup Language"),
            FlashCard("RAM", "Random Access Memory"),
            FlashCard("CPU", "Central Processing Unit")
        ]

        random.shuffle(self.cards)
        self.index = 0
        self.showing_term = True

        self.label = tk.Label(master, text="", font=("Arial", 18), wraplength=380, bg="#f0f8ff")
        self.label.pack(pady=50)

        self.flip_button = tk.Button(master, text="🔄 Flip", command=self.flip_card)
        self.flip_button.pack()

        self.next_button = tk.Button(master, text="➡️ Next", command=self.next_card)
        self.next_button.pack(pady=10)

        self.reset_button = tk.Button(master, text="🔁 Reset", command=self.reset)
        self.reset_button.pack()

        self.display_card()

    def display_card(self):
        card = self.cards[self.index]
        self.label.config(text=card.term if self.showing_term else card.definition)

    def flip_card(self):
        self.showing_term = not self.showing_term
        self.display_card()

    def next_card(self):
        self.index = (self.index + 1) % len(self.cards)
        self.showing_term = True
        self.display_card()

    def reset(self):
        random.shuffle(self.cards)
        self.index = 0
        self.showing_term = True
        self.display_card()

# Run the app
root = tk.Tk()
app = FlashCardApp(root)
root.mainloop()
