import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from lib import CTDataset, MNISTneuralNetwork, train_model
import tkinter as tk

# config
CANVAS_SIZE = 280
GRID_SIZE = 28
CELL = CANVAS_SIZE // GRID_SIZE
BRUSH_RADIUS = 1

# load model from saved state
model = MNISTneuralNetwork()
model.load_state_dict(torch.load('data/mnist_model.pth'))
model.eval()

class DrawApp():
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Model")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")
        
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        frame = tk.Frame(root, bg="#1a1a2e")
        frame.pack(padx=20, pady=20)
 
        tk.Label(frame, text="DIGIT RECOGNIZER", font=("Courier", 13, "bold"),
                 fg="#e0e0ff", bg="#1a1a2e").pack(pady=(0, 12))

        self.canvas = tk.Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="crosshair",
                                highlightthickness=2, highlightbackground="#4444aa")
        self.canvas.pack()

        self.pred_var = tk.StringVar(value="draw a digit")
        tk.Label(frame, textvariable=self.pred_var,
                 font=("Courier", 28, "bold"),
                 fg="#00ffcc", bg="#1a1a2e").pack(pady=(14, 6))
        
        self.conf_frame = tk.Frame(frame, bg="#1a1a2e")
        self.conf_frame.pack(fill="x", pady=(0, 10))
        self.conf_bars = []
        for i in range(10):
            row = tk.Frame(self.conf_frame, bg="#1a1a2e")
            row.pack(fill="x", pady=1)
            tk.Label(row, text=str(i), width=2,
                     font=("Courier", 9), fg="#888899", bg="#1a1a2e").pack(side="left")
            bar_bg = tk.Frame(row, bg="#2a2a4a", height=8)
            bar_bg.pack(side="left", fill="x", expand=True, padx=(4, 0))
            bar = tk.Frame(bar_bg, bg="#4444aa", height=8, width=0)
            bar.place(x=0, y=0, relheight=1)
            self.conf_bars.append((bar, bar_bg))

        btn_frame = tk.Frame(frame, bg="#1a1a2e")
        btn_frame.pack(pady=(6, 0))
        tk.Button(btn_frame, text="CLEAR", command=self.clear,
                  font=("Courier", 10, "bold"), bg="#2a2a4a", fg="#e0e0ff",
                  relief="flat", padx=16, pady=6,
                  activebackground="#3a3a6a", activeforeground="white").pack(side="left", padx=6)
        tk.Button(btn_frame, text="PREDICT", command=self.predict,
                  font=("Courier", 10, "bold"), bg="#00ffcc", fg="#1a1a2e",
                  relief="flat", padx=16, pady=6,
                  activebackground="#00ddaa", activeforeground="#1a1a2e").pack(side="left", padx=6)
        
        self.canvas.bind("<B1-Motion>",  self.paint)
        self.canvas.bind("<ButtonPress-1>", self.paint)

    def paint(self, event):
        
        col = event.x // CELL
        row = event.y // CELL

        for dr in range(-BRUSH_RADIUS - 1, BRUSH_RADIUS + 2):
            for dc in range(-BRUSH_RADIUS - 1, BRUSH_RADIUS + 2):
                r, c = row + dr, col + dc
                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    dist = 0.5 * (dr**2 + dc**2)
                    strength = max(0.0, 1.0 - dist / (BRUSH_RADIUS + 1))
                    self.grid[r, c] = min(1.0, self.grid[r, c] + strength)
                    self._draw_cell(r, c)
    
    def _draw_cell(self, row, col):
        val = self.grid[row, col]
        # Map 0-1 to dark blue → white
        r_ch = int(val * 220)
        g_ch = int(val * 220)
        b_ch = int(100 + val * 155)
        color = f"#{r_ch:02x}{g_ch:02x}{b_ch:02x}"
        x0 = col * CELL
        y0 = row * CELL
        self.canvas.create_rectangle(x0, y0, x0+CELL, y0+CELL,
                                     fill=color, outline="")
    
    def predict(self):
        if self.grid.max() == 0: return
        
        tensor = torch.tensor(self.grid).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=0).numpy()
        
        pred = int(np.argmax(probs))
        conf = probs[pred] * 100 
        self.pred_var.set(f"-> {pred} ({conf:.0f}%)")

        bar_width = 180   # max bar width in px
        for i, (bar, bar_bg) in enumerate(self.conf_bars):
            w = int(probs[i] * bar_width)
            color = "#00ffcc" if i == pred else "#4444aa"
            bar.configure(bg=color)
            bar.place(x=0, y=0, relheight=1, width=max(w, 1))

    def clear(self):
        self.grid[:] = 0
        self.canvas.delete("all")
        self.pred_var.set("draw a digit")
        for bar, _ in self.conf_bars:
            bar.place(x=0, y=0, relheight=1, width=1)

root = tk.Tk()
app = DrawApp(root)
root.mainloop()
        
