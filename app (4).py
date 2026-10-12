"""
SpamGuard - friendly spam call detector (desktop demo)
Run:  python app.py        (needs only Python 3 with Tkinter, nothing to install)

How it works
  1. You type a phone number and press "Simulate incoming call".
  2. The AI model (logistic regression, written in plain Python) gives a spam score 0..1.
  3. score >= 0.8 -> BLOCKED (red alert), 0.5-0.8 -> WARNED (orange alert), else allowed.
  4. Select a call in "Recent calls" and press "This was spam" / "Not spam":
     the model learns from your answer and gets smarter.
"""
import json
import math
import os
import random
import time
import tkinter as tk

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spamguard_data.json")
BAD_PREFIXES = ("+140", "+92", "000")          # edit to match spam you see
BLOCK_AT, WARN_AT = 0.8, 0.5

# colours
BG, GREEN, ORANGE, RED, BLUE = "#F3F6FB", "#2E7D32", "#F57C00", "#D32F2F", "#1976D2"
DARK, GREY, WHITE = "#1F2937", "#6B7280", "#FFFFFF"


def key(number):
    """Compare numbers by their last 10 digits."""
    return "".join(c for c in number if c.isdigit())[-10:]


# --------------------------------------------------------------------------- storage
class Store:
    def __init__(self):
        self.d = {"blocklist": [], "contacts": [], "history": [], "feedback": [], "calls": {}}
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE) as f:
                    self.d.update(json.load(f))
            except (OSError, ValueError):
                pass

    def save(self):
        with open(DATA_FILE, "w") as f:
            json.dump(self.d, f, indent=2)


# --------------------------------------------------------------------------- AI model
FEATURES = ["in_contacts", "bad_prefix", "repeat_calls", "hidden_number", "night_time"]


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-30, min(30, z))))


class Model:
    """Tiny logistic regression trained on synthetic data + your feedback."""

    def __init__(self, feedback):
        self.w = [0.0] * len(FEATURES)
        self.b = 0.0
        self.train(feedback)

    @staticmethod
    def synthetic(n=600):
        rnd = random.Random(1)
        rows = []
        for _ in range(n):
            if rnd.random() < 0.5:   # spam-like call
                x = [0, rnd.random() < 0.6, min(1, rnd.randint(0, 6) / 5), rnd.random() < 0.25, rnd.random() < 0.4]
                rows.append(([float(v) for v in x], 1))
            else:                    # normal call
                x = [rnd.random() < 0.6, rnd.random() < 0.03, 0.2 if rnd.random() < 0.1 else 0,
                     rnd.random() < 0.03, rnd.random() < 0.1]
                rows.append(([float(v) for v in x], 0))
        return rows

    def train(self, feedback, epochs=150, lr=0.1):
        self.w, self.b = [0.0] * len(FEATURES), 0.0
        data = self.synthetic() + [(f["x"], f["y"]) for f in feedback] * 5   # your answers count more
        rnd = random.Random(2)
        for _ in range(epochs):
            rnd.shuffle(data)
            for x, y in data:
                err = self.predict(x) - y
                self.w = [wi - lr * err * xi for wi, xi in zip(self.w, x)]
                self.b -= lr * err

    def predict(self, x):
        return sigmoid(sum(w * v for w, v in zip(self.w, x)) + self.b)


# --------------------------------------------------------------------------- detection
def features(store, number):
    k = key(number)
    now = time.time()
    recent = [t for t in store.d["calls"].get(k or "hidden", []) if now - t < 3600]
    hour = time.localtime().tm_hour
    return [
        float(k in store.d["contacts"]),
        float(number.strip().startswith(BAD_PREFIXES)),
        min(1.0, len(recent) / 5),
        float(number.strip() == ""),
        float(hour >= 22 or hour < 6),
    ]


def check_call(store, model, number):
    """Returns (score, features, action)."""
    k = key(number)
    x = features(store, number)
    if k and k in store.d["blocklist"]:
        score = 1.0
    elif k and k in store.d["contacts"]:
        score = 0.0
    else:
        score = model.predict(x)
    store.d["calls"].setdefault(k or "hidden", []).append(time.time())
    action = "BLOCKED" if score >= BLOCK_AT else "WARNED" if score >= WARN_AT else "allowed"
    return score, x, action


# --------------------------------------------------------------------------- UI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SpamGuard")
        self.geometry("460x800")
        self.configure(bg=BG)
        self.store = Store()
        self.model = Model(self.store.d["feedback"])
        self.build()
        self.refresh()

    # -- helpers
    def label(self, parent, text, size=11, color=DARK, bold=False, **kw):
        return tk.Label(parent, text=text, font=("Helvetica", size, "bold" if bold else "normal"),
                        fg=color, bg=kw.pop("bg", parent["bg"]), **kw)

    def button(self, parent, text, bg, command, fg=WHITE):
        return tk.Button(parent, text=text, bg=bg, fg=fg, activebackground=bg, relief="flat",
                         font=("Helvetica", 11, "bold"), pady=8, cursor="hand2", command=command)

    def card(self, bg=WHITE):
        f = tk.Frame(self, bg=bg, padx=14, pady=12)
        f.pack(fill="x", padx=14, pady=6)
        return f

    # -- layout
    def build(self):
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=18, pady=(16, 0))
        self.label(head, "SpamGuard", 24, bold=True).pack(anchor="w")
        self.label(head, "Your friendly call safety buddy", 11, GREY).pack(anchor="w")

        self.status = self.card(GREEN)                       # status card
        self.label(self.status, "🛡️", 36, WHITE).pack()
        self.label(self.status, "You're protected", 18, WHITE, True).pack()
        self.label(self.status, "I'm watching every incoming call for you.", 10, WHITE).pack()

        stats = tk.Frame(self, bg=BG)
        stats.pack(fill="x", padx=14)
        self.stat_labels = []
        for name, color in (("Checked", BLUE), ("Blocked", RED), ("Warnings", ORANGE)):
            box = tk.Frame(stats, bg=WHITE, padx=10, pady=8)
            box.pack(side="left", expand=True, fill="x", padx=3)
            num = self.label(box, "0", 20, color, True)
            num.pack()
            self.label(box, name, 9, GREY).pack()
            self.stat_labels.append(num)

        c = self.card()                                      # input card
        self.label(c, "Try a call", 13, bold=True).pack(anchor="w")
        self.label(c, "Type a number (empty = hidden number)", 9, GREY).pack(anchor="w")
        self.entry = tk.Entry(c, font=("Helvetica", 13), relief="solid", bd=1)
        self.entry.pack(fill="x", pady=6, ipady=4)
        self.button(c, "📞  Simulate incoming call", BLUE, self.simulate).pack(fill="x", pady=2)
        row = tk.Frame(c, bg=WHITE)
        row.pack(fill="x", pady=2)
        self.button(row, "🚫 Block", RED, self.block).pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.button(row, "👤 Add contact", GREEN, self.add_contact).pack(side="left", expand=True, fill="x", padx=(3, 0))
        self.lists = self.label(c, "", 9, GREY, justify="left", wraplength=400)
        self.lists.pack(anchor="w", pady=(6, 0))

        h = self.card()                                      # history card
        self.label(h, "Recent calls", 13, bold=True).pack(anchor="w")
        self.listbox = tk.Listbox(h, height=7, font=("Helvetica", 11), relief="flat", activestyle="none")
        self.listbox.pack(fill="x", pady=6)
        row = tk.Frame(h, bg=WHITE)
        row.pack(fill="x")
        self.button(row, "🚫 This was spam", RED, lambda: self.feedback(1)).pack(side="left", expand=True, fill="x", padx=(0, 3))
        self.button(row, "✅ Not spam", GREEN, lambda: self.feedback(0)).pack(side="left", expand=True, fill="x", padx=(3, 0))
        self.hint = self.label(h, "Select a call, then tell me. I learn from you!", 9, GREY)
        self.hint.pack(anchor="w", pady=(6, 0))

    # -- actions
    def simulate(self):
        number = self.entry.get().strip()
        score, x, action = check_call(self.store, self.model, number)
        self.store.d["history"].insert(0, {
            "time": time.strftime("%H:%M"), "number": number or "Hidden", "score": round(score, 2),
            "action": action, "x": x})
        self.store.d["history"] = self.store.d["history"][:50]
        self.store.save()
        self.refresh()
        if action != "allowed":
            self.show_alert(number or "Hidden number", action == "BLOCKED")

    def block(self):
        k = key(self.entry.get())
        if k and k not in self.store.d["blocklist"]:
            self.store.d["blocklist"].append(k)
            if k in self.store.d["contacts"]:
                self.store.d["contacts"].remove(k)
            self.store.save()
        self.entry.delete(0, "end")
        self.refresh()

    def add_contact(self):
        k = key(self.entry.get())
        if k and k not in self.store.d["contacts"]:
            self.store.d["contacts"].append(k)
            if k in self.store.d["blocklist"]:
                self.store.d["blocklist"].remove(k)
            self.store.save()
        self.entry.delete(0, "end")
        self.refresh()

    def feedback(self, is_spam):
        sel = self.listbox.curselection()
        if not sel:
            self.hint.config(text="👆 Please select a call first.", fg=ORANGE)
            return
        item = self.store.d["history"][sel[0]]
        self.store.d["feedback"].append({"x": item["x"], "y": is_spam})
        k = key(item["number"])
        if is_spam and k and k not in self.store.d["blocklist"]:
            self.store.d["blocklist"].append(k)          # also block it from now on
        self.model.train(self.store.d["feedback"])       # the AI learns
        self.store.save()
        self.hint.config(text="Thanks! I learned from that. 🧠", fg=GREEN)
        self.refresh()

    # -- display
    def refresh(self):
        hist = self.store.d["history"]
        for lab, val in zip(self.stat_labels, (len(hist), sum(h["action"] == "BLOCKED" for h in hist),
                                                sum(h["action"] == "WARNED" for h in hist))):
            lab.config(text=str(val))
        self.listbox.delete(0, "end")
        for h in hist[:20]:
            icon = {"BLOCKED": "🚫", "WARNED": "⚠️"}.get(h["action"], "✅")
            self.listbox.insert("end", f"{icon}  {h['time']}   {h['number']}   score {h['score']:.2f}")
        bl, ct = self.store.d["blocklist"], self.store.d["contacts"]
        self.lists.config(text=f"Blocked: {', '.join(bl) or 'none'}\nContacts: {', '.join(ct) or 'none'}")

    def show_alert(self, number, blocked):
        color = RED if blocked else ORANGE
        w = tk.Toplevel(self)
        w.title("Call alert")
        w.geometry("460x520")
        w.configure(bg=color)
        w.attributes("-topmost", True)
        box = tk.Frame(w, bg=color)
        box.pack(expand=True)
        self.label(box, "🚫" if blocked else "⚠️", 60, WHITE).pack()
        self.label(box, "SPAM CALL DETECTED" if blocked else "POSSIBLE SPAM CALL", 22, WHITE, True).pack(pady=6)
        self.label(box, "DO NOT ANSWER" if blocked else "Be careful. Share no details.", 16, WHITE, True).pack()
        self.label(box, number, 16, WHITE).pack(pady=10)
        if blocked:
            self.label(box, "We blocked this call for you.", 12, WHITE).pack()
        tk.Button(box, text="OK, got it", font=("Helvetica", 14, "bold"), fg=color, bg=WHITE, relief="flat",
                  padx=30, pady=8, command=w.destroy).pack(pady=24)


if __name__ == "__main__":
    App().mainloop()
