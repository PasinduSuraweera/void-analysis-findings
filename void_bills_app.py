"""
Void Bills Categorization Tool
A GUI application for classifying Pizza Hut void order reasons using AI.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import json
import time
import re
from datetime import datetime
from groq import Groq

# ============= CONSTANTS =============
BATCH_SIZE = 20
MODEL_NAME = "openai/gpt-oss-120b"
APP_VERSION = "1.0.0"

CATEGORIES = [
    "Call Center mistake",
    "Cashier mistake",
    "cus. Change the order",
    "cus.related issue",
    "Customer Cancel order",
    "Customer denied the order",
    "double punch",
    "grid issue",
    "location",
    "order cancelled by aggregator",
    "Order delay",
    "order type change",
    "other",
    "out of stock",
    "payment issue",
    "phone",
    "product issue or complain",
    "promotion",
    "rider issue",
    "system issue",
    "testing",
    "voids without clear reason/ remark"
]

# ============= RULE-BASED CLASSIFICATION =============
KEYWORD_RULES = {
    "testing": [
        r"\btest\s*(order|odar|oder)?\b", r"\btesting\b", r"\btes\s*oder\b",
        r"\bproduct\s*testing\b", r"\bfrom\s*(it|preshan)\b",
        r"\bit\s*team\s*check\b"
    ],
    "promotion": [
        r"\blsm\b", r"\bpromo(tion)?\b", r"\boffer\b", r"\b50\s*%\s*(off|flash|discount)?\b",
        r"\bdiscount\b", r"\bcyber\s*saving\b", r"\bmeal\s*deal\b", r"\bflash\s*offer\b",
        r"\bdon'?t\s*cook\b", r"\bhsbc\b", r"\bges\s*\d+%\b", r"\b\d+%\s*off\b",
        r"\b15\s*%\b", r"\b20\s*%\b", r"\b30\s*%\b", r"\b1000\s*off\b",
        r"\bhave\s*a?\s*\d+%\s*discount\b"
    ],
    "payment issue": [
        r"\bcredit\s*card\b", r"\bcard\s*(not\s*work|isn'?t\s*work|failed)\b",
        r"\bhnb\s*card\b", r"\bbank\s*card\b", r"\bvisa\b", r"\bmachine\b",
        r"\bpayment\s*(method|issue)?\b", r"\bpetty\s*cash\b", r"\bonline\s*payment\b",
        r"\bpaid\s*order\b", r"\bdon'?t\s*have\s*(enough\s*)?(money|cash)\b"
    ],
    "Cashier mistake": [
        r"\bcashier\s*(mistake|mistakenly|mistakly|wrong)\b", r"\bwrongly\s*punch(ed)?\b",
        r"\bmistakenly\s*(punch|close|add|collect|mark|order|dispatch)\b", r"\bmistakly\b",
        r"\bcashier\s*error\b", r"\bwrong\s*(punch|order|bill|close)\b",
        r"\bcashiar\b", r"\bcashiyar\b", r"\bwrong\s*by\s*cashier\b",
        r"\bwrong\s*order[sw]?\b", r"\bwrong\s*ordewr\b", r"\bmiss\s*communication\b",
        r"\bdispatcher\s*mistakenly\b", r"\bdispatcher\s*collected\b",
        r"\bmistakenly\s*orders?\s*split\b", r"\bdidn'?t\s*close\b", r"\bdidnt\s*close\b",
        r"\bdispac?ter\s*mistakenly\b"
    ],
    "Call Center mistake": [
        r"\bcsr\s*(error|mistake)?\b", r"\bsale\s*cent(er|re)\s*(error|mistake|issue|request)?\b",
        r"\bcall\s*cent(er|re)\s*(error|mistake|asked|have\s*wrongly)?\b", r"\bsales\s*cent(er|re)\b",
        r"\baccording\s*to\s*call\s*cent\b", r"\bacording\s*to\s*call\s*senter\b",
        r"\binformed\s*by\s*outlet\b", r"\binfomed\s*by\s*outlet\b",
        r"\bsale\s*center\s*mistac?ly\b", r"\bcall\s*center\s*have\s*wrongly\b"
    ],
    "Customer denied the order": [
        r"\bcustomer\s*denied\b", r"\bcux\s*denied\b", r"\bdenied\s*(the\s*)?order\b",
        r"\bcustermar\s*denied\b", r"\brefuse[d]?\s*(the\s*)?order\b",
        r"\breject(ed)?\s*(the\s*)?order\b", r"\bdeniend\b", r"\bdenaid\b",
        r"\bdidn'?t\s*place\s*(any\s*)?order\b", r"\bdidnt\s*place\s*(any\s*)?order\b",
        r"\bhe\s*didnt\s*place\b"
    ],
    "Customer Cancel order": [
        r"\bcustomer\s*(want\s*(to\s*)?)?cancel\b", r"\bcux\s*cancel\b",
        r"\bcx\s*(want\s*(to\s*)?)?cancel\b", r"\bcu\s*wont\s*to\s*cancel\b",
        r"\bcustomer\s*cansel\b", r"\bcustomer\s*cancell\b", r"\bcustomr\s*cancal\b",
        r"\bplease\s*cancel\b", r"\bcncl\b", r"\bcustomer\s*cancelled\b"
    ],
    "double punch": [
        r"\bordered\s*twice\b", r"\bsame\s*order\s*\d+\b", r"\b2\s*times?\s*(same\s*)?order\b",
        r"\btwo\s*orders?\s*(were\s*)?(placed|same)\b", r"\bdouble\b", r"\bdubble\b",
        r"\btwise\s*the\s*order\b", r"\bpast\s*same\s*order\b"
    ],
    "grid issue": [
        r"\bgrid\s*(issue)?\b", r"\bout\s*of\s*grid\b", r"\bgride\s*issue\b"
    ],
    "location": [
        r"\bwrong\s*address\b", r"\bwrong\s*location\b", r"\bdifferent\s*(location|outlet|city)\b",
        r"\bwant\s*to\s*deliver\s*\w+\s*outlet\b", r"\bgo(ing)?\s*(to|from)\s*\w+\b",
        r"\btransfer(red)?\s*to\b", r"\bdeliver\s*from\b",
        r"\bslave\s*island\b", r"\bnearest\s*location\b", r"\bsent\s*\d+\b",
        r"\bfrom\s+\w+\s*outlet\b", r"\bto\s+\w+\s*outlet\b", r"\boutlet\s*order\b",
        r"\bdelivery\s+from\s+\w+\b", r"\bwennappuwa\b", r"\bkoswattha\b",
        r"\bnew\s*dkt\s*\w*\s*\d+\b", r"\bneew\s*order\b", r"\bkochchikade\b",
        r"\bpanadura\b", r"\bpandura\b", r"\bhavelock\b", r"\b\d{2,3}\s*-\s*\w+\b",
        r"\bdifferent\s*city\s*with\s*different\s*out\s*let\b", r"\bsimilar\s*address\b"
    ],
    "phone": [
        r"\bphone\s*(number\s*)?(not\s*)?(work|answer|respond)\b",
        r"\bnot\s*(answer|respond)(ing|ed)?\s*(the\s*)?(call|phone|mobile)?\b",
        r"\bwrong\s*(phone\s*)?(number|no|mobile)\b", r"\bincorrect\s*number\b",
        r"\bcan'?t\s*contact\b", r"\bmobile\s*not\s*work\b", r"\bno\s*answer(ing)?\b",
        r"\bnumber\s*wrong\b", r"\bnumber\s*not\s*work\b",
        r"\bnot\s*respons\b", r"\bphone\s*call\s*(is\s*)?not\s*reac\b",
        r"\bcx\s*no\s*answering\b", r"\bnumber\s*is\s*not\s*working\b",
        r"\bdid\s*not\s*answer\s*(the\s*)?phone\b", r"\bdidn'?t\s*answer\b",
        r"\bnot\s*in\s*responded?\s*call\b", r"\bphone\s*not\s*responded\b",
        r"\bvoice\s*mail\b", r"\bgiven\s*number\s*(is\s*)?not\s*working\b",
        r"\bcalled\s*(the\s*)?customer\s*\d+\s*times\b"
    ],
    "Order delay": [
        r"\border\s*delay(ed)?\b", r"\bdelay\s*(issue|order)?\b", r"\blate\s*issue\b",
        r"\border\s*deley\b", r"\bpromise\s*time\b", r"\bcan'?t\s*wait\b",
        r"\bhea[vr]y\s*rain\b"
    ],
    "order type change": [
        r"\bchange\s*(to\s*)?(delivery|take\s*away|dine|pickup|t/?w)\b",
        r"\bwant\s*(to\s*)?(deliver|delivery)\b", r"\bwant\s*dine\b",
        r"\bpick\s*up\s*(for|to)\s*delivery\b", r"\btake\s*away\s*can[sc]al\b",
        r"\border\s*type\s*change\b"
    ],
    "cus. Change the order": [
        r"\bchange\s*(the\s*)?time\b", r"\btime\s*(order|change)\b",
        r"\bwant\s*(the\s*)?order\s*@\b", r"\bwanted\s*to\s*change\s*(the\s*)?order\b",
        r"\bcustomer\s*change\b", r"\bcx\s*want(s|ed)?\s*to\s*change\b",
        r"\bcux\s*want(s|ed)?\s*to\s*change\b", r"\bchange\s*(the\s*)?order\b",
        r"\bcux\s*want(s|ed)?\s*(large|medium|small|personal)\b",
        r"\bcustomer\s*want(s|ed)?\s*(large|medium|small|personal)\b",
        r"\border\s*replaced\b", r"\breplace\s*to\b", r"\breplaced\s*delivery\b",
        r"\bthis\s*order\s*was\s*placed\s*yesterday\b",
        r"\bcustomer\s*mistakenly\s*placed\b", r"\bchanged\s*to\s*no\.?\b"
    ],
    "out of stock": [
        r"\bout\s*of\s*stock\b", r"\boos\b", r"\bstock\s*out\b",
        r"\b(item|product|pizza|coke|coca|pepsi|drink|topping|ingredient)s?\s*(is\s*)?(not\s*)?(available|have)\b",
        r"\bnot\s*available\s*(at\s*)?\s*(main\s*)?supplier\b",
        r"\bsome\s*items\s*are\s*not\s*available\b"
    ],
    "rider issue": [
        r"\brider\s*(mistake|mistakenly|issue)\b", r"\brider'?s?\s*issue\b",
        r"\briderr?s?issue\b",
        r"\briders?\s*(not\s*)?(assigned|assinged|assined)\b",
        r"\bno\s*rider\s*(arrived|assigned|assinged)\b",
        r"\bnorider\s*arrived\b", r"\brider\s*not\s*(assigned|assinged|arrived)\b",
        r"\brider\s*arrived\s*yet\b", r"\briderarrived\s*yet\b"
    ],
    "system issue": [
        r"\bsystem\s*(error|issue)\b", r"\bsystem\s*show\b",
        r"\brider\s*app\b", r"\bcan\s*not\s*delivered?\s*in\s*rider\s*app\b",
        r"\bit\s*team\s*(is\s*)?busy\b"
    ],
    "order cancelled by aggregator": [
        r"\buber\b", r"\bpick\s*me\b", r"\bpickme\b", r"\baggregator\b",
        r"\bcancelled?\s*by\s*(uber|pick\s*me)\b", r"\border\s*cancel(led)?\s*by\b"
    ],
    "product issue or complain": [
        r"\bproduct\s*issue\b", r"\bcomplain\b", r"\bdissatisfy\b", r"\bwrong\s*pizza\b"
    ],
    "cus.related issue": [
        r"\bcustomer\s*(is\s*)?(not\s*)?(available|availble)\b", 
        r"\bcustomer\s*did(n'?t)?\s*come\b",
        r"\bcustomer\s*left\b", r"\bcustomer\s*visit\b",
        r"\boutlet\s*closed\b", r"\bpower\s*cut\b",
        r"\boven\s*breakdown\b", r"\bsecurity\s*department\b",
        r"\bcux?\s*(is\s*)?(not\s*)?(available|availble)\b",
        r"\bcustomer\s*not\s*available\b", r"\bcx\s*not\s*available\b",
        r"\bcalled\s*(the\s*)?customer\s*several\s*times\b",
        r"\bcustomer\s*(is\s*)?not\s*showed?\s*up\b",
        r"\bcustomer\s*(was\s*)?(not\s*)?(at\s*)?(the\s*)?location\b",
        r"\bcustomer\s*wasn'?t\s*available\s*at\s*(the\s*)?location\b",
        r"\bnot\s*at\s*home\b"
    ]
}

PRIORITY_ORDER = [
    "testing", "Customer denied the order", "double punch",
    "order cancelled by aggregator", "Cashier mistake", "Call Center mistake",
    "payment issue", "promotion", "grid issue", "rider issue", "phone",
    "cus.related issue", "out of stock", "Order delay", "system issue",
    "order type change", "location", "Customer Cancel order",
    "cus. Change the order", "product issue or complain"
]


class VoidBillsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Void Bills Categorization Tool")
        self.root.geometry("700x750")
        self.root.resizable(True, True)
        self.root.minsize(600, 650)
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar(value="categorized_orders_clean.xlsx")
        self.api_key = tk.StringVar()
        self.is_running = False
        self.client = None
        
        # Options
        self.ai_verify_rules = tk.BooleanVar(value=False)
        self.export_summary = tk.BooleanVar(value=True)
        
        # Store classification results for summary
        self.last_summary = None
        self.last_stats = {}
        
        # Load saved settings
        self.load_settings()
        
        # Create UI
        self.create_widgets()
    
    def create_widgets(self):
        # Main container with padding
        self.main_frame = ttk.Frame(self.root, padding="12")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        self.title_label = ttk.Label(self.main_frame, text="Void Bills Categorization Tool", 
                                     font=("Segoe UI", 12, "bold"))
        self.title_label.pack(pady=(0, 10))
        
        # === API Key Section ===
        self.api_frame = ttk.LabelFrame(self.main_frame, text="API Configuration", padding="10")
        self.api_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(self.api_frame, text="Groq API Key:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        
        api_entry_frame = ttk.Frame(self.api_frame)
        api_entry_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.api_entry = ttk.Entry(api_entry_frame, textvariable=self.api_key, show="*", width=50)
        self.api_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.show_key_btn = ttk.Button(api_entry_frame, text="Show", width=6, command=self.toggle_api_visibility)
        self.show_key_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Button(api_entry_frame, text="Save Key", command=self.save_api_key).pack(side=tk.LEFT, padx=(5, 0))
        
        self.api_hint_label = ttk.Label(self.api_frame, text="Get your API key from: https://console.groq.com", 
                                         foreground="gray")
        self.api_hint_label.pack(anchor=tk.W, pady=(5, 0))
        
        # === File Selection Section ===
        self.file_frame = ttk.LabelFrame(self.main_frame, text="File Selection", padding="10")
        self.file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input file instructions
        note_text = "Note: Clean up the void bill listing first (remove extra rows/columns) and name it PH_VoidBillListing.xlsx"
        self.input_note = ttk.Label(self.file_frame, text=note_text, foreground="#666666", wraplength=550, font=("Segoe UI", 8))
        self.input_note.pack(anchor=tk.W, pady=(0, 8))
        
        # Input file
        ttk.Label(self.file_frame, text="Input Excel File:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        input_frame = ttk.Frame(self.file_frame)
        input_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Entry(input_frame, textvariable=self.input_file, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_frame, text="Browse...", command=self.browse_input).pack(side=tk.LEFT, padx=(5, 0))
        
        # Output file
        ttk.Label(self.file_frame, text="Output Excel File:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        output_frame = ttk.Frame(self.file_frame)
        output_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Entry(output_frame, textvariable=self.output_file, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).pack(side=tk.LEFT, padx=(5, 0))
        
        # === Options Section ===
        self.options_frame = ttk.LabelFrame(self.main_frame, text="Options", padding="10")
        self.options_frame.pack(fill=tk.X, pady=(0, 10))
        
        options_row1 = ttk.Frame(self.options_frame)
        options_row1.pack(fill=tk.X, pady=(0, 5))
        
        # AI Verify Rules checkbox
        self.ai_verify_cb = ttk.Checkbutton(
            options_row1, 
            text="AI verify rule-based classifications",
            variable=self.ai_verify_rules,
            command=self.save_settings
        )
        self.ai_verify_cb.pack(side=tk.LEFT, padx=(0, 20))
        
        options_row2 = ttk.Frame(self.options_frame)
        options_row2.pack(fill=tk.X, pady=(0, 5))
        
        # Export Summary checkbox
        self.export_cb = ttk.Checkbutton(
            options_row2, 
            text="Export summary report",
            variable=self.export_summary,
            command=self.save_settings
        )
        self.export_cb.pack(side=tk.LEFT, padx=(0, 20))
        
        # Run button
        self.run_frame = ttk.Frame(self.main_frame)
        self.run_frame.pack(fill=tk.X, pady=(5, 8))
        
        self.run_btn = ttk.Button(
            self.run_frame, 
            text="Run Classification",
            command=self.run_classification
        )
        self.run_btn.pack(side=tk.LEFT)
        
        # === Progress Section ===
        self.progress_frame = ttk.LabelFrame(self.main_frame, text="Progress", padding="10")
        self.progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.status_label = ttk.Label(self.progress_frame, text="Ready", foreground="gray")
        self.status_label.pack(anchor=tk.W)
        
        # === Log Section ===
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=12, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # === Bottom Buttons ===
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill=tk.X)
        
        ttk.Button(self.bottom_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.bottom_frame, text="View Summary", command=self.show_summary).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(self.bottom_frame, text="Open Folder", command=self.open_output_folder).pack(side=tk.RIGHT)
        
        # Version label
        version_label = ttk.Label(self.bottom_frame, text=f"v{APP_VERSION}", foreground="gray")
        version_label.pack(side=tk.RIGHT, padx=(0, 10))
    
    def toggle_api_visibility(self):
        if self.api_entry.cget("show") == "*":
            self.api_entry.config(show="")
            self.show_key_btn.config(text="Hide")
        else:
            self.api_entry.config(show="*")
            self.show_key_btn.config(text="Show")
    
    def load_settings(self):
        """Load settings from .env file"""
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("API_KEY="):
                        self.api_key.set(line.split("=", 1)[1].strip())
                    elif line.startswith("AI_VERIFY_RULES="):
                        self.ai_verify_rules.set(line.split("=", 1)[1].strip().lower() == "true")
                    elif line.startswith("EXPORT_SUMMARY="):
                        self.export_summary.set(line.split("=", 1)[1].strip().lower() == "true")
        
        # Try environment variable for API key
        env_key = os.getenv("API_KEY")
        if env_key and not self.api_key.get():
            self.api_key.set(env_key)
    
    def save_settings(self):
        """Save all settings to .env file"""
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        with open(env_path, "w") as f:
            f.write(f"API_KEY={self.api_key.get()}\n")
            f.write(f"AI_VERIFY_RULES={str(self.ai_verify_rules.get()).lower()}\n")
            f.write(f"EXPORT_SUMMARY={str(self.export_summary.get()).lower()}\n")
    
    def save_api_key(self):
        """Save API key and show confirmation"""
        self.save_settings()
        self.log("Settings saved to .env file")
        messagebox.showinfo("Saved", "API key and settings saved successfully!")
    
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            # Keep output as categorized_orders_clean.xlsx for Colab analysis
    
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
    
    def log(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
    
    def update_status(self, text, progress=None):
        self.status_label.config(text=text)
        if progress is not None:
            self.progress_var.set(progress)
        self.root.update_idletasks()
    
    def open_output_folder(self):
        output = self.output_file.get()
        if output:
            folder = os.path.dirname(os.path.abspath(output))
            if os.path.exists(folder):
                os.startfile(folder)
            else:
                messagebox.showwarning("Warning", "Output folder does not exist yet.")
    
    def show_summary(self):
        """Show the last classification summary in a popup"""
        if not self.last_summary:
            messagebox.showinfo("No Summary", "No classification has been run yet.\nRun a classification first to see the summary.")
            return
        
        # Create summary popup
        popup = tk.Toplevel(self.root)
        popup.title("Classification Summary")
        popup.geometry("500x400")
        popup.resizable(True, True)
        
        # Summary text
        text = scrolledtext.ScrolledText(popup, wrap=tk.WORD, font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text.insert(tk.END, self.last_summary)
        text.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=10)
    
    def apply_keyword_rules(self, text):
        """Apply rule-based classification"""
        if not text or pd.isna(text):
            return None
        
        text_lower = str(text).lower()
        
        for category in PRIORITY_ORDER:
            if category in KEYWORD_RULES:
                for pattern in KEYWORD_RULES[category]:
                    if re.search(pattern, text_lower):
                        return category
        return None
    
    def extract_new_bill_id(self, text):
        """Extract new bill numbers from text"""
        if not text or pd.isna(text):
            return None

        clean_text = str(text).upper()
        
        patterns = [
            r'(?:NEW\s*BILL?\s*(?:NO|NUMBER|NOMBER|NUBBER)?[:\s-]*|NBN[:\s-]*|N\.?B\.?N[:\s-]*)([A-Z]{1,2}[\s-]?\d{4,7})',
            r'(?:NEW\s*(?:ORDER|DOCKET|DKT|DOC|TRANX)\s*(?:NO|NUMBER)?[:\s-]*)([A-Z]{0,2}[\s-]?\d{3,7})',
            r'(?:ORDER\s*(?:NO|NUMBER)?[:\s-]*)(\d{2,3})(?:\s|$|,)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_text)
            if match:
                result = match.group(1).replace(" ", "").replace("-", "")
                if len(result) >= 2:
                    return result
        
        match = re.search(r'\b([A-Z]{1,2}\d{4,7})\b', clean_text)
        if match:
            return match.group(1)
            
        return None
    
    def classify_batch(self, text_list, verify_mode=False):
        """AI-based classification for texts"""
        task_desc = "verify the pre-classified categories" if verify_mode else "classify each customer log"
        
        prompt = f"""You are an expert data classifier for a Pizza Hut restaurant chain analyzing void order reasons.

TASK: {task_desc} into EXACTLY ONE category from this list:
{json.dumps(CATEGORIES)}

CLASSIFICATION RULES (apply in order):

1. "testing" - Test orders from IT, product testing
2. "promotion" - LSM offers, % discounts, meal deals, flash offers
3. "payment issue" - Card/cash/payment problems
4. "Cashier mistake" - Cashier punched wrong order, mistakenly closed
5. "Call Center mistake" - CSR/sales center errors
6. "Customer denied the order" - Customer refused/rejected
7. "Customer Cancel order" - Customer requested cancellation
8. "double punch" - Order placed twice/duplicate
9. "grid issue" - Delivery grid/coverage problems
10. "location" - Wrong address, different outlet, wrong location
11. "phone" - Phone not working/answering, wrong number
12. "Order delay" - Late delivery, delay issues
13. "order type change" - Changing from dine-in to delivery, etc.
14. "cus. Change the order" - Customer changing items/time/order details
15. "out of stock" - Items not available
16. "rider issue" - Delivery rider problems
17. "system issue" - System/technical errors
18. "order cancelled by aggregator" - Uber/PickMe cancellation
19. "product issue or complain" - Quality/product complaints
20. "cus.related issue" - Other customer-specific issues
21. "other" - ONLY if nothing else fits
22. "voids without clear reason/ remark" - Use this when text exists but does NOT explain WHY the order was voided (e.g. "new order 116", "customer place takeaway order", random descriptions)

INPUT DATA TO CLASSIFY:
{json.dumps(text_list, indent=2)}

OUTPUT: Return a JSON object with key "predictions" containing a list of category strings.
"""

        try:
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a precise data classification API. Output only valid JSON with a 'predictions' array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            predictions = data.get("predictions", [])
            
            validated = []
            for pred in predictions:
                if pred in CATEGORIES:
                    validated.append(pred)
                else:
                    pred_lower = pred.lower()
                    matched = False
                    for cat in CATEGORIES:
                        if cat.lower() == pred_lower or pred_lower in cat.lower():
                            validated.append(cat)
                            matched = True
                            break
                    if not matched:
                        validated.append("other")
            
            return validated

        except Exception as e:
            self.log(f"API Error: {e}")
            return ["ERROR"] * len(text_list)
    
    def post_process_category(self, text, ai_category):
        """Post-process AI predictions with additional validation"""
        if not text or pd.isna(text):
            return ai_category
        
        text_lower = str(text).lower()
        
        if ai_category == "other":
            rule_category = self.apply_keyword_rules(text)
            if rule_category:
                return rule_category
        
        if re.search(r'new\s*(bill|order|dkt)', text_lower) and re.search(r'change|want', text_lower):
            if ai_category in ["other", "cus.related issue"]:
                return "cus. Change the order"
        
        if re.search(r'(from|to)\s+\w+\s*(outlet|branch)', text_lower):
            return "location"
        
        return ai_category
    
    def export_summary_report(self, summary_text, output_path):
        """Export summary report to a separate file"""
        summary_path = output_path.replace(".xlsx", "_summary.txt")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"Void Bills Classification Summary\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input File: {self.input_file.get()}\n")
                f.write(f"Output File: {output_path}\n")
                f.write("=" * 50 + "\n\n")
                f.write(summary_text)
            self.log(f"Summary exported to: {os.path.basename(summary_path)}")
        except Exception as e:
            self.log(f"Could not export summary: {e}")
    
    def run_classification(self):
        """Main classification process"""
        if self.is_running:
            messagebox.showwarning("Running", "Classification is already in progress!")
            return
        
        # Validate inputs
        if not self.api_key.get():
            messagebox.showerror("Error", "Please enter your Groq API key!")
            return
        
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input file!")
            return
        
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Input file not found!")
            return
        
        # Start in a separate thread
        thread = threading.Thread(target=self._run_classification_thread)
        thread.daemon = True
        thread.start()
    
    def _run_classification_thread(self):
        """Classification thread to keep UI responsive"""
        self.is_running = True
        self.root.after(0, lambda: self.run_btn.config(state=tk.DISABLED, text="Running..."))
        
        try:
            # Initialize Groq client
            self.log("Initializing API connection...")
            self.client = Groq(api_key=self.api_key.get())
            
            # Read input file
            self.update_status("Reading input file...", 5)
            self.log(f"Reading {os.path.basename(self.input_file.get())}...")
            
            df = pd.read_excel(self.input_file.get())
            order_col_name = 'Order No'
            
            df['Temp_Order_ID'] = df[order_col_name].ffill()
            
            def combine_text(x):
                return " ".join(set([str(s).strip() for s in x if pd.notna(s) and str(s).strip() != '']))
            
            grouped = df.groupby('Temp_Order_ID')[['Reason', 'Remark']].agg(combine_text)
            grouped['AI_Input'] = (grouped['Reason'] + " " + grouped['Remark']).str.strip()
            
            self.log("Extracting New Bill Numbers...")
            grouped['Extracted_Bill_No'] = grouped['AI_Input'].apply(self.extract_new_bill_id)
            
            bill_number_map = grouped['Extracted_Bill_No'].to_dict()
            
            orders_with_text = grouped[grouped['AI_Input'].str.len() > 1].copy()
            orders_empty = grouped[grouped['AI_Input'].str.len() <= 1].index.tolist()
            
            total_orders = len(grouped)
            self.log(f"Total Orders: {total_orders}")
            self.log(f"Orders to Classify: {len(orders_with_text)}")
            self.log(f"Empty Orders: {len(orders_empty)}")
            
            # Step 1: Rule-based classification
            self.update_status("Applying rule-based classification...", 15)
            self.log("\nStep 1: Applying rule-based classification...")
            
            orders_with_text['Rule_Category'] = orders_with_text['AI_Input'].apply(self.apply_keyword_rules)
            
            rule_classified = orders_with_text[orders_with_text['Rule_Category'].notna()]
            needs_ai = orders_with_text[orders_with_text['Rule_Category'].isna()]
            
            self.log(f"  Rule-based classified: {len(rule_classified)} orders")
            self.log(f"  Needs AI classification: {len(needs_ai)} orders")
            
            category_map = {}
            
            # Add rule-classified orders to map
            for order_id, row in rule_classified.iterrows():
                category_map[order_id] = row['Rule_Category']
            
            # AI Verify Rules option - verify rule-based classifications
            if self.ai_verify_rules.get() and len(rule_classified) > 0:
                self.log("\nAI Verification: Double-checking rule-based classifications...")
                rule_ids = rule_classified.index.tolist()
                rule_texts = rule_classified['AI_Input'].tolist()
                
                total_verify_batches = (len(rule_texts) + BATCH_SIZE - 1) // BATCH_SIZE
                verified_count = 0
                changed_count = 0
                
                for i in range(0, len(rule_texts), BATCH_SIZE):
                    batch_num = i // BATCH_SIZE + 1
                    progress = 15 + (batch_num / total_verify_batches) * 20
                    self.update_status(f"Verifying batch {batch_num}/{total_verify_batches}...", progress)
                    
                    batch_ids = rule_ids[i : i + BATCH_SIZE]
                    batch_texts = rule_texts[i : i + BATCH_SIZE]
                    ai_results = self.classify_batch(batch_texts, verify_mode=True)
                    
                    for j, order_id in enumerate(batch_ids):
                        if j < len(ai_results):
                            rule_cat = category_map[order_id]
                            ai_cat = ai_results[j]
                            if ai_cat != "ERROR" and ai_cat != rule_cat:
                                # AI disagrees, use AI's classification
                                category_map[order_id] = ai_cat
                                changed_count += 1
                            verified_count += 1
                    
                    time.sleep(0.3)
                
                self.log(f"  Verified {verified_count} orders, {changed_count} corrections made")
            
            # Step 2: AI classification for unclassified orders
            if len(needs_ai) > 0:
                self.log("\nStep 2: AI classification for remaining orders...")
                ids_to_classify = needs_ai.index.tolist()
                texts_to_classify = needs_ai['AI_Input'].tolist()
                ai_predictions = []
                
                total_batches = (len(texts_to_classify) + BATCH_SIZE - 1) // BATCH_SIZE
                base_progress = 35 if self.ai_verify_rules.get() else 20
                
                for i in range(0, len(texts_to_classify), BATCH_SIZE):
                    batch_num = i // BATCH_SIZE + 1
                    progress = base_progress + (batch_num / total_batches) * 50
                    self.update_status(f"Processing batch {batch_num}/{total_batches}...", progress)
                    
                    batch = texts_to_classify[i : i + BATCH_SIZE]
                    batch_results = self.classify_batch(batch)
                    
                    if len(batch_results) != len(batch):
                        diff = len(batch) - len(batch_results)
                        if diff > 0:
                            batch_results += ["other"] * diff
                        else:
                            batch_results = batch_results[:len(batch)]
                    
                    ai_predictions.extend(batch_results)
                    self.log(f"  Batch {batch_num}/{total_batches} complete")
                    time.sleep(0.5)
                
                # Post-process
                self.log("\nStep 3: Post-processing AI predictions...")
                for idx, order_id in enumerate(ids_to_classify):
                    text = texts_to_classify[idx]
                    ai_cat = ai_predictions[idx]
                    final_cat = self.post_process_category(text, ai_cat)
                    category_map[order_id] = final_cat
            
            # Handle empty orders
            for order_id in orders_empty:
                category_map[order_id] = "no reason/remark"
            
            # Apply results
            self.update_status("Applying results...", 85)
            df['Predicted_Category'] = df['Temp_Order_ID'].map(category_map)
            df['Extracted_New_Bill'] = df['Temp_Order_ID'].map(bill_number_map)
            
            mask_child_rows = df[order_col_name].isna()
            df.loc[mask_child_rows, 'Predicted_Category'] = None
            df.loc[mask_child_rows, 'Extracted_New_Bill'] = None
            
            del df['Temp_Order_ID']
            
            # Build summary
            parent_rows = df[df[order_col_name].notna()]
            summary = parent_rows['Predicted_Category'].value_counts()
            
            summary_text = f"TOTAL ORDERS PROCESSED: {total_orders}\n"
            summary_text += f"Rule-based classified: {len(rule_classified)}\n"
            summary_text += f"AI classified: {len(needs_ai)}\n"
            summary_text += f"Empty orders: {len(orders_empty)}\n\n"
            summary_text += "CATEGORY BREAKDOWN:\n"
            summary_text += "-" * 40 + "\n"
            for cat, count in summary.items():
                pct = (count / total_orders) * 100
                summary_text += f"{cat}: {count} ({pct:.1f}%)\n"
            
            self.last_summary = summary_text
            
            # Log summary
            self.log("\n" + "=" * 50)
            self.log("CLASSIFICATION SUMMARY")
            self.log("=" * 50)
            for cat, count in summary.items():
                self.log(f"  {cat}: {count}")
            self.log("=" * 50)
            
            # Export summary if enabled
            if self.export_summary.get():
                self.export_summary_report(summary_text, self.output_file.get())
            
            # Apply highlighting and save
            self.update_status("Saving output file...", 95)
            self.log(f"\nSaving to {os.path.basename(self.output_file.get())}...")
            
            def highlight_rows(row):
                styles = [''] * len(row)
                cat_val = row['Predicted_Category']
                if cat_val == "no reason/remark":
                    idx = row.index.get_loc('Predicted_Category')
                    styles[idx] = 'background-color: #FFFF00'  # Yellow
                elif cat_val == "voids without clear reason/ remark":
                    idx = row.index.get_loc('Predicted_Category')
                    styles[idx] = 'background-color: #FFD700'  # Gold
                elif cat_val == "ERROR":
                    idx = row.index.get_loc('Predicted_Category')
                    styles[idx] = 'background-color: #FF6B6B'  # Red
                return styles
            
            styled_df = df.style.apply(highlight_rows, axis=1)
            styled_df.to_excel(self.output_file.get(), index=False)
            
            self.update_status("Done!", 100)
            self.log("\nClassification complete!")
            
            # Show success message
            self.root.after(0, lambda: messagebox.showinfo(
                "Success", 
                f"Classification complete!\n\n"
                f"Processed {total_orders} orders.\n"
                f"- Rule-based: {len(rule_classified)}\n"
                f"- AI classified: {len(needs_ai)}\n\n"
                f"Output saved to:\n{os.path.basename(self.output_file.get())}"
            ))
            
        except FileNotFoundError:
            self.log("Error: File not found!")
            self.root.after(0, lambda: messagebox.showerror("Error", "Input file not found!"))
        except Exception as e:
            self.log(f"Error: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"An error occurred:\n{str(e)}"))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.run_btn.config(
                state=tk.NORMAL, 
                text="Run Classification"
            ))


def main():
    root = tk.Tk()
    
    # Set icon if available
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
    
    app = VoidBillsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
