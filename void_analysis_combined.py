"""
Void Bills Analysis Tool - Combined Edition
A comprehensive GUI application combining:
1. Categorization (from void_bills_app.py)
2. Void Bills Report (from Void_Bills_Report_Colab.ipynb)
3. Fraud Detection Analysis (from Fraud_Detection_Analysis.ipynb)
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import json
import time
import re
from datetime import datetime, timedelta

# Chart imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ============= CONSTANTS =============
BATCH_SIZE = 20
MODEL_NAME = "openai/gpt-oss-120b"
APP_VERSION = "2.0.0"

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

# Friendly names mapping (from Void_Bills_Report_Colab.ipynb)
FRIENDLY_NAMES = {
    'cus. Change the order': 'Change of customer request',
    'promotion': 'Promotion',
    'Cashier mistake': 'Cashier mistake',
    'Customer denied the order': 'Customer denied the order',
    'Customer Cancel order': 'Customer Cancel order',
    'cus.related issue': 'Customer Related Issue',
    'grid issue': 'Grid issue',
    'phone': 'Contact Number Issues',
    'order without reason/ remark': 'Orders Without Reason / Remark',
    'voids without clear reason/ remark': 'Voids Without Clear Reason / Remark',
    'no reason/remark': 'No Reason/Remark',
    'Order delay': 'Order Delay',
    'order type change': 'Order Type Change',
    'system issue': 'System issue / breakdown',
    'rider issue': 'Riders related issues',
    'double punch': 'Order Double Punched',
    'payment issue': 'Payment Issues',
    'out of stock': 'Out of Stock',
    'location': 'Location',
    'testing': 'Testing',
    'Call Center mistake': 'CSR Issue',
    'product issue or complain': 'Product issue or complain',
    'order cancelled by aggregator': 'Order cancelled by aggregator',
    'other': 'Other'
}

# ============= RULE-BASED CLASSIFICATION (from void_bills_app.py) =============
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


def apply_keyword_rules(text):
    """Apply rule-based classification (from void_bills_app.py)"""
    if not text or pd.isna(text):
        return None
    text_lower = str(text).lower()
    for category in PRIORITY_ORDER:
        if category in KEYWORD_RULES:
            for pattern in KEYWORD_RULES[category]:
                if re.search(pattern, text_lower):
                    return category
    return None


def extract_new_bill_id(text):
    """Extract new bill numbers from text (from void_bills_app.py)"""
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


def is_suspiciously_round(amount):
    """Check if amount is suspiciously round (from Fraud_Detection_Analysis.ipynb)"""
    if pd.isna(amount) or amount < 1000:
        return False
    return amount % 1000 == 0 or amount % 500 == 0


class VoidAnalysisCombined:
    """Main application combining all three analysis tools."""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"Void Bills Analysis Tool v{APP_VERSION}")
        self.root.geometry("1300x850")
        self.root.minsize(1100, 750)
        
        # Data storage
        self.raw_df = None
        self.categorized_df = None
        self.parent_df = None
        self.order_col = None
        self.avail_cols = []
        
        # Fraud analysis results (from Fraud_Detection_Analysis.ipynb)
        self.high_value_voids = pd.DataFrame()
        self.no_reason_voids = pd.DataFrame()
        self.late_night_voids = pd.DataFrame()
        self.round_voids = pd.DataFrame()
        self.testing_voids = pd.DataFrame()
        self.extreme_delay_voids = pd.DataFrame()
        self.repeat_phone_df = pd.DataFrame()
        self.phone_summary = pd.DataFrame()
        self.voider_stats = pd.DataFrame()
        self.frequent_voiders = pd.DataFrame()
        self.outlet_stats = pd.DataFrame()
        self.anomaly_outlets = pd.DataFrame()
        self.high_risk_orders = pd.DataFrame()
        self.critical_orders = pd.DataFrame()
        
        # Thresholds
        self.amount_threshold = 0
        self.avg_voids = 0
        
        # Processing state
        self.is_running = False
        self.client = None
        
        # API settings
        self.api_key = tk.StringVar()
        self.ai_verify_rules = tk.BooleanVar(value=False)
        
        # Setup UI
        self.setup_styles()
        self.create_ui()
        self.load_settings()
        
    def setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 10), padding=5)
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('SubHeader.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('TNotebook', background='#f0f0f0')
        style.configure('TNotebook.Tab', font=('Segoe UI', 10), padding=[15, 5])
        
    def create_ui(self):
        """Create the main user interface."""
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Label(main_frame, text="Void Bills Analysis Tool", style='Header.TLabel')
        header.pack(pady=(0, 10))
        
        # Notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Categorization
        self.create_categorization_tab()
        
        # Tab 2: Void Bills Report
        self.create_report_tab()
        
        # Tab 3: Fraud Detection
        self.create_fraud_tab()
        
        # Tab 4: Charts
        self.create_charts_tab()
        
        # Tab 5: Export
        self.create_export_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        
    # ==================== TAB 1: CATEGORIZATION ====================
    def create_categorization_tab(self):
        """Create categorization tab (from void_bills_app.py)."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="1. Categorization")
        
        # Two columns
        left_frame = ttk.Frame(tab)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(tab)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Left: Input settings
        # API Key
        api_frame = ttk.LabelFrame(left_frame, text="API Configuration", padding=10)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(api_frame, text="Groq API Key:").pack(anchor=tk.W)
        api_row = ttk.Frame(api_frame)
        api_row.pack(fill=tk.X, pady=5)
        
        self.api_entry = ttk.Entry(api_row, textvariable=self.api_key, show="*", width=40)
        self.api_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(api_row, text="Save", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        
        # File selection
        file_frame = ttk.LabelFrame(left_frame, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="Note: Clean up void bill listing first", foreground="gray").pack(anchor=tk.W)
        
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar(value="categorized_orders_clean.xlsx")
        
        ttk.Label(file_frame, text="Input Excel File:").pack(anchor=tk.W, pady=(5, 0))
        input_row = ttk.Frame(file_frame)
        input_row.pack(fill=tk.X, pady=2)
        ttk.Entry(input_row, textvariable=self.input_file, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_row, text="Browse", command=self.browse_input).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(file_frame, text="Output File:").pack(anchor=tk.W, pady=(5, 0))
        output_row = ttk.Frame(file_frame)
        output_row.pack(fill=tk.X, pady=2)
        ttk.Entry(output_row, textvariable=self.output_file, width=40).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Options
        options_frame = ttk.LabelFrame(left_frame, text="Options", padding=10)
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(options_frame, text="AI verify rule-based classifications", 
                       variable=self.ai_verify_rules).pack(anchor=tk.W)
        
        # Run button
        self.run_cat_btn = ttk.Button(left_frame, text="Run Categorization", command=self.run_categorization)
        self.run_cat_btn.pack(pady=10)
        
        # Progress
        self.cat_progress = ttk.Progressbar(left_frame, mode='determinate')
        self.cat_progress.pack(fill=tk.X, pady=5)
        
        # Right: Log
        log_frame = ttk.LabelFrame(right_frame, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.cat_log = scrolledtext.ScrolledText(log_frame, height=20, wrap=tk.WORD, font=('Consolas', 9))
        self.cat_log.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(right_frame, text="Clear Log", command=lambda: self.cat_log.delete(1.0, tk.END)).pack(pady=5)
        
    # ==================== TAB 2: VOID BILLS REPORT ====================
    def create_report_tab(self):
        """Create void bills report tab (from Void_Bills_Report_Colab.ipynb)."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="2. Void Bills Report")
        
        # Header
        ttk.Label(tab, text="Load categorized data to view report", style='SubHeader.TLabel').pack(anchor=tk.W)
        
        # Load button
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Load Categorized Data", command=self.load_categorized_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Refresh Report", command=self.refresh_report).pack(side=tk.LEFT, padx=5)
        
        # Report display area with scrollbar
        report_container = ttk.Frame(tab)
        report_container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(report_container)
        scrollbar = ttk.Scrollbar(report_container, orient=tk.VERTICAL, command=canvas.yview)
        self.report_frame = ttk.Frame(canvas)
        
        self.report_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.report_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    # ==================== TAB 3: FRAUD DETECTION ====================
    def create_fraud_tab(self):
        """Create fraud detection tab (from Fraud_Detection_Analysis.ipynb)."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="3. Fraud Detection")
        
        # Header and button
        header_frame = ttk.Frame(tab)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(header_frame, text="Fraud Detection Analysis", style='SubHeader.TLabel').pack(side=tk.LEFT)
        ttk.Button(header_frame, text="Run Fraud Detection", command=self.run_fraud_detection).pack(side=tk.RIGHT)
        
        # Progress
        self.fraud_progress = ttk.Progressbar(tab, mode='determinate')
        self.fraud_progress.pack(fill=tk.X, pady=5)
        
        # Summary panel
        summary_frame = ttk.LabelFrame(tab, text="Fraud Risk Summary", padding=10)
        summary_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.fraud_summary_text = tk.Text(summary_frame, height=8, font=('Consolas', 10), bg='#2d2d2d', fg='white')
        self.fraud_summary_text.pack(fill=tk.X)
        self.fraud_summary_text.insert(tk.END, "Run fraud detection to see summary...")
        self.fraud_summary_text.config(state=tk.DISABLED)
        
        # High risk orders table
        table_frame = ttk.LabelFrame(tab, text="High Risk Orders (Priority Investigation)", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Order', 'Outlet', 'Void By', 'Amount', 'Category', 'Flags', 'Risk')
        self.fraud_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12)
        
        for col in columns:
            self.fraud_tree.heading(col, text=col)
            self.fraud_tree.column(col, width=120)
        
        v_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.fraud_tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.fraud_tree.xview)
        self.fraud_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.fraud_tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
    # ==================== TAB 4: CHARTS ====================
    def create_charts_tab(self):
        """Create charts tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="4. Charts")
        
        # Chart selection
        select_frame = ttk.LabelFrame(tab, text="Select Chart", padding=10)
        select_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.chart_var = tk.StringVar(value="category_breakdown")
        charts = [
            ("Category Breakdown", "category_breakdown"),
            ("Top Outlets by Count", "outlet_count"),
            ("Top Outlets by Value", "outlet_value"),
            ("Order Type Distribution", "order_type"),
            ("Channel-wise by Outlet", "channel_wise"),
            ("Fraud Risk Distribution", "fraud_risk"),
            ("Top Voiders", "top_voiders"),
            ("Void Hour Distribution", "void_hours"),
        ]
        
        for i, (text, value) in enumerate(charts):
            ttk.Radiobutton(select_frame, text=text, variable=self.chart_var, 
                           value=value, command=self.update_chart).grid(row=i//4, column=i%4, sticky=tk.W, padx=10, pady=2)
        
        # Chart canvas
        self.chart_container = ttk.Frame(tab, relief=tk.SUNKEN, borderwidth=1)
        self.chart_container.pack(fill=tk.BOTH, expand=True)
        
        self.chart_placeholder = ttk.Label(self.chart_container, text="Load data and select a chart type")
        self.chart_placeholder.pack(expand=True)
        
    # ==================== TAB 5: EXPORT ====================
    def create_export_tab(self):
        """Create export tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="5. Export")
        
        ttk.Label(tab, text="Export Reports", style='SubHeader.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Export options
        export_frame = ttk.LabelFrame(tab, text="Export Options", padding=15)
        export_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Button(export_frame, text="Export Void Bills Report", 
                  command=lambda: self.export_report('void_bills')).pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="Export Fraud Detection Report", 
                  command=lambda: self.export_report('fraud')).pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="Export Combined Full Report", 
                  command=lambda: self.export_report('combined')).pack(fill=tk.X, pady=5)
        
        # Export status
        self.export_status = ttk.Label(tab, text="")
        self.export_status.pack(anchor=tk.W, pady=10)
        
    # ==================== HELPER METHODS ====================
    def log(self, message):
        """Add message to categorization log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.cat_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.cat_log.see(tk.END)
        self.root.update_idletasks()
        
    def load_settings(self):
        """Load settings from .env file."""
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("API_KEY="):
                        self.api_key.set(line.split("=", 1)[1].strip())
                    elif line.startswith("AI_VERIFY_RULES="):
                        self.ai_verify_rules.set(line.split("=", 1)[1].strip().lower() == "true")
        
    def save_settings(self):
        """Save settings to .env file."""
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        with open(env_path, "w") as f:
            f.write(f"API_KEY={self.api_key.get()}\n")
            f.write(f"AI_VERIFY_RULES={str(self.ai_verify_rules.get()).lower()}\n")
        messagebox.showinfo("Saved", "Settings saved!")
        
    def browse_input(self):
        """Browse for input file."""
        filename = filedialog.askopenfilename(
            title="Select Input Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if filename:
            self.input_file.set(filename)
            
    # ==================== CATEGORIZATION (from void_bills_app.py) ====================
    def run_categorization(self):
        """Run categorization process."""
        if self.is_running:
            return
            
        if not self.api_key.get():
            messagebox.showerror("Error", "Please enter API key")
            return
            
        if not self.input_file.get() or not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Please select valid input file")
            return
            
        self.is_running = True
        self.run_cat_btn.config(state='disabled')
        threading.Thread(target=self._categorize_thread, daemon=True).start()
        
    def _categorize_thread(self):
        """Categorization worker thread (logic from void_bills_app.py)."""
        try:
            self.log("Initializing API connection...")
            self.client = Groq(api_key=self.api_key.get())
            
            self.log(f"Reading {os.path.basename(self.input_file.get())}...")
            df = pd.read_excel(self.input_file.get())
            order_col_name = 'Order No'
            
            df['Temp_Order_ID'] = df[order_col_name].ffill()
            
            def combine_text(x):
                return " ".join(set([str(s).strip() for s in x if pd.notna(s) and str(s).strip() != '']))
            
            grouped = df.groupby('Temp_Order_ID')[['Reason', 'Remark']].agg(combine_text)
            grouped['AI_Input'] = (grouped['Reason'] + " " + grouped['Remark']).str.strip()
            grouped['Extracted_Bill_No'] = grouped['AI_Input'].apply(extract_new_bill_id)
            
            bill_number_map = grouped['Extracted_Bill_No'].to_dict()
            
            orders_with_text = grouped[grouped['AI_Input'].str.len() > 1].copy()
            orders_empty = grouped[grouped['AI_Input'].str.len() <= 1].index.tolist()
            
            total_orders = len(grouped)
            self.log(f"Total Orders: {total_orders}")
            self.log(f"Orders to Classify: {len(orders_with_text)}")
            
            # Rule-based classification
            self.log("Applying rule-based classification...")
            orders_with_text['Rule_Category'] = orders_with_text['AI_Input'].apply(apply_keyword_rules)
            
            rule_classified = orders_with_text[orders_with_text['Rule_Category'].notna()]
            needs_ai = orders_with_text[orders_with_text['Rule_Category'].isna()]
            
            self.log(f"  Rule-based: {len(rule_classified)} orders")
            self.log(f"  Needs AI: {len(needs_ai)} orders")
            
            category_map = {}
            for order_id, row in rule_classified.iterrows():
                category_map[order_id] = row['Rule_Category']
            
            # AI verification if enabled
            if self.ai_verify_rules.get() and len(rule_classified) > 0:
                self.log("AI Verification: Checking rule-based classifications...")
                self._ai_verify_batch(rule_classified, category_map)
            
            # AI classification for remaining
            if len(needs_ai) > 0:
                self.log("AI classification for remaining orders...")
                self._ai_classify_batch(needs_ai, category_map)
            
            # Handle empty orders
            for order_id in orders_empty:
                category_map[order_id] = "no reason/remark"
            
            # Apply results
            self.cat_progress['value'] = 90
            df['Predicted_Category'] = df['Temp_Order_ID'].map(category_map)
            df['Extracted_New_Bill'] = df['Temp_Order_ID'].map(bill_number_map)
            
            mask_child_rows = df[order_col_name].isna()
            df.loc[mask_child_rows, 'Predicted_Category'] = None
            df.loc[mask_child_rows, 'Extracted_New_Bill'] = None
            
            del df['Temp_Order_ID']
            
            # Save output
            output_path = os.path.join(os.path.dirname(self.input_file.get()), self.output_file.get())
            df.to_excel(output_path, index=False)
            
            self.raw_df = df.copy()
            self.categorized_df = df.copy()
            
            self.cat_progress['value'] = 100
            self.log(f"Saved to: {self.output_file.get()}")
            self.log("Categorization complete!")
            
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Categorization complete!\n{total_orders} orders processed."))
            
        except Exception as e:
            self.log(f"Error: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.run_cat_btn.config(state='normal'))
            
    def _ai_verify_batch(self, rule_classified, category_map):
        """AI verification of rule-based classifications."""
        rule_ids = rule_classified.index.tolist()
        rule_texts = rule_classified['AI_Input'].tolist()
        
        for i in range(0, len(rule_texts), BATCH_SIZE):
            batch_ids = rule_ids[i:i+BATCH_SIZE]
            batch_texts = rule_texts[i:i+BATCH_SIZE]
            
            progress = 15 + ((i / len(rule_texts)) * 20)
            self.cat_progress['value'] = progress
            
            ai_results = self._classify_batch(batch_texts)
            
            for j, order_id in enumerate(batch_ids):
                if j < len(ai_results) and ai_results[j] != "ERROR":
                    if ai_results[j] != category_map[order_id]:
                        category_map[order_id] = ai_results[j]
            
            time.sleep(0.3)
            
    def _ai_classify_batch(self, needs_ai, category_map):
        """AI classification for unclassified orders."""
        ids_list = needs_ai.index.tolist()
        texts_list = needs_ai['AI_Input'].tolist()
        
        for i in range(0, len(texts_list), BATCH_SIZE):
            batch_ids = ids_list[i:i+BATCH_SIZE]
            batch_texts = texts_list[i:i+BATCH_SIZE]
            
            progress = 35 + ((i / len(texts_list)) * 50)
            self.cat_progress['value'] = progress
            
            ai_results = self._classify_batch(batch_texts)
            
            for j, order_id in enumerate(batch_ids):
                if j < len(ai_results):
                    category_map[order_id] = ai_results[j] if ai_results[j] != "ERROR" else "other"
            
            self.log(f"  Batch {i//BATCH_SIZE + 1} complete")
            time.sleep(0.5)
            
    def _classify_batch(self, text_list):
        """AI classification batch (from void_bills_app.py)."""
        prompt = f"""Classify each void order reason into ONE category from: {json.dumps(CATEGORIES)}

INPUT: {json.dumps(text_list, indent=2)}

OUTPUT: JSON with "predictions" array of category strings."""

        try:
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Classification API. Output valid JSON only."},
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
                    validated.append("other")
            return validated
            
        except Exception as e:
            self.log(f"API Error: {e}")
            return ["ERROR"] * len(text_list)
            
    # ==================== VOID BILLS REPORT (from Void_Bills_Report_Colab.ipynb) ====================
    def load_categorized_data(self):
        """Load categorized data for report."""
        if self.categorized_df is not None:
            self._prepare_parent_df()
            self.refresh_report()
            return
            
        # Try to load from file
        cat_file = filedialog.askopenfilename(
            title="Select Categorized Excel File",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile="categorized_orders_clean.xlsx"
        )
        
        if cat_file and os.path.exists(cat_file):
            self.categorized_df = pd.read_excel(cat_file)
            self._prepare_parent_df()
            self.refresh_report()
            self.status_var.set(f"Loaded: {os.path.basename(cat_file)}")
            
    def _prepare_parent_df(self):
        """Prepare parent dataframe for analysis."""
        df = self.categorized_df.copy()
        self.order_col = df.columns[0]
        self.parent_df = df[df[self.order_col].notna()].copy()
        
        # Parse dates (from Fraud_Detection_Analysis.ipynb)
        if 'Order Date' in self.parent_df.columns:
            self.parent_df['Order_Date_Parsed'] = pd.to_datetime(self.parent_df['Order Date'], errors='coerce')
        if 'Void Date' in self.parent_df.columns:
            self.parent_df['Void_Date_Parsed'] = pd.to_datetime(self.parent_df['Void Date'], errors='coerce')
        if 'Order Time' in self.parent_df.columns:
            self.parent_df['Order_Time_Parsed'] = pd.to_datetime(self.parent_df['Order Time'], errors='coerce')
        
        # Calculate time gap
        if 'Order_Time_Parsed' in self.parent_df.columns and 'Void_Date_Parsed' in self.parent_df.columns:
            self.parent_df['Time_Gap_Hours'] = (self.parent_df['Void_Date_Parsed'] - self.parent_df['Order_Time_Parsed']).dt.total_seconds() / 3600
        
        # Standard columns
        display_cols = [self.order_col, 'Outlet', 'Order Type', 'Order Date', 'Reason', 'Void By ', 'Amount']
        self.avail_cols = [c for c in display_cols if c in self.parent_df.columns]
        
    def refresh_report(self):
        """Refresh the void bills report display."""
        if self.parent_df is None:
            return
            
        # Clear existing content
        for widget in self.report_frame.winfo_children():
            widget.destroy()
            
        parent_df = self.parent_df
        order_col = self.order_col
        
        # Get month from data
        month_str = "Current Period"
        if 'Order Date' in parent_df.columns:
            try:
                dates = pd.to_datetime(parent_df['Order Date'], errors='coerce')
                if dates.notna().any():
                    month_str = dates.dropna().iloc[0].strftime('%B, %Y')
            except:
                pass
        
        # Title
        title = ttk.Label(self.report_frame, text=f"VOID BILLS ANALYSIS - {month_str}", 
                         font=('Segoe UI', 14, 'bold'), foreground='#1976D2')
        title.pack(pady=10)
        
        # Key Findings (from Void_Bills_Report_Colab.ipynb)
        total_voids = len(parent_df)
        outlet_counts = parent_df['Outlet'].value_counts()
        top_outlet = outlet_counts.index[0] if len(outlet_counts) > 0 else "N/A"
        top_outlet_count = outlet_counts.iloc[0] if len(outlet_counts) > 0 else 0
        
        reason_counts = parent_df['Predicted_Category'].value_counts()
        main_reason = reason_counts.index[0] if len(reason_counts) > 0 else "N/A"
        main_reason_count = reason_counts.iloc[0] if len(reason_counts) > 0 else 0
        
        total_value = parent_df['Amount'].sum() if 'Amount' in parent_df.columns else 0
        
        # Summary box
        summary_frame = ttk.LabelFrame(self.report_frame, text="Key Findings", padding=10)
        summary_frame.pack(fill=tk.X, pady=10, padx=10)
        
        findings = f"""Total Void Bills: {total_voids}
Total Void Value: Rs. {total_value:,.2f}
Top Outlet: {top_outlet} ({top_outlet_count} voids)
Main Reason: {main_reason} ({main_reason_count} cases)
Number of Outlets: {parent_df['Outlet'].nunique()}"""
        
        ttk.Label(summary_frame, text=findings, font=('Consolas', 10)).pack(anchor=tk.W)
        
        # Top 10 by Amount
        top_frame = ttk.LabelFrame(self.report_frame, text="Top 10 Void Bills by Amount", padding=10)
        top_frame.pack(fill=tk.X, pady=10, padx=10)
        
        top_amount = parent_df.nlargest(10, 'Amount')[self.avail_cols].copy()
        self._create_table(top_frame, top_amount)
        
        # Category Breakdown
        cat_frame = ttk.LabelFrame(self.report_frame, text="Category Breakdown", padding=10)
        cat_frame.pack(fill=tk.X, pady=10, padx=10)
        
        all_reasons = reason_counts.reset_index()
        all_reasons.columns = ['Category', 'Count']
        all_reasons['Friendly Name'] = all_reasons['Category'].map(FRIENDLY_NAMES).fillna(all_reasons['Category'])
        self._create_table(cat_frame, all_reasons[['Friendly Name', 'Count']])
        
    def _create_table(self, parent, df, max_rows=15):
        """Create a simple table from dataframe."""
        tree = ttk.Treeview(parent, columns=list(df.columns), show='headings', height=min(len(df), max_rows))
        
        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120)
        
        for _, row in df.head(max_rows).iterrows():
            values = []
            for v in row:
                if isinstance(v, float):
                    values.append(f"{v:,.2f}")
                else:
                    values.append(str(v)[:50])
            tree.insert('', tk.END, values=values)
        
        tree.pack(fill=tk.X)
        
    # ==================== FRAUD DETECTION (from Fraud_Detection_Analysis.ipynb) ====================
    def run_fraud_detection(self):
        """Run fraud detection analysis (exact logic from Fraud_Detection_Analysis.ipynb)."""
        if self.parent_df is None:
            self.load_categorized_data()
            if self.parent_df is None:
                messagebox.showerror("Error", "Please load categorized data first")
                return
                
        threading.Thread(target=self._fraud_thread, daemon=True).start()
        
    def _fraud_thread(self):
        """Fraud detection worker thread (logic from Fraud_Detection_Analysis.ipynb)."""
        try:
            parent_df = self.parent_df.copy()
            order_col = self.order_col
            
            self.fraud_progress['value'] = 5
            
            # Flag 1: High value voids (above 95th percentile)
            self.amount_threshold = parent_df['Amount'].quantile(0.95)
            self.high_value_voids = parent_df[parent_df['Amount'] >= self.amount_threshold].sort_values('Amount', ascending=False).copy()
            
            self.fraud_progress['value'] = 15
            
            # Flag 2: Frequent voiders (from notebook)
            if 'Void By ' in parent_df.columns:
                self.voider_stats = parent_df.groupby('Void By ').agg({
                    order_col: 'count',
                    'Amount': ['sum', 'mean', 'max'],
                    'Outlet': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Multiple'
                }).reset_index()
                self.voider_stats.columns = ['Void By', 'Void Count', 'Total Value', 'Avg Value', 'Max Value', 'Primary Outlet']
                self.voider_stats = self.voider_stats.sort_values('Void Count', ascending=False)
                
                self.avg_voids = self.voider_stats['Void Count'].mean()
                self.frequent_voiders = self.voider_stats[self.voider_stats['Void Count'] > self.avg_voids * 1.5].copy()
            
            self.fraud_progress['value'] = 25
            
            # Flag 3: Voids without reason
            self.no_reason_voids = parent_df[parent_df['Predicted_Category'].isin([
                'order without reason/ remark', 'voids without clear reason/ remark', 'no reason/remark'
            ])].sort_values('Amount', ascending=False).copy()
            
            self.fraud_progress['value'] = 35
            
            # Flag 4: Late night voids (from notebook)
            if 'Void_Date_Parsed' in parent_df.columns:
                parent_df['Void_Hour'] = parent_df['Void_Date_Parsed'].dt.hour
                self.late_night_voids = parent_df[(parent_df['Void_Hour'] >= 22) | (parent_df['Void_Hour'] <= 5)].sort_values('Amount', ascending=False).copy()
            else:
                self.late_night_voids = pd.DataFrame()
            
            self.fraud_progress['value'] = 45
            
            # Flag 5: Round number amounts (from notebook)
            parent_df['Is_Round'] = parent_df['Amount'].apply(is_suspiciously_round)
            self.round_voids = parent_df[parent_df['Is_Round'] == True].sort_values('Amount', ascending=False).copy()
            
            self.fraud_progress['value'] = 55
            
            # Flag 6: Repeat phone numbers (from notebook)
            if 'Contact no' in parent_df.columns:
                parent_df['Contact_Clean'] = parent_df['Contact no'].astype(str).str.strip()
                phone_counts = parent_df['Contact_Clean'].value_counts()
                repeat_phones = phone_counts[phone_counts > 1].index.tolist()
                self.repeat_phone_df = parent_df[parent_df['Contact_Clean'].isin(repeat_phones)].copy()
                
                self.phone_summary = parent_df[parent_df['Contact_Clean'].isin(repeat_phones)].groupby('Contact_Clean').agg({
                    order_col: 'count',
                    'Amount': 'sum',
                    'Outlet': lambda x: ', '.join(x.unique()[:3])
                }).reset_index()
                self.phone_summary.columns = ['Contact No', 'Void Count', 'Total Value', 'Outlets']
                self.phone_summary = self.phone_summary.sort_values('Void Count', ascending=False)
            
            self.fraud_progress['value'] = 65
            
            # Flag 7: Outlet anomalies (from notebook)
            self.outlet_stats = parent_df.groupby('Outlet').agg({
                order_col: 'count',
                'Amount': ['sum', 'mean', 'max']
            }).reset_index()
            self.outlet_stats.columns = ['Outlet', 'Void Count', 'Total Value', 'Avg Value', 'Max Value']
            
            self.outlet_stats['Count_ZScore'] = (self.outlet_stats['Void Count'] - self.outlet_stats['Void Count'].mean()) / self.outlet_stats['Void Count'].std()
            self.outlet_stats['Value_ZScore'] = (self.outlet_stats['Total Value'] - self.outlet_stats['Total Value'].mean()) / self.outlet_stats['Total Value'].std()
            
            self.anomaly_outlets = self.outlet_stats[(self.outlet_stats['Count_ZScore'] > 1.5) | (self.outlet_stats['Value_ZScore'] > 1.5)].copy()
            
            self.fraud_progress['value'] = 75
            
            # Flag 8: Testing category
            self.testing_voids = parent_df[parent_df['Predicted_Category'] == 'testing'].sort_values('Amount', ascending=False).copy()
            
            # Flag 9: Extreme delays (from notebook)
            if 'Time_Gap_Hours' in parent_df.columns:
                self.extreme_delay_voids = parent_df[parent_df['Time_Gap_Hours'] > 24].sort_values('Time_Gap_Hours', ascending=False).copy()
            else:
                self.extreme_delay_voids = pd.DataFrame()
            
            self.fraud_progress['value'] = 85
            
            # Combined fraud risk score (from notebook)
            parent_df['Fraud_Flags'] = 0
            parent_df['Fraud_Reasons'] = ''
            
            # Add flags with weights
            parent_df.loc[parent_df['Amount'] >= self.amount_threshold, 'Fraud_Flags'] += 1
            parent_df.loc[parent_df['Amount'] >= self.amount_threshold, 'Fraud_Reasons'] += 'High Value; '
            
            parent_df.loc[parent_df['Predicted_Category'].isin(['order without reason/ remark', 'voids without clear reason/ remark', 'no reason/remark']), 'Fraud_Flags'] += 2
            parent_df.loc[parent_df['Predicted_Category'].isin(['order without reason/ remark', 'voids without clear reason/ remark', 'no reason/remark']), 'Fraud_Reasons'] += 'No Reason; '
            
            parent_df.loc[parent_df['Predicted_Category'] == 'testing', 'Fraud_Flags'] += 1
            parent_df.loc[parent_df['Predicted_Category'] == 'testing', 'Fraud_Reasons'] += 'Testing; '
            
            parent_df.loc[parent_df['Is_Round'] == True, 'Fraud_Flags'] += 1
            parent_df.loc[parent_df['Is_Round'] == True, 'Fraud_Reasons'] += 'Round Amount; '
            
            if 'Void_Hour' in parent_df.columns:
                parent_df.loc[(parent_df['Void_Hour'] >= 22) | (parent_df['Void_Hour'] <= 5), 'Fraud_Flags'] += 1
                parent_df.loc[(parent_df['Void_Hour'] >= 22) | (parent_df['Void_Hour'] <= 5), 'Fraud_Reasons'] += 'Late Night; '
            
            if 'Time_Gap_Hours' in parent_df.columns:
                parent_df.loc[parent_df['Time_Gap_Hours'] > 24, 'Fraud_Flags'] += 2
                parent_df.loc[parent_df['Time_Gap_Hours'] > 24, 'Fraud_Reasons'] += 'Extreme Delay; '
            
            parent_df['Risk_Level'] = pd.cut(parent_df['Fraud_Flags'], bins=[-1, 0, 1, 2, 10],
                                             labels=['Low', 'Medium', 'High', 'Critical'])
            
            self.high_risk_orders = parent_df[parent_df['Fraud_Flags'] >= 2].sort_values(['Fraud_Flags', 'Amount'], ascending=[False, False]).copy()
            self.critical_orders = parent_df[parent_df['Fraud_Flags'] >= 3].copy()
            
            self.parent_df = parent_df
            
            self.fraud_progress['value'] = 100
            
            # Update UI
            self.root.after(0, self._update_fraud_ui)
            
        except Exception as e:
            messagebox.showerror("Error", f"Fraud detection failed: {e}")
            import traceback
            traceback.print_exc()
            
    def _update_fraud_ui(self):
        """Update fraud detection UI."""
        # Update summary text
        summary = f"""FRAUD RISK SUMMARY
{'='*50}
High-Value Voids (>Rs.{self.amount_threshold:,.0f}): {len(self.high_value_voids)}
Voids Without Reason:                    {len(self.no_reason_voids)}
Late Night Voids (10PM-5AM):             {len(self.late_night_voids)}
Round Amount Voids:                      {len(self.round_voids)}
Testing Category:                        {len(self.testing_voids)}
Extreme Delays (>24hr):                  {len(self.extreme_delay_voids)}
Frequent Voiders:                        {len(self.frequent_voiders)}
Outlet Anomalies:                        {len(self.anomaly_outlets)}
{'='*50}
CRITICAL RISK Orders (3+ flags):         {len(self.critical_orders)}
HIGH RISK Orders (2+ flags):             {len(self.high_risk_orders)}
"""
        
        self.fraud_summary_text.config(state=tk.NORMAL)
        self.fraud_summary_text.delete(1.0, tk.END)
        self.fraud_summary_text.insert(tk.END, summary)
        self.fraud_summary_text.config(state=tk.DISABLED)
        
        # Update tree
        for item in self.fraud_tree.get_children():
            self.fraud_tree.delete(item)
            
        for _, row in self.high_risk_orders.head(100).iterrows():
            order = str(row[self.order_col])[:15]
            outlet = str(row.get('Outlet', ''))[:15]
            void_by = str(row.get('Void By ', ''))[:15]
            amount = f"{row.get('Amount', 0):,.0f}"
            category = str(row.get('Predicted_Category', ''))[:20]
            flags = str(row.get('Fraud_Reasons', ''))[:30]
            risk = str(row.get('Risk_Level', ''))
            
            self.fraud_tree.insert('', tk.END, values=(order, outlet, void_by, amount, category, flags, risk))
            
    # ==================== CHARTS ====================
    def update_chart(self):
        """Update chart display."""
        if self.parent_df is None:
            return
            
        # Clear container
        for widget in self.chart_container.winfo_children():
            widget.destroy()
            
        chart_type = self.chart_var.get()
        parent_df = self.parent_df
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        if chart_type == "category_breakdown":
            data = parent_df['Predicted_Category'].value_counts().head(15)
            bars = ax.barh(range(len(data)), data.values, color='steelblue')
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels([FRIENDLY_NAMES.get(x, x)[:25] for x in data.index], fontsize=8)
            ax.set_xlabel('Count')
            ax.set_title('Void Bills by Category')
            ax.invert_yaxis()
            
        elif chart_type == "outlet_count":
            data = parent_df['Outlet'].value_counts().head(15)
            ax.barh(range(len(data)), data.values, color='teal')
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index, fontsize=8)
            ax.set_xlabel('Number of Voids')
            ax.set_title('Top 15 Outlets by Void Count')
            ax.invert_yaxis()
            
        elif chart_type == "outlet_value":
            data = parent_df.groupby('Outlet')['Amount'].sum().sort_values(ascending=False).head(15)
            ax.barh(range(len(data)), data.values, color='coral')
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index, fontsize=8)
            ax.set_xlabel('Total Value (Rs.)')
            ax.set_title('Top 15 Outlets by Void Value')
            ax.invert_yaxis()
            
        elif chart_type == "order_type":
            data = parent_df['Order Type'].value_counts()
            colors = ['#4CAF50', '#2196F3', '#FFC107', '#9C27B0', '#FF5722']
            ax.pie(data.values, labels=data.index, autopct='%1.1f%%', colors=colors[:len(data)])
            ax.set_title('Order Type Distribution')
            
        elif chart_type == "channel_wise":
            # Channel-wise bar chart (from Void_Bills_Report_Colab.ipynb)
            channel_data = pd.crosstab(parent_df['Outlet'], parent_df['Order Type']).head(15)
            channel_data.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Channel-wise Void Bills by Outlet')
            ax.set_xlabel('Outlet')
            ax.set_ylabel('Count')
            ax.legend(title='Order Type', loc='upper right')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
        elif chart_type == "fraud_risk":
            if 'Risk_Level' in parent_df.columns:
                risk_counts = parent_df['Risk_Level'].value_counts().reindex(['Critical', 'High', 'Medium', 'Low'])
                colors = ['#D32F2F', '#FF5722', '#FFC107', '#4CAF50']
                ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=colors)
                ax.set_title('Fraud Risk Distribution')
            else:
                ax.text(0.5, 0.5, 'Run fraud detection first', ha='center', va='center', transform=ax.transAxes)
                
        elif chart_type == "top_voiders":
            if len(self.voider_stats) > 0:
                data = self.voider_stats.head(20)
                ax.barh(range(len(data)), data['Void Count'].values, color='purple')
                ax.set_yticks(range(len(data)))
                ax.set_yticklabels(data['Void By'].values, fontsize=8)
                ax.set_xlabel('Number of Voids')
                ax.set_title('Top 20 Staff by Void Count')
                if self.avg_voids > 0:
                    ax.axvline(x=self.avg_voids * 1.5, color='red', linestyle='--', label=f'Threshold: {self.avg_voids*1.5:.0f}')
                    ax.legend()
                ax.invert_yaxis()
            else:
                ax.text(0.5, 0.5, 'Run fraud detection first', ha='center', va='center', transform=ax.transAxes)
                
        elif chart_type == "void_hours":
            if 'Void_Hour' in parent_df.columns:
                hours = parent_df['Void_Hour'].dropna()
                ax.hist(hours, bins=24, range=(0, 24), color='navy', edgecolor='white')
                ax.axvspan(22, 24, alpha=0.3, color='red', label='Late Night (10PM-5AM)')
                ax.axvspan(0, 5, alpha=0.3, color='red')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Number of Voids')
                ax.set_title('Void Distribution by Hour')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'Void time data not available', ha='center', va='center', transform=ax.transAxes)
                
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, self.chart_container)
        toolbar.update()
        
    # ==================== EXPORT (combined from both notebooks) ====================
    def export_report(self, report_type):
        """Export reports."""
        if self.parent_df is None:
            messagebox.showerror("Error", "No data to export. Load data first.")
            return
            
        output_path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfilename=f"{report_type}_report.xlsx"
        )
        
        if not output_path:
            return
            
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                parent_df = self.parent_df
                order_col = self.order_col
                
                if report_type in ['void_bills', 'combined']:
                    # From Void_Bills_Report_Colab.ipynb
                    parent_df.to_excel(writer, sheet_name='All Orders', index=False)
                    
                    reason_counts = parent_df['Predicted_Category'].value_counts().reset_index()
                    reason_counts.columns = ['Category', 'Count']
                    reason_counts.to_excel(writer, sheet_name='Reason Summary', index=False)
                    
                    channel_pivot = pd.crosstab(parent_df['Outlet'], parent_df['Order Type'], margins=True)
                    channel_pivot.to_excel(writer, sheet_name='Channel-wise')
                    
                    value_pivot = parent_df.pivot_table(values='Amount', index='Outlet', columns='Order Type', aggfunc='sum', fill_value=0)
                    value_pivot['Total'] = value_pivot.sum(axis=1)
                    value_pivot.to_excel(writer, sheet_name='Outlet Values')
                
                if report_type in ['fraud', 'combined']:
                    # From Fraud_Detection_Analysis.ipynb
                    if len(self.high_risk_orders) > 0:
                        self.high_risk_orders.to_excel(writer, sheet_name='HIGH_RISK', index=False)
                    if len(self.critical_orders) > 0:
                        self.critical_orders.to_excel(writer, sheet_name='CRITICAL', index=False)
                    if len(self.high_value_voids) > 0:
                        self.high_value_voids.to_excel(writer, sheet_name='High_Value', index=False)
                    if len(self.no_reason_voids) > 0:
                        self.no_reason_voids.to_excel(writer, sheet_name='No_Reason', index=False)
                    if len(self.late_night_voids) > 0:
                        self.late_night_voids.to_excel(writer, sheet_name='Late_Night', index=False)
                    if len(self.round_voids) > 0:
                        self.round_voids.to_excel(writer, sheet_name='Round_Amounts', index=False)
                    if len(self.voider_stats) > 0:
                        self.voider_stats.to_excel(writer, sheet_name='Voider_Stats', index=False)
                    if len(self.frequent_voiders) > 0:
                        self.frequent_voiders.to_excel(writer, sheet_name='Frequent_Voiders', index=False)
                    if len(self.anomaly_outlets) > 0:
                        self.anomaly_outlets.to_excel(writer, sheet_name='Anomaly_Outlets', index=False)
                    if len(self.testing_voids) > 0:
                        self.testing_voids.to_excel(writer, sheet_name='Testing', index=False)
                    if len(self.extreme_delay_voids) > 0:
                        self.extreme_delay_voids.to_excel(writer, sheet_name='Extreme_Delays', index=False)
                    
                    # Summary sheet
                    summary_df = pd.DataFrame({
                        'Metric': [
                            'Total Orders Analyzed',
                            'High-Value Voids',
                            'Voids Without Reason',
                            'Late Night Voids',
                            'Round Amount Voids',
                            'Testing Category',
                            'Extreme Delays (>24hr)',
                            'Frequent Voiders',
                            'Anomaly Outlets',
                            'CRITICAL RISK Orders (3+ flags)',
                            'HIGH RISK Orders (2+ flags)'
                        ],
                        'Count': [
                            len(parent_df),
                            len(self.high_value_voids),
                            len(self.no_reason_voids),
                            len(self.late_night_voids),
                            len(self.round_voids),
                            len(self.testing_voids),
                            len(self.extreme_delay_voids),
                            len(self.frequent_voiders),
                            len(self.anomaly_outlets),
                            len(self.critical_orders),
                            len(self.high_risk_orders)
                        ]
                    })
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
            self.export_status.config(text=f"Exported to: {os.path.basename(output_path)}")
            messagebox.showinfo("Success", f"Report exported to:\n{output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")


def main():
    root = tk.Tk()
    app = VoidAnalysisCombined(root)
    root.mainloop()


if __name__ == "__main__":
    main()
