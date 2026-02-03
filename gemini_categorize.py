import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; environment variables can be used instead
    pass

import pandas as pd
import json
import time
import re
from tqdm import tqdm
from groq import Groq


API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise SystemExit("API_KEY not set. Add API_KEY=your_key to a .env file or set the environment variable.")

INPUT_FILE = "PH_VoidBillListing.xlsx"
OUTPUT_FILE = "categorized_orders_clean.xlsx"
BATCH_SIZE = 20
MODEL_NAME = "openai/gpt-oss-120b"
AI_VERIFY_RULES = False

client = Groq(api_key=API_KEY)

# Standardized categories
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
    "order without reason/ remark",
    "other",
    "out of stock",
    "payment issue",
    "phone",
    "product issue or complain",
    "promotion",
    "rider issue",
    "system issue",
    "testing"
]

# ============= RULE-BASED PRE-CLASSIFICATION =============
# These rules are applied BEFORE AI classification for higher accuracy

KEYWORD_RULES = {
    # TESTING - highest priority
    "testing": [
        r"\btest\s*(order|odar|oder)?\b", r"\btesting\b", r"\btes\s*oder\b",
        r"\bproduct\s*testing\b", r"\bfrom\s*(it|preshan)\b"
    ],
    
    # PROMOTION - LSM, offers, discounts
    "promotion": [
        r"\blsm\b", r"\bpromo(tion)?\b", r"\boffer\b", r"\b50\s*%\s*(off|flash|discount)?\b",
        r"\bdiscount\b", r"\bcyber\s*saving\b", r"\bmeal\s*deal\b", r"\bflash\s*offer\b",
        r"\bdon'?t\s*cook\b", r"\bhsbc\b", r"\bges\s*\d+%\b", r"\b\d+%\s*off\b",
        r"\b15\s*%\b", r"\b20\s*%\b", r"\b30\s*%\b", r"\b1000\s*off\b"
    ],
    
    # PAYMENT ISSUE - card, cash, machine problems
    "payment issue": [
        r"\bcredit\s*card\b", r"\bcard\s*(not\s*work|isn'?t\s*work|failed)\b",
        r"\bhnb\s*card\b", r"\bbank\s*card\b", r"\bvisa\b", r"\bmachine\b",
        r"\bpayment\s*(method|issue)?\b", r"\bpetty\s*cash\b", r"\bonline\s*payment\b",
        r"\bpaid\s*order\b", r"\bdon'?t\s*have\s*(enough\s*)?(money|cash)\b"
    ],
    
    # CASHIER MISTAKE - wrongly punch, mistakenly
    "Cashier mistake": [
        r"\bcashier\s*(mistake|mistakenly|mistakly|wrong)\b", r"\bwrongly\s*punch\b",
        r"\bmistakenly\s*(punch|close|add|collect|mark)\b", r"\bmistakly\b",
        r"\bcashier\s*error\b", r"\bwrong\s*(punch|order|bill|close)\b",
        r"\bcashiar\b", r"\bcashiyar\b", r"\bwrong\s*by\s*cashier\b",
        r"\bwrong\s*order[sw]?\b", r"\bwrong\s*ordewr\b", r"\bmiss\s*communication\b"
    ],
    
    # CALL CENTER MISTAKE - CSR, sale center errors
    "Call Center mistake": [
        r"\bcsr\s*(error|mistake)?\b", r"\bsale\s*cent(er|re)\s*(error|mistake|issue|request)\b",
        r"\bcall\s*cent(er|re)\s*(error|mistake|asked)\b", r"\bsales\s*cent(er|re)\b",
        r"\baccording\s*to\s*call\s*cent\b", r"\bacording\s*to\s*call\s*senter\b",
        r"\binformed\s*by\s*outlet\b", r"\binfomed\s*by\s*outlet\b"
    ],
    
    # CUSTOMER DENIED - refused, rejected order
    "Customer denied the order": [
        r"\bcustomer\s*denied\b", r"\bcux\s*denied\b", r"\bdenied\s*(the\s*)?order\b",
        r"\bcustermar\s*denied\b", r"\brefuse[d]?\s*(the\s*)?order\b",
        r"\breject(ed)?\s*(the\s*)?order\b", r"\bdeniend\b", r"\bdenaid\b"
    ],
    
    # CUSTOMER CANCEL - explicit cancellation
    "Customer Cancel order": [
        r"\bcustomer\s*(want\s*(to\s*)?)?cancel\b", r"\bcux\s*cancel\b",
        r"\bcx\s*(want\s*(to\s*)?)?cancel\b", r"\bcu\s*wont\s*to\s*cancel\b",
        r"\bcustomer\s*cansel\b", r"\bcustomer\s*cancell\b", r"\bcustomr\s*cancal\b",
        r"\bplease\s*cancel\b", r"\bcncl\b", r"\bcustomer\s*cancelled\b"
    ],
    
    # DOUBLE PUNCH - ordered twice
    "double punch": [
        r"\bordered\s*twice\b", r"\bsame\s*order\s*\d+\b", r"\b2\s*times?\s*(same\s*)?order\b",
        r"\btwo\s*orders?\s*(were\s*)?(placed|same)\b", r"\bdouble\b",
        r"\btwise\s*the\s*order\b", r"\bpast\s*same\s*order\b"
    ],
    
    # GRID ISSUE - out of grid, location problems
    "grid issue": [
        r"\bgrid\s*(issue)?\b", r"\bout\s*of\s*grid\b", r"\bgride\s*issue\b"
    ],
    
    # LOCATION - wrong address, different outlet, transfer to other outlet
    "location": [
        r"\bwrong\s*address\b", r"\bwrong\s*location\b", r"\bdifferent\s*(location|outlet)\b",
        r"\bwant\s*to\s*deliver\s*\w+\s*outlet\b", r"\bgo(ing)?\s*(to|from)\s*\w+\b",
        r"\btransfer(red)?\s*to\b", r"\bdeliver\s*from\b",
        r"\bslave\s*island\b", r"\bnearest\s*location\b", r"\bsent\s*\d+\b",
        r"\bfrom\s+\w+\s*outlet\b", r"\bto\s+\w+\s*outlet\b", r"\boutlet\s*order\b",
        r"\bdelivery\s+from\s+\w+\b", r"\bwennappuwa\b", r"\bkoswattha\b",
        r"\bnew\s*dkt\s*\w*\s*\d+\b", r"\bneew\s*order\b", r"\bkochchikade\b",
        r"\bpanadura\b", r"\bpandura\b", r"\bhavelock\b", r"\b\d{2,3}\s*-\s*\w+\b"
    ],
    
    # PHONE - not answering, wrong number
    "phone": [
        r"\bphone\s*(number\s*)?(not\s*)?(work|answer|respond)\b",
        r"\bnot\s*(answer|respond)(ing)?\s*(the\s*)?(call|phone|mobile)?\b",
        r"\bwrong\s*(phone\s*)?(number|no|mobile)\b", r"\bincorrect\s*number\b",
        r"\bcan'?t\s*contact\b", r"\bmobile\s*not\s*work\b", r"\bno\s*answer\b",
        r"\bnumber\s*wrong\b", r"\bnumber\s*not\s*work\b"
    ],
    
    # ORDER DELAY
    "Order delay": [
        r"\border\s*delay(ed)?\b", r"\bdelay\s*(issue|order)?\b", r"\blate\s*issue\b",
        r"\border\s*deley\b", r"\bpromise\s*time\b", r"\bcan'?t\s*wait\b",
        r"\bhea[vr]y\s*rain\b"
    ],
    
    # ORDER TYPE CHANGE - dine in to delivery, etc.
    "order type change": [
        r"\bchange\s*(to\s*)?(delivery|take\s*away|dine|pickup|t/?w)\b",
        r"\bwant\s*(to\s*)?(deliver|delivery)\b", r"\bwant\s*dine\b",
        r"\bpick\s*up\s*(for|to)\s*delivery\b", r"\btake\s*away\s*can[sc]al\b",
        r"\border\s*type\s*change\b"
    ],
    
    # TIME CHANGE - customer wants different time or order change
    "cus. Change the order": [
        r"\bchange\s*(the\s*)?time\b", r"\btime\s*(order|change)\b",
        r"\bwant\s*(the\s*)?order\s*@\b", r"\bwanted\s*to\s*change\s*(the\s*)?order\b",
        r"\bcustomer\s*change\b", r"\bcx\s*want(s|ed)?\s*to\s*change\b",
        r"\bcux\s*want(s|ed)?\s*to\s*change\b", r"\bchange\s*(the\s*)?order\b",
        r"\bcux\s*want(s|ed)?\s*(large|medium|small|personal)\b",
        r"\bcustomer\s*want(s|ed)?\s*(large|medium|small|personal)\b",
        r"\border\s*replaced\b", r"\breplace\s*to\b", r"\breplaced\s*delivery\b",
        r"\bthis\s*order\s*was\s*placed\s*yesterday\b"
    ],
    
    # OUT OF STOCK
    "out of stock": [
        r"\bout\s*of\s*stock\b", r"\bnot\s*(available|have)\b"
    ],
    
    # RIDER ISSUE
    "rider issue": [
        r"\brider\s*(mistake|mistakenly|issue)?\b", r"\brider'?s?\s*issue\b",
        r"\briderr?s?issue\b", r"\bdelivery\s*(boy|guy)\b",
        r"\bat\s*door\b", r"\bnot\s*at\s*home\b",
        r"\briders?\s*(not\s*)?(assigned|assinged|assined)\b",
        r"\brider\s*arrived\b", r"\briderarrived\b"
    ],
    
    # SYSTEM ISSUE
    "system issue": [
        r"\bsystem\s*(error|issue)\b", r"\bsystem\s*show\b"
    ],
    
    # ORDER CANCELLED BY AGGREGATOR - Uber, PickMe, etc.
    "order cancelled by aggregator": [
        r"\buber\b", r"\bpick\s*me\b", r"\bpickme\b", r"\baggregator\b",
        r"\bcancelled?\s*by\s*(uber|pick\s*me)\b", r"\border\s*cancel(led)?\s*by\b"
    ],
    
    # PRODUCT ISSUE
    "product issue or complain": [
        r"\bproduct\s*issue\b", r"\bcomplain\b", r"\bdissatisfy\b", r"\bwrong\s*pizza\b"
    ],
    
    # CUSTOMER RELATED ISSUE - customer not available, didn't come
    "cus.related issue": [
        r"\bcustomer\s*(is\s*)?(not\s*)?(available|availble)\b", 
        r"\bcustomer\s*did(n'?t)?\s*come\b",
        r"\bcustomer\s*left\b", r"\bcustomer\s*visit\b",
        r"\bday\s*end\b", r"\boutlet\s*closed\b", r"\bpower\s*cut\b",
        r"\boven\s*breakdown\b", r"\bsecurity\s*department\b"
    ]
}

def apply_keyword_rules(text):
    """
    Apply rule-based classification using keyword matching.
    Returns category if a rule matches, otherwise None.
    """
    if not text or pd.isna(text):
        return None
    
    text_lower = str(text).lower()
    
    # Priority order for rules (most specific first)
    priority_order = [
        "testing",
        "Customer denied the order",
        "double punch",
        "order cancelled by aggregator",
        "Cashier mistake",
        "Call Center mistake",
        "payment issue",
        "promotion",
        "grid issue",
        "out of stock",
        "Order delay",
        "phone",
        "system issue",
        "rider issue",
        "order type change",
        "location",
        "Customer Cancel order",
        "cus. Change the order",
        "product issue or complain",
        "cus.related issue"
    ]
    
    for category in priority_order:
        if category in KEYWORD_RULES:
            for pattern in KEYWORD_RULES[category]:
                if re.search(pattern, text_lower):
                    return category
    
    return None

def classify_batch(text_list):
    """
    AI-based classification for texts that couldn't be classified by rules.
    Uses detailed prompt with examples for better accuracy.
    """
    prompt = f"""You are an expert data classifier for a Pizza Hut restaurant chain analyzing void order reasons.

TASK: Classify each customer log into EXACTLY ONE category from this list:
{json.dumps(CATEGORIES)}

CLASSIFICATION RULES (apply in order):

1. "testing" - Test orders from IT, product testing
   Examples: "test order", "TEST ORDER FROM IT", "product testing"

2. "promotion" - LSM offers, % discounts, meal deals, flash offers
   Examples: "customer want LSM offer", "50% flash offer", "HSBC 30% discount", "meal deal", "don't cook promotion"

3. "payment issue" - Card/cash/payment problems
   Examples: "credit card not working", "card isn't working", "visa", "petty cash", "online payment issue"

4. "Cashier mistake" - Cashier punched wrong order, mistakenly closed
   Examples: "cashier mistakenly punch", "wrongly punch", "cashier error"

5. "Call Center mistake" - CSR/sales center errors
   Examples: "CSR error", "sale center mistake", "call center asked wrong"

6. "Customer denied the order" - Customer refused/rejected
   Examples: "customer denied", "customer refused", "customer rejected the order"

7. "Customer Cancel order" - Customer requested cancellation
   Examples: "customer want to cancel", "please cancel", "customer cancelled"

8. "double punch" - Order placed twice/duplicate
   Examples: "ordered twice", "same order 2 times", "two orders placed"

9. "grid issue" - Delivery grid/coverage problems
   Examples: "out of grid", "grid issue"

10. "location" - Wrong address, different outlet, wrong location
    Examples: "wrong address", "deliver from X outlet", "transfer to Y", "going from Z"

11. "phone" - Phone not working/answering, wrong number
    Examples: "phone not answer", "wrong phone number", "customer not responding"

12. "Order delay" - Late delivery, delay issues
    Examples: "order delay", "late issue", "couldn't deliver within promise time"

13. "order type change" - Changing from dine-in to delivery, etc.
    Examples: "change to delivery", "want dine in", "change to take away"

14. "cus. Change the order" - Customer changing items/time/order details
    Examples: "customer change the order", "change time", "want different pizza"

15. "out of stock" - Items not available
    Examples: "out of stock", "not available"

16. "rider issue" - Delivery rider problems
    Examples: "rider mistake", "rider mistakenly clicked"

17. "system issue" - System/technical errors
    Examples: "system error", "system issue"

18. "order cancelled by aggregator" - Uber/PickMe cancellation
    Examples: "uber cancelled", "cancelled by PickMe"

19. "product issue or complain" - Quality/product complaints
    Examples: "product issue", "customer complain", "dissatisfied"

20. "cus.related issue" - Other customer-specific issues
    Examples: "customer not available", "customer didn't come"

21. "other" - ONLY if nothing else fits

22. "order without reason/ remark" - No meaningful reason given

INPUT DATA TO CLASSIFY:
{json.dumps(text_list, indent=2)}

OUTPUT: Return a JSON object with key "predictions" containing a list of category strings.
Each prediction must be EXACTLY one of the categories listed above.
"""

    try:
        completion = client.chat.completions.create(
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
        
        # Validate predictions are in allowed categories
        validated = []
        for pred in predictions:
            if pred in CATEGORIES:
                validated.append(pred)
            else:
                # Try to find closest match
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
        print(f"API Error: {e}")
        return ["ERROR"] * len(text_list)

def extract_new_bill_id(text):
    """
    Extracts new bill numbers from text using multiple regex patterns.
    Handles formats like: Y22196, P-69112, HJ 0042, L27016, G81216, etc.
    """
    if not text or pd.isna(text): 
        return None

    clean_text = str(text).upper()
    
    # Pattern 1: Explicit "new bill" mentions
    # Matches: NEW BILL NO Y22196, NBN L27169, new bill number M45055
    patterns = [
        r'(?:NEW\s*BILL?\s*(?:NO|NUMBER|NOMBER|NUBBER)?[:\s-]*|NBN[:\s-]*|N\.?B\.?N[:\s-]*)([A-Z]{1,2}[\s-]?\d{4,7})',
        r'(?:NEW\s*(?:ORDER|DOCKET|DKT|DOC|TRANX)\s*(?:NO|NUMBER)?[:\s-]*)([A-Z]{0,2}[\s-]?\d{3,7})',
        r'(?:ORDER\s*(?:NO|NUMBER)?[:\s-]*)(\d{2,3})(?:\s|$|,)',  # Just order numbers like "order no 18"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean_text)
        if match:
            result = match.group(1).replace(" ", "").replace("-", "")
            # Validate it's a reasonable bill number
            if len(result) >= 2:
                return result
    
    # Pattern 2: General bill ID pattern (fallback)
    # Matches standalone bill IDs like Y22196, HJ0042
    match = re.search(r'\b([A-Z]{1,2}\d{4,7})\b', clean_text)
    if match:
        return match.group(1)
        
    return None


def post_process_category(text, ai_category):
    """
    Post-process AI predictions with additional validation.
    Corrects common misclassifications.
    """
    if not text or pd.isna(text):
        return ai_category
    
    text_lower = str(text).lower()
    
    # If AI said "other" but we can find a better match
    if ai_category == "other":
        rule_category = apply_keyword_rules(text)
        if rule_category:
            return rule_category
    
    # Specific corrections based on common patterns
    
    # If mentions new bill number and "change", it's usually customer change
    if re.search(r'new\s*(bill|order|dkt)', text_lower) and re.search(r'change|want', text_lower):
        if ai_category in ["other", "cus.related issue"]:
            return "cus. Change the order"
    
    # If mentions outlet transfer or different outlet
    if re.search(r'(from|to)\s+\w+\s*(outlet|branch)', text_lower):
        return "location"
    
    return ai_category

def main():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print("File not found.")
        return

    order_col_name = 'Order No'
    
    df['Temp_Order_ID'] = df[order_col_name].ffill()

    def combine_text(x):
        return " ".join(set([str(s).strip() for s in x if pd.notna(s) and str(s).strip() != '']))

    grouped = df.groupby('Temp_Order_ID')[['Reason', 'Remark']].agg(combine_text)
    grouped['AI_Input'] = (grouped['Reason'] + " " + grouped['Remark']).str.strip()
    
    print("Extracting New Bill Numbers using Regex...")
    grouped['Extracted_Bill_No'] = grouped['AI_Input'].apply(extract_new_bill_id)
    
    bill_number_map = grouped['Extracted_Bill_No'].to_dict()

    orders_with_text = grouped[grouped['AI_Input'].str.len() > 1].copy()
    orders_empty = grouped[grouped['AI_Input'].str.len() <= 1].index.tolist()
    
    print(f"Total Orders: {len(grouped)}")
    print(f"Orders to Classify: {len(orders_with_text)}")
    print(f"Empty Orders: {len(orders_empty)}")

    # Step 1: Apply rule-based classification first
    print("\nStep 1: Applying rule-based classification...")
    orders_with_text['Rule_Category'] = orders_with_text['AI_Input'].apply(apply_keyword_rules)
    
    rule_classified = orders_with_text[orders_with_text['Rule_Category'].notna()]
    needs_ai = orders_with_text[orders_with_text['Rule_Category'].isna()]
    
    print(f"  - Rule-based classified: {len(rule_classified)} orders")
    print(f"  - Needs AI classification: {len(needs_ai)} orders")
    
    # Step 2: Use AI only for orders that couldn't be classified by rules
    category_map = {}
    
    # Add rule-classified orders to map
    for order_id, row in rule_classified.iterrows():
        category_map[order_id] = row['Rule_Category']
    
    # AI classification for remaining orders
    if len(needs_ai) > 0:
        print("\nStep 2: AI classification for remaining orders...")
        ids_to_classify = needs_ai.index.tolist()
        texts_to_classify = needs_ai['AI_Input'].tolist()
        ai_predictions = []
        
        for i in tqdm(range(0, len(texts_to_classify), BATCH_SIZE)):
            batch = texts_to_classify[i : i + BATCH_SIZE]
            batch_results = classify_batch(batch)
            
            if len(batch_results) != len(batch):
                diff = len(batch) - len(batch_results)
                if diff > 0: 
                    batch_results += ["other"] * diff
                else: 
                    batch_results = batch_results[:len(batch)]
                
            ai_predictions.extend(batch_results)
            time.sleep(0.5)
        
        # Post-process AI predictions
        print("\nStep 3: Post-processing AI predictions...")
        for idx, order_id in enumerate(ids_to_classify):
            text = texts_to_classify[idx]
            ai_cat = ai_predictions[idx]
            final_cat = post_process_category(text, ai_cat)
            category_map[order_id] = final_cat
    
    # Handle empty orders
    for order_id in orders_empty:
        category_map[order_id] = "order without reason/ remark"

    # ============= APPLY RESULTS =============
    df['Predicted_Category'] = df['Temp_Order_ID'].map(category_map)
    df['Extracted_New_Bill'] = df['Temp_Order_ID'].map(bill_number_map)

    mask_child_rows = df[order_col_name].isna()
    df.loc[mask_child_rows, 'Predicted_Category'] = None
    df.loc[mask_child_rows, 'Extracted_New_Bill'] = None

    del df['Temp_Order_ID']

    # ============= STATISTICS =============
    print("\n" + "="*50)
    print("CLASSIFICATION SUMMARY")
    print("="*50)
    parent_rows = df[df[order_col_name].notna()]
    print(parent_rows['Predicted_Category'].value_counts().to_string())
    print("="*50)

    print("\nApplying highlighting...")

    def highlight_rows(row):
        styles = [''] * len(row)
        
        cat_val = row['Predicted_Category']
        
        if cat_val == "order without reason/ remark":
            idx = row.index.get_loc('Predicted_Category')
            styles[idx] = 'background-color: #FFFF00'  # Yellow
        elif cat_val == "ERROR":
            idx = row.index.get_loc('Predicted_Category')
            styles[idx] = 'background-color: #FF6B6B'  # Red for errors
            
        return styles

    styled_df = df.style.apply(highlight_rows, axis=1)

    print(f"Saving to {OUTPUT_FILE}...")
    styled_df.to_excel(OUTPUT_FILE, index=False)
    print("Done!")

if __name__ == "__main__":
    main()