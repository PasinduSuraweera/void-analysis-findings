"""
AI-Only Void Bills Classification
Uses comprehensive prompt engineering for maximum accuracy with openai/gpt-oss-120b
No rule-based classification - 100% AI powered
"""

import os
import pandas as pd
import json
import time
import re
from tqdm import tqdm
from groq import Groq

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise SystemExit("API_KEY not set. Add API_KEY=your_key to a .env file or set the environment variable.")

INPUT_FILE = "PH_VoidBillListing.xlsx"
OUTPUT_FILE = "categorized_orders_clean.xlsx"
BATCH_SIZE = 10
MODEL_NAME = "openai/gpt-oss-120b"

client = Groq(api_key=API_KEY)

# All valid category names
VALID_CATEGORIES = [
    "testing",
    "promotion", 
    "payment issue",
    "Cashier mistake",
    "Call Center mistake",
    "Customer denied the order",
    "Customer Cancel order",
    "double punch",
    "grid issue",
    "location",
    "phone",
    "Order delay",
    "order type change",
    "cus. Change the order",
    "out of stock",
    "rider issue",
    "system issue",
    "order cancelled by aggregator",
    "product issue or complain",
    "cus.related issue",
    "other",
    "voids without clear reason/ remark"
]


def build_comprehensive_prompt(text_list):
    """
    Build the most comprehensive prompt with ALL details for maximum accuracy.
    This is a 100% AI-only classification approach.
    """
    
    prompt = f'''You are an expert classifier for Pizza Hut Sri Lanka void order reasons. Your task is to analyze WHY each order was voided/cancelled based on staff notes.

═══════════════════════════════════════════════════════════════════════════════
                              CRITICAL CONTEXT
═══════════════════════════════════════════════════════════════════════════════
- These are internal staff notes (not customer-facing)
- Written in "Singlish" (Sri Lankan English) with many typos and abbreviations
- Staff are quickly typing reasons, so expect informal language
- Your job: identify the PRIMARY reason the order was voided

═══════════════════════════════════════════════════════════════════════════════
                         ABBREVIATIONS & SLANG DICTIONARY
═══════════════════════════════════════════════════════════════════════════════
CUSTOMER TERMS:
  cux, cx, cus, cu = customer
  cux denied = customer denied
  cx cancel = customer cancel

ORDER TERMS:
  dkt = docket (order ticket)
  NBN = new bill number
  odar, oder, ordewr = order
  T/W, t/w = take away

STAFF/LOCATION:
  CSR = Customer Service Representative (Call Center)
  sale center, sales centre = Call Center
  IT team, from IT, from preshan = Testing team

COMMON MISSPELLINGS:
  mistakly, mistakely, mistacly = mistakenly
  dubble, doubble = double  
  availble, availabel = available
  cansel, cancell, cancal = cancel
  gride = grid
  deley = delay
  cashiar, cashiyar = cashier
  assinged, assined = assigned
  respons, responsd = respond
  senter, centar = center
  wont = want
  didnt, didn = didn't
  infomed = informed
  acording = according

═══════════════════════════════════════════════════════════════════════════════
                              22 CATEGORIES
═══════════════════════════════════════════════════════════════════════════════

1. "testing"
   ├── WHAT: Test orders created by IT/tech team for system testing
   ├── KEYWORDS: test, testing, test order, IT team, from IT, from preshan, product testing
   ├── EXAMPLES:
   │   ✓ "test order"
   │   ✓ "TEST ORDER FROM IT"
   │   ✓ "testing from preshan"
   │   ✓ "IT team check the system"
   │   ✓ "product testing"
   └── NOT THIS: "customer taste test" (not system testing)

2. "promotion"
   ├── WHAT: Order voided due to promotional offer issues, discounts, campaigns
   ├── KEYWORDS: LSM, promo, offer, discount, %, flash, HSBC, meal deal, cyber saving, don't cook
   ├── EXAMPLES:
   │   ✓ "customer want LSM offer"
   │   ✓ "50% flash offer"
   │   ✓ "HSBC 30% discount"
   │   ✓ "have a 15% discount"
   │   ✓ "don't cook wednesday offer"
   │   ✓ "cyber saving promo"
   │   ✓ "customer asking 20% off"
   └── NOT THIS: general price complaints without promo mention

3. "payment issue"
   ├── WHAT: Problems with payment method - card failed, no cash, machine broken
   ├── KEYWORDS: credit card, card not work, visa, payment, cash, machine, HNB, online payment
   ├── EXAMPLES:
   │   ✓ "credit card not working"
   │   ✓ "card isn't working"
   │   ✓ "don't have enough money"
   │   ✓ "payment machine issue"
   │   ✓ "petty cash"
   │   ✓ "online payment failed"
   │   ✓ "HNB card declined"
   └── NOT THIS: general order issues

4. "Cashier mistake"
   ├── WHAT: Errors made by in-store cashier/staff - wrong punch, mistakenly closed
   ├── KEYWORDS: cashier, wrongly punch, mistakenly punch/close/add/collect, wrong order, dispatcher
   ├── EXAMPLES:
   │   ✓ "cashier mistakenly punch"
   │   ✓ "wrongly punched the order"
   │   ✓ "wrong order entered"
   │   ✓ "dispatcher mistakenly collected"
   │   ✓ "mistakenly close the bill"
   │   ✓ "didn't close properly"
   │   ✓ "miss communication"
   │   ✓ "cashiar wrong order" (typo for cashier)
   └── NOT THIS: Call Center/CSR mistakes (use "Call Center mistake")

5. "Call Center mistake"
   ├── WHAT: Errors made by call center/CSR/sales center staff (remote staff)
   ├── KEYWORDS: CSR, sale center, call center, sales centre, informed by outlet
   ├── EXAMPLES:
   │   ✓ "CSR error"
   │   ✓ "sale center mistake"
   │   ✓ "call center have wrongly placed"
   │   ✓ "sale center mistacly punch" (typo for mistakenly)
   │   ✓ "according to call center"
   │   ✓ "informed by outlet to cancel"
   └── NOT THIS: In-store cashier errors (use "Cashier mistake")

6. "Customer denied the order"
   ├── WHAT: Customer claims they NEVER placed the order / refuses to accept
   ├── KEYWORDS: denied, refuse, reject, didn't place order, not ordered
   ├── EXAMPLES:
   │   ✓ "customer denied the order"
   │   ✓ "cux denied"
   │   ✓ "he didnt place any order"
   │   ✓ "customer told rider didn't place order"
   │   ✓ "denied the order"
   │   ✓ "customer refused to accept"
   │   ✓ "said he never ordered"
   └── KEY DIFFERENCE: Customer says "I never ordered this" (DENIAL)
       vs "I want to cancel" (use "Customer Cancel order")

7. "Customer Cancel order"
   ├── WHAT: Customer placed order but now explicitly wants to CANCEL it
   ├── KEYWORDS: customer cancel, want to cancel, please cancel, cux cancel, cx cancel
   ├── EXAMPLES:
   │   ✓ "customer want to cancel"
   │   ✓ "please cancel this order"
   │   ✓ "cx want cancel"
   │   ✓ "cux cancel the order"
   │   ✓ "customer cancelled"
   └── KEY DIFFERENCE: Customer says "cancel my order" (CANCEL)
       vs "I never placed this" (use "Customer denied the order")

8. "double punch"
   ├── WHAT: Same order was entered/punched twice - duplicate order
   ├── KEYWORDS: twice, double, 2 times, same order, duplicate, dubble
   ├── EXAMPLES:
   │   ✓ "ordered twice"
   │   ✓ "double punch"
   │   ✓ "same order 2 times"
   │   ✓ "dubble punch" (typo)
   │   ✓ "two orders were placed same"
   │   ✓ "duplicate order"
   └── NOT THIS: two different orders

9. "grid issue"
   ├── WHAT: Delivery location is outside outlet's delivery coverage area
   ├── KEYWORDS: grid, out of grid, coverage, gride
   ├── EXAMPLES:
   │   ✓ "out of grid"
   │   ✓ "grid issue"
   │   ✓ "gride issue" (typo)
   │   ✓ "location not in our grid"
   └── NOT THIS: wrong address (use "location")

10. "location"
    ├── WHAT: Wrong address, different outlet, order transfer between outlets
    ├── KEYWORDS: wrong address, different outlet, transfer, wrong location, from X outlet, deliver from
    ├── EXAMPLES:
    │   ✓ "wrong address"
    │   ✓ "deliver from koswattha outlet"
    │   ✓ "transfer to panadura"
    │   ✓ "different city with different outlet"
    │   ✓ "customer in different location"
    │   ✓ "sent from dehiwala to panadura"
    │   ✓ "outlet order" (transfer between outlets)
    └── NOT THIS: out of grid (use "grid issue")

11. "phone"
    ├── WHAT: Cannot contact customer - phone not answering, wrong number
    ├── KEYWORDS: phone not answer, wrong number, not responding, can't contact, no answer, voice mail
    ├── EXAMPLES:
    │   ✓ "phone not answering"
    │   ✓ "wrong phone number"
    │   ✓ "customer not responding call"
    │   ✓ "did not answer phone"
    │   ✓ "given number is not working"
    │   ✓ "called customer 3 times no answer"
    │   ✓ "voice mail"
    │   ✓ "cx no answering"
    └── NOT THIS: customer not at location (use "cus.related issue")

12. "Order delay"
    ├── WHAT: Order is delayed, late delivery, customer can't wait
    ├── KEYWORDS: delay, late, can't wait, promise time, heavy rain
    ├── EXAMPLES:
    │   ✓ "order delay"
    │   ✓ "late issue"
    │   ✓ "customer can't wait anymore"
    │   ✓ "heavy rain delay"
    │   ✓ "promise time exceeded"
    │   ✓ "too late"
    └── NOT THIS: general cancellation

13. "order type change"
    ├── WHAT: Customer wants to change order TYPE (dine-in ↔ delivery ↔ takeaway)
    ├── KEYWORDS: change to delivery, change to take away, want delivery, want dine in
    ├── EXAMPLES:
    │   ✓ "change to delivery"
    │   ✓ "want to change to take away"
    │   ✓ "pick up for delivery"
    │   ✓ "order type change"
    │   ✓ "customer want T/W instead of delivery"
    └── NOT THIS: changing items/size (use "cus. Change the order")

14. "cus. Change the order"
    ├── WHAT: Customer wants to modify order - change items, size, time, details
    ├── KEYWORDS: change the order, change time, want different, customer change, replace
    ├── EXAMPLES:
    │   ✓ "customer change the order"
    │   ✓ "change time to 7pm"
    │   ✓ "customer wants large instead of medium"
    │   ✓ "order replaced"
    │   ✓ "customer mistakenly placed wrong order and wants to change"
    │   ✓ "cux wants personal instead of regular"
    └── NOT THIS: changing order type (use "order type change")

15. "out of stock"
    ├── WHAT: Product/item is not available - stock issue
    ├── KEYWORDS: out of stock, OOS, not available (PRODUCT), stock out, item not available
    ├── EXAMPLES:
    │   ✓ "out of stock"
    │   ✓ "OOS"
    │   ✓ "pizza not available"
    │   ✓ "coke not available"
    │   ✓ "some items are not available"
    │   ✓ "topping not available"
    └── CRITICAL: "customer not available" is NOT this! (use "cus.related issue")

16. "rider issue"
    ├── WHAT: Problem WITH the delivery rider - not assigned, not arrived, rider made mistake
    ├── KEYWORDS: rider issue, rider not assigned, no rider, rider mistake
    ├── EXAMPLES:
    │   ✓ "rider not assigned"
    │   ✓ "no rider arrived"
    │   ✓ "rider issue"
    │   ✓ "rider mistakenly clicked"
    │   ✓ "riders not assinged" (typo)
    └── NOT THIS: "customer told rider..." (that's about customer, not rider)

17. "system issue"
    ├── WHAT: Technical/system problems - system error, app issues
    ├── KEYWORDS: system error, system issue, rider app, IT team busy
    ├── EXAMPLES:
    │   ✓ "system error"
    │   ✓ "system issue"
    │   ✓ "can not delivered in rider app"
    │   ✓ "IT team is busy"
    │   ✓ "app not working"
    └── NOT THIS: testing orders (use "testing")

18. "order cancelled by aggregator"
    ├── WHAT: Order cancelled by food delivery aggregator (Uber, PickMe)
    ├── KEYWORDS: Uber, PickMe, aggregator, cancelled by Uber/PickMe
    ├── EXAMPLES:
    │   ✓ "uber cancelled"
    │   ✓ "cancelled by PickMe"
    │   ✓ "aggregator cancelled"
    │   ✓ "uber order cancelled"
    └── NOT THIS: customer cancelled (use "Customer Cancel order")

19. "product issue or complain"
    ├── WHAT: Customer complaint about product quality
    ├── KEYWORDS: product issue, complain, dissatisfy, wrong pizza delivered
    ├── EXAMPLES:
    │   ✓ "product issue"
    │   ✓ "customer complain"
    │   ✓ "dissatisfied with quality"
    │   ✓ "wrong pizza delivered"
    │   ✓ "cold pizza"
    └── NOT THIS: out of stock

20. "cus.related issue"
    ├── WHAT: Customer availability issues - not at location, didn't show up, left
    ├── KEYWORDS: customer not available, customer left, not at home, didn't come, not showed up
    ├── EXAMPLES:
    │   ✓ "customer not available"
    │   ✓ "cx not available at location"
    │   ✓ "customer left the location"
    │   ✓ "customer not at home"
    │   ✓ "customer didn't come to outlet"
    │   ✓ "customer wasn't available"
    │   ✓ "called several times customer not available"
    └── CRITICAL DIFFERENCE:
        - "customer not available" → "cus.related issue" (physical availability)
        - "phone not answer" → "phone" (contact issue)
        - "product not available" → "out of stock" (stock issue)

21. "other"
    ├── WHAT: ONLY when absolutely nothing else fits
    ├── USAGE: Last resort - try all other categories first
    └── EXAMPLES:
        ✓ Truly unique situations not covered above

22. "voids without clear reason/ remark"
    ├── WHAT: Text exists but does NOT explain WHY order was voided
    ├── KEYWORDS: just new bill number, just order description, no actual reason
    ├── EXAMPLES:
    │   ✓ "new order no 116"
    │   ✓ "customer place takeaway order"
    │   ✓ "veg melt"
    │   ✓ "NBN Y22196"
    │   ✓ "new bill M45055"
    │   ✓ "dkt 18"
    │   ✓ "supreme pizza"
    │   ✓ "2 large pizzas"
    └── USE WHEN: Text describes WHAT was ordered, not WHY it was voided

═══════════════════════════════════════════════════════════════════════════════
                           DECISION PRIORITY RULES
═══════════════════════════════════════════════════════════════════════════════

When multiple categories could apply, use this priority:

1. "testing" - HIGHEST PRIORITY (any mention of test/IT team)
2. "Customer denied the order" - Customer claims never ordered
3. "double punch" - Duplicate order
4. "order cancelled by aggregator" - Uber/PickMe cancelled
5. "Cashier mistake" - In-store staff error
6. "Call Center mistake" - CSR/sales center error  
7. "payment issue" - Payment problems
8. "promotion" - Promo/discount issues
9. "grid issue" - Out of coverage area
10. "rider issue" - Rider problems
11. "phone" - Can't contact customer
12. "cus.related issue" - Customer availability
13. "out of stock" - Product unavailable
14. "Order delay" - Late/delayed
15. "system issue" - Tech problems
16. "order type change" - Changing type
17. "location" - Address/outlet issues
18. "Customer Cancel order" - Customer wants cancel
19. "cus. Change the order" - Customer modifying
20. "product issue or complain" - Quality complaints
21. "voids without clear reason/ remark" - No clear reason
22. "other" - LAST RESORT ONLY

═══════════════════════════════════════════════════════════════════════════════
                              EXAMPLES TO LEARN FROM
═══════════════════════════════════════════════════════════════════════════════

Example 1: "test order from IT"
→ Category: "testing"
→ Why: Contains "test order" and "IT" - clearly a test

Example 2: "customer denied the order. he didnt place any order"
→ Category: "Customer denied the order"
→ Why: Customer claims they never placed the order

Example 3: "new bill number Y22196"
→ Category: "voids without clear reason/ remark"
→ Why: Only mentions new bill number, no reason given

Example 4: "customer not available at location called several times"
→ Category: "cus.related issue"
→ Why: Customer availability (not at location) is primary

Example 5: "sale center mistacly punch the order"
→ Category: "Call Center mistake"
→ Why: "sale center" made the error (despite typo)

Example 6: "rider not assigned for this order"
→ Category: "rider issue"
→ Why: Problem is with rider assignment

Example 7: "customer told rider didn't place order"
→ Category: "Customer denied the order"
→ Why: Customer denying they placed it

Example 8: "pizza coke not available"
→ Category: "out of stock"
→ Why: Products not available

Example 9: "deliver from koswattha outlet to panadura"
→ Category: "location"
→ Why: Order transfer between outlets

Example 10: "dubble punch same order"
→ Category: "double punch"
→ Why: "dubble" = double, order punched twice

Example 11: "phone not answering tried 5 times"
→ Category: "phone"
→ Why: Contact issue - can't reach customer

Example 12: "customer wants to cancel the order"
→ Category: "Customer Cancel order"
→ Why: Customer requesting cancellation

Example 13: "cashier wrongly punched the items"
→ Category: "Cashier mistake"
→ Why: Cashier made an error

Example 14: "customer want 50% flash offer"
→ Category: "promotion"
→ Why: Promotional offer issue

Example 15: "credit card declined"
→ Category: "payment issue"
→ Why: Payment method problem

Example 16: "uber cancelled the order"
→ Category: "order cancelled by aggregator"
→ Why: Uber (aggregator) cancelled

Example 17: "order delay heavy rain"
→ Category: "Order delay"
→ Why: Delayed due to rain

Example 18: "change to delivery"
→ Category: "order type change"
→ Why: Changing order type

Example 19: "customer wants large instead of medium"
→ Category: "cus. Change the order"
→ Why: Changing item size

Example 20: "out of grid location"
→ Category: "grid issue"
→ Why: Outside delivery coverage

═══════════════════════════════════════════════════════════════════════════════
                              YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Classify each of these void order reasons:
{json.dumps(text_list, indent=2)}

RESPOND WITH ONLY A JSON OBJECT:
{{"predictions": ["category1", "category2", ...]}}

Each prediction MUST be one of these exact strings:
{json.dumps(VALID_CATEGORIES, indent=2)}

CRITICAL REMINDERS:
1. One category per input text
2. Use exact category names (case-sensitive)
3. "customer not available" → "cus.related issue" (NOT "out of stock")
4. Just a bill number/order description with no reason → "voids without clear reason/ remark"
5. Only use "other" when nothing else fits at all
'''

    return prompt


def classify_batch_ai(text_list, retry_count=3):
    """AI-only classification with comprehensive prompting and retry logic."""
    
    prompt = build_comprehensive_prompt(text_list)
    
    for attempt in range(retry_count):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise JSON classification API for Pizza Hut Sri Lanka void orders. Output ONLY valid JSON with a 'predictions' array. Each prediction must be exactly one of the valid category names."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"},
                timeout=60
            )
            
            response_text = completion.choices[0].message.content
            data = json.loads(response_text)
            predictions = data.get("predictions", [])
            
            # Validate and normalize predictions
            validated = []
            for pred in predictions:
                pred_clean = str(pred).strip()
                if pred_clean in VALID_CATEGORIES:
                    validated.append(pred_clean)
                else:
                    # Try case-insensitive match
                    matched = False
                    for cat in VALID_CATEGORIES:
                        if cat.lower() == pred_clean.lower():
                            validated.append(cat)
                            matched = True
                            break
                    if not matched:
                        # Try partial match
                        for cat in VALID_CATEGORIES:
                            if pred_clean.lower() in cat.lower() or cat.lower() in pred_clean.lower():
                                validated.append(cat)
                                matched = True
                                break
                    if not matched:
                        validated.append("other")
            
            # Ensure we have right number of predictions
            while len(validated) < len(text_list):
                validated.append("other")
            validated = validated[:len(text_list)]
            
            return validated
            
        except json.JSONDecodeError as e:
            print(f"\n  JSON Parse Error (attempt {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(2)
                continue
            return ["other"] * len(text_list)
        except Exception as e:
            print(f"\n  API Error (attempt {attempt+1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(3)
                continue
            return ["ERROR"] * len(text_list)


def extract_new_bill_id(text):
    """Extract new bill numbers from text."""
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


def main():
    print("="*60)
    print("AI-ONLY VOID BILLS CLASSIFICATION")
    print("100% AI powered with comprehensive prompts")
    print("="*60)
    
    print(f"\nReading {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"File not found: {INPUT_FILE}")
        return

    order_col_name = 'Order No'
    df['Temp_Order_ID'] = df[order_col_name].ffill()

    def combine_text(x):
        return " ".join(set([str(s).strip() for s in x if pd.notna(s) and str(s).strip() != '']))

    grouped = df.groupby('Temp_Order_ID')[['Reason', 'Remark']].agg(combine_text)
    grouped['AI_Input'] = (grouped['Reason'] + " " + grouped['Remark']).str.strip()
    
    print("Extracting New Bill Numbers...")
    grouped['Extracted_Bill_No'] = grouped['AI_Input'].apply(extract_new_bill_id)
    bill_number_map = grouped['Extracted_Bill_No'].to_dict()

    orders_with_text = grouped[grouped['AI_Input'].str.len() > 1].copy()
    orders_empty = grouped[grouped['AI_Input'].str.len() <= 1].index.tolist()
    
    print(f"\nTotal Orders: {len(grouped)}")
    print(f"Orders to Classify (AI): {len(orders_with_text)}")
    print(f"Empty Orders: {len(orders_empty)}")
    
    category_map = {}
    
    # AI-only classification for ALL orders with text
    if len(orders_with_text) > 0:
        print(f"\n[AI Classification] Processing {len(orders_with_text)} orders...")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Model: {MODEL_NAME}")
        print()
        
        ids_to_classify = orders_with_text.index.tolist()
        texts_to_classify = orders_with_text['AI_Input'].tolist()
        ai_predictions = []
        
        total_batches = (len(texts_to_classify) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in tqdm(range(0, len(texts_to_classify), BATCH_SIZE), 
                      desc="Processing", total=total_batches):
            batch = texts_to_classify[i:i+BATCH_SIZE]
            batch_results = classify_batch_ai(batch)
            
            # Ensure correct length
            while len(batch_results) < len(batch):
                batch_results.append("other")
            batch_results = batch_results[:len(batch)]
            
            ai_predictions.extend(batch_results)
            time.sleep(0.5)  # Rate limiting
        
        # Add AI results to map
        for idx, order_id in enumerate(ids_to_classify):
            if idx < len(ai_predictions):
                category_map[order_id] = ai_predictions[idx]
            else:
                category_map[order_id] = "other"
    
    # Handle empty orders
    for order_id in orders_empty:
        category_map[order_id] = "no reason/remark"

    # Apply results
    print("\nApplying results to dataframe...")
    df['Predicted_Category'] = df['Temp_Order_ID'].map(category_map)
    df['Extracted_New_Bill'] = df['Temp_Order_ID'].map(bill_number_map)

    mask_child_rows = df[order_col_name].isna()
    df.loc[mask_child_rows, 'Predicted_Category'] = None
    df.loc[mask_child_rows, 'Extracted_New_Bill'] = None

    del df['Temp_Order_ID']

    # Statistics
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    parent_rows = df[df[order_col_name].notna()]
    results = parent_rows['Predicted_Category'].value_counts()
    print(results.to_string())
    print("="*60)
    print(f"Total: {len(parent_rows)} orders classified")

    # Apply highlighting
    def highlight_rows(row):
        styles = [''] * len(row)
        cat_val = row['Predicted_Category']
        if cat_val == "no reason/remark":
            idx = row.index.get_loc('Predicted_Category')
            styles[idx] = 'background-color: #FFFF00'
        elif cat_val == "voids without clear reason/ remark":
            idx = row.index.get_loc('Predicted_Category')
            styles[idx] = 'background-color: #FFD700'
        elif cat_val == "ERROR":
            idx = row.index.get_loc('Predicted_Category')
            styles[idx] = 'background-color: #FF6B6B'
        return styles

    styled_df = df.style.apply(highlight_rows, axis=1)

    print(f"\nSaving to {OUTPUT_FILE}...")
    styled_df.to_excel(OUTPUT_FILE, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
