"""
=============================================================
  Synthetic Credit Card Fraud Detection Dataset — v2
  REALISTIC VERSION — Hard to detect patterns
  Rows: 400,000 | Fraud Rate: ~8.5%
  Output: fraud_v2.csv

  Key design principles:
  ─────────────────────────────────────────────
  1. NO single feature separates fraud from legit
  2. Overlapping distributions everywhere
  3. Real-world noise injected throughout
  4. Multiple fraud types with different profiles
  5. Fraud rings — shared devices across users
  6. Legitimate users who LOOK suspicious
  7. Fraudsters who LOOK legitimate
  8. Label noise — real fraud labels are never 100% clean
=============================================================
Install:
    pip install pandas numpy faker tqdm

Run:
    python generate_fraud_v2.py

Expected output size: ~140MB
Expected runtime:     ~6–9 min
=============================================================
"""

import pandas as pd
import numpy as np
from faker import Faker
from tqdm import tqdm
import random
import hashlib
import uuid
import warnings
warnings.filterwarnings("ignore")

fake   = Faker("en_IN")
rng    = np.random.default_rng(57)          # numpy rng with seed
random.seed(57)

N          = 400_000
FRAUD_RATE = 0.085
CHUNK_SIZE = 50_000

# ─────────────────────────────────────────────────────────────
# FRAUD TYPE DEFINITIONS
# Different fraud types have very different behavioural profiles
# This prevents any single pattern from being THE fraud signal
# ─────────────────────────────────────────────────────────────
FRAUD_TYPES = {
    # Stolen card — quick high-value purchases before victim notices
    "stolen_card":         0.28,
    # Account takeover — fraudster logs in slowly, behaves normally
    "account_takeover":    0.22,
    # Fraud ring — organised crime, multiple accounts one device
    "fraud_ring":          0.18,
    # Synthetic identity — fake KYC, looks clean initially
    "synthetic_identity":  0.15,
    # Insider/merchant fraud — legit looking transactions
    "merchant_collusion":  0.10,
    # Card-not-present — online fraud with stolen card details
    "card_not_present":    0.07,
}

# Pre-generate shared device pool for fraud rings
# (same device fingerprint appears across different user IDs)
RING_DEVICES = [
    hashlib.sha256(f"ring_device_{i}".encode()).hexdigest()[:32]
    for i in range(150)
]
RING_IPS = [
    f"{random.choice([45,103,185,220])}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    for _ in range(150)
]

# ─────────────────────────────────────────────────────────────
# LOOKUP TABLES
# ─────────────────────────────────────────────────────────────
BANKS = [
    "SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra",
    "Yes Bank", "Punjab National Bank", "Bank of Baroda", "Canara Bank",
    "IndusInd Bank", "Federal Bank", "IDFC First Bank", "AU Small Finance",
    "RBL Bank", "Bandhan Bank", "DBS India", "Standard Chartered India",
    "HSBC India", "Citibank India", "Deutsche Bank India"
]
CARD_NETWORKS  = ["Visa", "Mastercard", "RuPay", "Amex", "Diners Club"]
CARD_TYPES     = ["Credit", "Debit", "Prepaid", "Corporate"]
CARD_BIN_MAP   = {
    "Visa":        ["411111","427034","453978","476148","491177","400000"],
    "Mastercard":  ["512345","523456","540000","556677","530000","545454"],
    "RuPay":       ["607000","608000","607387","608180","606985","607523"],
    "Amex":        ["378282","371449","370000","379764"],
    "Diners Club": ["300000","301000","305000","380000"],
}
DEVICE_OS      = ["Android","iOS","Windows","macOS","Linux","HarmonyOS","KaiOS"]
DEVICE_BRANDS  = ["Samsung","Apple","Xiaomi","Realme","OnePlus","Vivo","Oppo",
                  "Motorola","Nokia","Itel","Tecno","Infinix","Lava"]
BROWSERS       = ["Chrome","Firefox","Safari","Edge","Samsung Internet",
                  "Opera Mini","UC Browser","Brave"]
MERCHANT_CATS  = [
    "Grocery","E-Commerce","Fuel","Restaurants","Travel & Airlines",
    "Electronics","Jewellery","Pharmaceuticals","Utility Bills",
    "Gaming","Crypto Exchange","International Wire","ATM Withdrawal",
    "Luxury Goods","Insurance Premium","Real Estate","Education Fees",
    "Subscription Services","Healthcare","Automobile"
]
CITIES = [
    "Mumbai","Delhi","Bangalore","Hyderabad","Chennai","Kolkata",
    "Pune","Ahmedabad","Jaipur","Lucknow","Surat","Bhopal","Indore",
    "Nagpur","Chandigarh","Coimbatore","Kochi","Patna","Bhubaneswar",
    "Visakhapatnam","Vadodara","Gurgaon","Noida","Thane","Navi Mumbai"
]
STATES = {
    "Mumbai":"Maharashtra","Thane":"Maharashtra","Pune":"Maharashtra",
    "Nagpur":"Maharashtra","Navi Mumbai":"Maharashtra","Delhi":"Delhi",
    "Gurgaon":"Haryana","Noida":"Uttar Pradesh","Lucknow":"Uttar Pradesh",
    "Patna":"Bihar","Bangalore":"Karnataka","Hyderabad":"Telangana",
    "Chennai":"Tamil Nadu","Coimbatore":"Tamil Nadu","Kochi":"Kerala",
    "Kolkata":"West Bengal","Bhubaneswar":"Odisha","Visakhapatnam":"Andhra Pradesh",
    "Ahmedabad":"Gujarat","Surat":"Gujarat","Vadodara":"Gujarat",
    "Jaipur":"Rajasthan","Chandigarh":"Punjab",
    "Bhopal":"Madhya Pradesh","Indore":"Madhya Pradesh",
}
EMPLOYMENT     = ["Salaried","Self-Employed","Business Owner","Freelancer",
                  "Retired","Student","Unemployed","Government Employee"]
EDUCATION      = ["Below 10th","10th Pass","12th Pass","Graduate",
                  "Post-Graduate","Doctorate","Professional Degree"]
KYC_STATUSES   = ["Full KYC","Min KYC","Pending","Failed","Expired"]
TXN_CHANNELS   = ["UPI","Net Banking","POS","ATM","Contactless",
                  "NEFT/RTGS","IMPS","Wallet","QR Code"]
UPI_HANDLES    = ["@okaxis","@oksbi","@okicici","@ybl","@paytm",
                  "@upi","@kotak","@ibl","@rbl","@axl"]
SCREEN_RES     = ["1080x2400","1170x2532","1080x1920","2560x1440",
                  "1366x768","1920x1080","720x1520","1440x3200",
                  "390x844","414x896"]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def wc(options, weights):
    return random.choices(options, weights=weights, k=1)[0]

def noisy(val, noise_pct=0.08):
    """Add small random noise to a value — simulates measurement imperfection"""
    return val * (1 + random.uniform(-noise_pct, noise_pct))

def pick_fraud_type():
    types  = list(FRAUD_TYPES.keys())
    probs  = list(FRAUD_TYPES.values())
    return random.choices(types, weights=probs, k=1)[0]

def gen_device_fp(fraud_type=None, ring_idx=None):
    if fraud_type == "fraud_ring" and ring_idx is not None:
        return RING_DEVICES[ring_idx % len(RING_DEVICES)]
    raw = f"{fake.mac_address()}-{fake.user_agent()}-{random.randint(1000,9999)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]

def gen_digital_sig():
    raw = f"{uuid.uuid4()}-{fake.iban()}"
    return hashlib.sha512(raw.encode()).hexdigest()[:64]

def gen_ip(fraud_type=None, ring_idx=None, vpn=False):
    if fraud_type == "fraud_ring" and ring_idx is not None:
        return RING_IPS[ring_idx % len(RING_IPS)]
    if vpn:
        return f"{random.choice([45,185,220,103])}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    return fake.ipv4_private() if random.random() < 0.3 else fake.ipv4_public()

def gen_card_number(bin_prefix):
    return f"{bin_prefix}{''.join([str(random.randint(0,9)) for _ in range(12)])}"[:16]

# ─────────────────────────────────────────────────────────────
# CORE RECORD BUILDER
# Fraud type drives subtle behavioural differences
# NO feature is a clean separator on its own
# ─────────────────────────────────────────────────────────────

def build_record(is_fraud: bool, ring_idx: int = None) -> dict:

    fraud_type = pick_fraud_type() if is_fraud else None

    # ── DEMOGRAPHICS ─────────────────────────────────────────
    # Fraudsters span ALL age groups — no clean age signal
    age = int(np.clip(rng.normal(34, 13), 18, 74))

    # Fraud slightly higher in some employment types but very noisy
    if is_fraud:
        employment = wc(EMPLOYMENT, [28,22,14,12,7,8,6,3])
    else:
        employment = wc(EMPLOYMENT, [35,18,14,9,7,7,4,6])

    education  = wc(EDUCATION,  [4,9,18,35,24,4,6])
    city       = random.choice(CITIES)
    state      = STATES.get(city, "Unknown")
    gender     = random.choice(["M","F","Other"])

    # Income: fraudsters exist at ALL income levels
    # Synthetic identity fraud often targets middle-income range
    annual_income = int(np.clip(rng.lognormal(12.5, 0.7), 100000, 9000000))

    # PAN / Aadhaar — synthetic identity fraudsters have fake but "verified" docs
    if fraud_type == "synthetic_identity":
        pan_verified   = random.random() < 0.78   # fake but verified
        aadhaar_linked = random.random() < 0.72
    elif is_fraud:
        pan_verified   = random.random() < 0.55
        aadhaar_linked = random.random() < 0.58
    else:
        pan_verified   = random.random() < 0.88
        aadhaar_linked = random.random() < 0.91

    # ── BUREAU / CREDIT SCORES ───────────────────────────────
    # KEY DESIGN DECISION:
    # Means are close, std is wide → heavy overlap between fraud/legit
    # Synthetic identity has GOOD scores (they build credit before striking)
    # Account takeover has GOOD scores (they steal good accounts)
    if fraud_type == "synthetic_identity":
        cibil_score    = int(np.clip(rng.normal(695, 95),  300, 900))
        crif_score     = int(np.clip(rng.normal(680, 98),  300, 900))
        experian_score = int(np.clip(rng.normal(688, 92),  300, 900))
        equifax_score  = int(np.clip(rng.normal(683, 94),  300, 900))
        bearo_score    = int(np.clip(rng.normal(580, 130), 0,   1000))
    elif fraud_type == "account_takeover":
        cibil_score    = int(np.clip(rng.normal(718, 88),  300, 900))
        crif_score     = int(np.clip(rng.normal(705, 90),  300, 900))
        experian_score = int(np.clip(rng.normal(710, 85),  300, 900))
        equifax_score  = int(np.clip(rng.normal(708, 88),  300, 900))
        bearo_score    = int(np.clip(rng.normal(610, 140), 0,   1000))
    elif fraud_type == "merchant_collusion":
        # Merchant collusion — customer is real, merchant is fraudulent
        cibil_score    = int(np.clip(rng.normal(712, 82),  300, 900))
        crif_score     = int(np.clip(rng.normal(700, 85),  300, 900))
        experian_score = int(np.clip(rng.normal(706, 80),  300, 900))
        equifax_score  = int(np.clip(rng.normal(704, 83),  300, 900))
        bearo_score    = int(np.clip(rng.normal(640, 125), 0,   1000))
    elif is_fraud:
        cibil_score    = int(np.clip(rng.normal(672, 105), 300, 900))
        crif_score     = int(np.clip(rng.normal(658, 108), 300, 900))
        experian_score = int(np.clip(rng.normal(665, 100), 300, 900))
        equifax_score  = int(np.clip(rng.normal(661, 103), 300, 900))
        bearo_score    = int(np.clip(rng.normal(545, 145), 0,   1000))
    else:
        cibil_score    = int(np.clip(rng.normal(708, 95),  300, 900))
        crif_score     = int(np.clip(rng.normal(695, 97),  300, 900))
        experian_score = int(np.clip(rng.normal(701, 92),  300, 900))
        equifax_score  = int(np.clip(rng.normal(698, 94),  300, 900))
        bearo_score    = int(np.clip(rng.normal(695, 110), 0,   1000))

    # ── CARD DETAILS ─────────────────────────────────────────
    card_network = wc(CARD_NETWORKS, [38,33,20,6,3])
    card_type    = wc(CARD_TYPES,    [48,38,9,5])
    card_bin     = random.choice(CARD_BIN_MAP[card_network])
    card_number  = gen_card_number(card_bin)
    card_expiry  = f"{random.randint(1,12):02d}/{random.randint(25,31)}"
    issuer_bank  = random.choice(BANKS)
    card_limit   = int(np.clip(rng.lognormal(11.5, 0.58), 10000, 3500000))

    # Stolen card → card age can be anything (they steal old cards too)
    # Account takeover → older accounts (more valuable)
    if fraud_type == "account_takeover":
        card_age_days = int(np.clip(rng.exponential(1200), 180, 3650))
    elif fraud_type == "stolen_card":
        card_age_days = int(np.clip(rng.exponential(700),  1,   3650))
    else:
        card_age_days = int(np.clip(rng.exponential(900 if not is_fraud else 400), 1, 3650))

    # Virtual cards — slightly higher in fraud but legit users also use them
    is_virtual_card = random.random() < (0.22 if is_fraud else 0.14)

    # ── TRANSACTION ──────────────────────────────────────────
    # Merchant category: fraud is spread across ALL categories
    # No one category screams fraud — real world reality
    merchant_cat = wc(MERCHANT_CATS,
        [10,16,7,7,6,7,4,5,7,4,3,3,5,2,3,2,4,3,3,4])

    # Stolen card does skew to high-value categories, but not always
    if fraud_type == "stolen_card" and random.random() < 0.40:
        merchant_cat = wc(
            ["Electronics","Jewellery","Luxury Goods","Travel & Airlines","Crypto Exchange"],
            [30,25,20,15,10]
        )
    # Merchant collusion looks completely normal
    elif fraud_type == "merchant_collusion":
        merchant_cat = wc(
            ["Grocery","Restaurants","E-Commerce","Pharmaceuticals","Utility Bills"],
            [25,20,25,15,15]
        )

    # AMOUNT — This is the trickiest one
    # Fraudsters don't always do huge transactions
    # Many do small amounts to test cards first, then mid-range
    if fraud_type == "stolen_card":
        # Quick high-value before victim notices — but not always huge
        txn_amount = round(np.clip(rng.lognormal(9.2, 1.1), 800, 600000), 2)
    elif fraud_type == "account_takeover":
        # Starts small to test, escalates
        txn_amount = round(np.clip(rng.lognormal(8.5, 1.3), 200, 400000), 2)
    elif fraud_type == "fraud_ring":
        # Systematic mid-range transactions
        txn_amount = round(np.clip(rng.lognormal(8.8, 0.9), 500, 250000), 2)
    elif fraud_type == "card_not_present":
        # Online — tends to be mid-range
        txn_amount = round(np.clip(rng.lognormal(8.3, 1.0), 300, 180000), 2)
    elif fraud_type in ("merchant_collusion","synthetic_identity"):
        # Looks completely normal
        txn_amount = round(np.clip(rng.lognormal(7.9, 1.0), 100, 150000), 2)
    else:
        txn_amount = round(np.clip(rng.lognormal(7.8, 1.0), 50, 200000), 2)

    # Add noise to amounts — real amounts are messy
    txn_amount = round(noisy(txn_amount, 0.05), 2)

    txn_currency = wc(["INR","USD","EUR","GBP","AED","SGD","JPY"],
                       [80,10,4,2,2,1,1])
    # Card-not-present and stolen card slightly more international
    if fraud_type in ("card_not_present","stolen_card") and random.random() < 0.25:
        txn_currency = wc(["USD","EUR","GBP","AED"], [40,30,20,10])
    is_intl_txn  = txn_currency != "INR"

    # Channel — account takeover mimics legitimate user behaviour
    if fraud_type == "account_takeover":
        txn_channel = wc(TXN_CHANNELS, [30,28,15,8,6,5,4,3,1])
    elif fraud_type == "card_not_present":
        txn_channel = wc(TXN_CHANNELS, [25,35,5,5,5,10,8,5,2])
    elif fraud_type == "fraud_ring":
        txn_channel = wc(TXN_CHANNELS, [22,25,18,12,8,6,4,3,2])
    elif is_fraud:
        txn_channel = wc(TXN_CHANNELS, [26,24,16,12,8,6,4,3,1])
    else:
        txn_channel = wc(TXN_CHANNELS, [35,18,18,8,8,4,4,3,2])

    upi_id = (
        f"{fake.user_name()}{random.choice(UPI_HANDLES)}"
        if txn_channel in ("UPI","QR Code") else "NOT_UPI"
    )

    # TIME — fraud doesn't only happen at 2AM anymore
    # Account takeover happens during business hours to avoid suspicion
    if fraud_type == "account_takeover":
        txn_hour = wc(list(range(24)),
            [1,1,1,1,1,2,5,7,8,8,8,7,7,7,7,6,5,5,4,3,2,2,1,1])
    elif fraud_type == "merchant_collusion":
        # During business hours — looks legitimate
        txn_hour = wc(list(range(24)),
            [0,0,0,0,0,1,3,6,8,9,9,8,8,8,8,7,6,5,4,3,2,1,0,0])
    elif is_fraud:
        # Mix of hours — not only late night
        txn_hour = wc(list(range(24)),
            [2,2,3,3,3,2,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,3,2])
    else:
        txn_hour = wc(list(range(24)),
            [1,1,1,1,1,2,4,6,7,7,7,6,7,7,7,7,6,6,5,4,3,2,2,1])

    txn_day_of_week = random.randint(0, 6)
    txn_month       = random.randint(1, 12)
    txn_weekend     = int(txn_day_of_week >= 5)

    merchant_id   = f"MID{random.randint(1000000,9999999)}"
    merchant_city = random.choice(CITIES)

    # City match — account takeover happens in home city often
    if fraud_type == "account_takeover":
        txn_city_match = random.random() < 0.60
    elif fraud_type == "merchant_collusion":
        txn_city_match = random.random() < 0.72
    elif is_fraud:
        txn_city_match = random.random() < 0.42
    else:
        txn_city_match = random.random() < 0.74

    # Split transactions — structuring to avoid detection limits
    # Small amounts, many transactions
    if fraud_type == "fraud_ring":
        is_split_txn = random.random() < 0.38
    elif is_fraud:
        is_split_txn = random.random() < 0.18
    else:
        is_split_txn = random.random() < 0.04
    split_count  = random.randint(2,7) if is_split_txn else 1

    # ── VELOCITY FEATURES ────────────────────────────────────
    # CRITICAL: These were too separated before
    # Now distributions overlap significantly
    # Legit users also sometimes do multiple transactions (travel, shopping sprees)
    # Fraudsters don't always go crazy on velocity

    if fraud_type == "account_takeover":
        # Low velocity — they're being careful
        txn_count_1h  = int(np.clip(rng.poisson(2),  0, 15))
        txn_count_24h = int(np.clip(rng.poisson(6),  0, 40))
        txn_count_7d  = int(np.clip(rng.poisson(22), 0, 100))
        txn_count_30d = int(np.clip(rng.poisson(55), 0, 250))
    elif fraud_type == "fraud_ring":
        # High velocity — systematic
        txn_count_1h  = int(np.clip(rng.poisson(6),  0, 25))
        txn_count_24h = int(np.clip(rng.poisson(18), 0, 70))
        txn_count_7d  = int(np.clip(rng.poisson(48), 0, 180))
        txn_count_30d = int(np.clip(rng.poisson(95), 0, 400))
    elif fraud_type == "stolen_card":
        # Burst pattern — high in 1h and 24h but short window
        txn_count_1h  = int(np.clip(rng.poisson(5),  0, 20))
        txn_count_24h = int(np.clip(rng.poisson(12), 0, 50))
        txn_count_7d  = int(np.clip(rng.poisson(18), 0, 80))
        txn_count_30d = int(np.clip(rng.poisson(28), 0, 120))
    elif is_fraud:
        txn_count_1h  = int(np.clip(rng.poisson(3),  0, 20))
        txn_count_24h = int(np.clip(rng.poisson(9),  0, 55))
        txn_count_7d  = int(np.clip(rng.poisson(28), 0, 130))
        txn_count_30d = int(np.clip(rng.poisson(62), 0, 280))
    else:
        txn_count_1h  = int(np.clip(rng.poisson(1),  0, 12))
        txn_count_24h = int(np.clip(rng.poisson(4),  0, 30))
        txn_count_7d  = int(np.clip(rng.poisson(16), 0, 80))
        txn_count_30d = int(np.clip(rng.poisson(48), 0, 200))

    avg_txn_7d        = round(np.clip(rng.lognormal(7.2 if is_fraud else 7.0, 0.85), 100, 250000), 2)
    txn_vs_avg_ratio  = round(txn_amount / (avg_txn_7d + 1), 4)

    # Failed transactions — also noisy
    failed_txn_24h    = int(np.clip(rng.poisson(1.2 if is_fraud else 0.4), 0, 12))
    diff_merchants_24h= int(np.clip(rng.poisson(4 if is_fraud else 2), 0, 20))
    diff_cities_7d    = int(np.clip(rng.poisson(2 if is_fraud else 1), 0, 10))
    max_txn_30d       = round(np.clip(rng.lognormal(9.0 if is_fraud else 8.6, 0.9), 200, 700000), 2)
    amt_pct_rank      = round(np.clip(rng.beta(1.8 if is_fraud else 3.5, 3), 0.0, 1.0), 4)

    # ── DEVICE / DIGITAL ─────────────────────────────────────
    device_os    = wc(DEVICE_OS, [37,27,17,8,6,4,1])
    device_brand = wc(DEVICE_BRANDS, [22,18,17,11,9,7,5,4,3,1,1,1,1])
    browser      = wc(BROWSERS, [48,9,19,7,7,3,4,3])
    screen_res   = random.choice(SCREEN_RES)

    # New device — real signal but much noisier than before
    # Legitimate users get new phones too
    if fraud_type == "account_takeover":
        is_new_device = random.random() < 0.72   # classic ATO signal
    elif fraud_type in ("fraud_ring","stolen_card"):
        is_new_device = random.random() < 0.45
    elif is_fraud:
        is_new_device = random.random() < 0.30
    else:
        is_new_device = random.random() < 0.16   # legit users also get new phones

    # VPN — much lower probability than before, even for fraud
    if fraud_type in ("card_not_present","fraud_ring"):
        vpn_flag = random.random() < 0.22
    elif is_fraud:
        vpn_flag = random.random() < 0.14
    else:
        vpn_flag = random.random() < 0.06    # some legit users use VPN

    # Emulator — lower signal than before
    emulator_flag = random.random() < (0.12 if is_fraud else 0.02)

    # Fingerprint and IP
    device_fp = gen_device_fp(fraud_type, ring_idx)
    ip_address = gen_ip(fraud_type, ring_idx, vpn=vpn_flag)

    ip_country = "IN" if not vpn_flag else wc(
        ["IN","US","NL","DE","RU","CN","PK","NG","UA","TR"],
        [25,20,12,10,8,7,5,5,4,4]
    )

    location_spoofed  = random.random() < (0.18 if is_fraud else 0.02)
    device_changed_30d= int(np.clip(rng.poisson(1 if is_fraud else 0.2), 0, 6))
    browser_lang      = wc(["en-IN","hi-IN","en-US","zh-CN","ar"],
                            [55,20,12,7,6])

    # ── AUTHENTICATION ────────────────────────────────────────
    # Account takeover is CAREFUL — they pass OTP (they have the phone)
    if fraud_type == "account_takeover":
        otp_attempts  = int(np.clip(rng.poisson(1.2), 1, 5))
        login_attempts= int(np.clip(rng.poisson(1.3), 1, 5))
        biometric_used= random.random() < 0.45
        mpin_used     = random.random() < 0.55
    elif is_fraud:
        otp_attempts  = int(np.clip(rng.poisson(1.8), 1, 8))
        login_attempts= int(np.clip(rng.poisson(2.0), 1, 10))
        biometric_used= random.random() < 0.22
        mpin_used     = random.random() < 0.32
    else:
        otp_attempts  = int(np.clip(rng.poisson(1.1), 1, 5))
        login_attempts= int(np.clip(rng.poisson(1.1), 1, 6))
        biometric_used= random.random() < 0.55
        mpin_used     = random.random() < 0.60

    otp_success         = otp_attempts == 1 if not is_fraud else random.random() < 0.72
    txn_pin_attempts    = int(np.clip(rng.poisson(1.5 if is_fraud else 1.0), 1, 6))
    session_duration    = int(np.clip(
        rng.exponential(120 if is_fraud else 280), 5, 1800
    ))
    # Add noise — some legit sessions are short, some fraud sessions are long
    session_duration    = int(noisy(session_duration, 0.20))
    password_changed_30d= int(random.random() < (0.22 if is_fraud else 0.06))

    # ── ACCOUNT HEALTH ────────────────────────────────────────
    if fraud_type == "account_takeover":
        account_age_days = int(np.clip(rng.exponential(1400), 180, 9125))
    elif fraud_type == "synthetic_identity":
        account_age_days = int(np.clip(rng.exponential(320), 30, 1800))
    elif is_fraud:
        account_age_days = int(np.clip(rng.exponential(500), 1, 5000))
    else:
        account_age_days = int(np.clip(rng.exponential(950), 1, 9125))

    # KYC — synthetic identity has good KYC (that's the point)
    if fraud_type == "synthetic_identity":
        kyc_status = wc(KYC_STATUSES, [55,25,12,5,3])
    elif is_fraud:
        kyc_status = wc(KYC_STATUSES, [32,22,14,24,8])
    else:
        kyc_status = wc(KYC_STATUSES, [72,14,8,2,4])

    existing_loans     = int(np.clip(rng.poisson(1.6), 0, 8))
    credit_util        = round(np.clip(rng.beta(2.2 if is_fraud else 2.0,
                                                2.8 if is_fraud else 3.5), 0.01, 1.0), 4)
    emi_bounce         = int(np.clip(rng.poisson(0.8 if is_fraud else 0.25), 0, 10))
    avg_monthly_bal    = int(np.clip(rng.lognormal(9.8 if not is_fraud else 9.2, 0.9),
                                     500, 2500000))
    salary_credit      = int(random.random() < (0.35 if is_fraud else 0.64))
    dispute_12m        = int(np.clip(rng.poisson(0.8 if is_fraud else 0.08), 0, 8))
    chargeback_12m     = int(np.clip(rng.poisson(0.5 if is_fraud else 0.04), 0, 6))
    inactive_days      = int(np.clip(rng.exponential(8 if is_fraud else 22), 0, 200))

    # ── NETWORK / SOCIAL RISK ─────────────────────────────────
    linked_flagged     = int(random.random() < (0.14 if is_fraud else 0.02))
    bene_first_time    = int(random.random() < (0.42 if is_fraud else 0.18))
    bene_risk_score    = round(np.clip(
        rng.beta(2.0 if is_fraud else 4.5, 3.5), 0.0, 1.0
    ), 4)

    # ── LABEL WITH REALISTIC NOISE ───────────────────────────
    # Real fraud datasets always have label noise:
    # - Some fraud transactions are never caught and labelled legit
    # - Some legit transactions are wrongly charged back → labelled fraud
    label_noise_rate = 0.025    # 2.5% label noise — realistic for bank data
    noise            = random.random() < label_noise_rate
    fraud_label      = int(is_fraud) ^ int(noise)

    return {
        # IDs
        "transaction_id":              str(uuid.uuid4()),
        "user_id":                     f"USR{random.randint(10000000,99999999)}",
        "session_id":                  str(uuid.uuid4()),

        # Demographics
        "age":                         age,
        "gender":                      gender,
        "city":                        city,
        "state":                       state,
        "employment_type":             employment,
        "education_level":             education,
        "annual_income_inr":           annual_income,
        "pan_verified":                int(pan_verified),
        "aadhaar_linked":              int(aadhaar_linked),

        # Bureau / Credit Scores
        "cibil_score":                 cibil_score,
        "crif_score":                  crif_score,
        "experian_score":              experian_score,
        "equifax_score":               equifax_score,
        "bearo_score":                 bearo_score,

        # Card
        "card_network":                card_network,
        "card_type":                   card_type,
        "card_bin":                    card_bin,
        "card_last4":                  card_number[-4:],
        "card_expiry":                 card_expiry,
        "issuer_bank":                 issuer_bank,
        "card_limit_inr":              card_limit,
        "card_age_days":               card_age_days,
        "is_virtual_card":             int(is_virtual_card),

        # Transaction
        "txn_amount":                  txn_amount,
        "txn_currency":                txn_currency,
        "is_international_txn":        int(is_intl_txn),
        "txn_channel":                 txn_channel,
        "upi_id":                      upi_id,
        "merchant_category":           merchant_cat,
        "merchant_id":                 merchant_id,
        "merchant_city":               merchant_city,
        "txn_city_matches_home":       int(txn_city_match),
        "txn_hour":                    txn_hour,
        "txn_day_of_week":             txn_day_of_week,
        "txn_month":                   txn_month,
        "txn_weekend":                 txn_weekend,
        "is_split_txn":                int(is_split_txn),
        "split_count":                 split_count,

        # Velocity
        "txn_count_last_1h":           txn_count_1h,
        "txn_count_last_24h":          txn_count_24h,
        "txn_count_last_7d":           txn_count_7d,
        "txn_count_last_30d":          txn_count_30d,
        "avg_txn_amount_7d":           avg_txn_7d,
        "txn_vs_avg_ratio":            txn_vs_avg_ratio,
        "failed_txn_count_24h":        failed_txn_24h,
        "diff_merchants_24h":          diff_merchants_24h,
        "diff_cities_7d":              diff_cities_7d,
        "max_single_txn_30d":          max_txn_30d,
        "amt_percentile_rank":         amt_pct_rank,

        # Device / Digital
        "device_os":                   device_os,
        "device_brand":                device_brand,
        "browser":                     browser,
        "browser_language":            browser_lang,
        "device_fingerprint":          device_fp,
        "digital_signature":           gen_digital_sig(),
        "screen_resolution":           screen_res,
        "is_new_device":               int(is_new_device),
        "vpn_proxy_flag":              int(vpn_flag),
        "emulator_rooted_flag":        int(emulator_flag),
        "ip_address":                  ip_address,
        "ip_country":                  ip_country,
        "location_spoofed":            int(location_spoofed),
        "device_changed_30d":          device_changed_30d,

        # Authentication
        "otp_attempts":                otp_attempts,
        "otp_success":                 int(otp_success),
        "login_attempts":              login_attempts,
        "txn_pin_attempts":            txn_pin_attempts,
        "mpin_used":                   int(mpin_used),
        "biometric_used":              int(biometric_used),
        "session_duration_sec":        session_duration,
        "password_changed_30d":        password_changed_30d,

        # Account Health
        "account_age_days":            account_age_days,
        "kyc_status":                  kyc_status,
        "existing_loans_count":        existing_loans,
        "credit_utilization_ratio":    credit_util,
        "emi_bounce_count":            emi_bounce,
        "avg_monthly_balance_inr":     avg_monthly_bal,
        "salary_credit_flag":          salary_credit,
        "dispute_count_12m":           dispute_12m,
        "chargeback_count_12m":        chargeback_12m,
        "inactive_days":               inactive_days,

        # Network / Social Risk
        "linked_accounts_flagged":     linked_flagged,
        "beneficiary_first_time":      bene_first_time,
        "beneficiary_risk_score":      bene_risk_score,

        # Label
        "is_fraud":                    fraud_label,
    }


# ─────────────────────────────────────────────────────────────
# MAIN — CHUNKED WRITE
# ─────────────────────────────────────────────────────────────

output_file  = "fraud_v2.csv"
n_fraud      = int(N * FRAUD_RATE)
n_legit      = N - n_fraud
fraud_indices= set(random.sample(range(N), n_fraud))

# Pre-assign ring indices for fraud ring transactions
ring_assign  = {i: random.randint(0,149) for i in fraud_indices}

print(f"⚙️  Generating {N:,} realistic fraud detection records")
print(f"   Fraud rate       : {FRAUD_RATE*100:.1f}%  ({n_fraud:,} fraud / {n_legit:,} legit)")
print(f"   Fraud types      : {len(FRAUD_TYPES)} distinct profiles")
print(f"   Label noise rate : 2.5%")
print(f"   Chunk size       : {CHUNK_SIZE:,}")
print(f"   Output           : {output_file}\n")

first_chunk  = True
total_fraud  = 0
total_legit  = 0

for chunk_start in range(0, N, CHUNK_SIZE):
    chunk_end  = min(chunk_start + CHUNK_SIZE, N)
    records    = []

    for i in tqdm(range(chunk_start, chunk_end),
                  desc=f"  Chunk {chunk_start//CHUNK_SIZE+1}/{N//CHUNK_SIZE}",
                  unit="rows"):
        is_fraud   = i in fraud_indices
        ring_idx   = ring_assign.get(i)
        records.append(build_record(is_fraud, ring_idx))

    df = pd.DataFrame(records)
    df.to_csv(output_file,
              mode="w" if first_chunk else "a",
              header=first_chunk,
              index=False)

    chunk_fraud  = df["is_fraud"].sum()
    total_fraud += chunk_fraud
    total_legit += (chunk_end - chunk_start) - chunk_fraud
    first_chunk  = False
    print(f"   ✔ Rows {chunk_start+1:>7,}–{chunk_end:>7,} done  |  fraud in chunk: {chunk_fraud:,}\n")

print("=" * 58)
print(f"✅  Done → {output_file}")
print(f"   Total rows    : {N:,}")
print(f"   Fraud rows    : {total_fraud:,}  ({total_fraud/N*100:.2f}%)")
print(f"   Legit rows    : {total_legit:,}  ({total_legit/N*100:.2f}%)")
print(f"   Columns       : 70")
print("=" * 58)
print("""
What makes this dataset hard for XGBoost:
──────────────────────────────────────────
✓ 6 fraud types — different behaviour profiles
✓ Overlapping distributions (wide std, close means)
✓ Fraud ring shared devices across user IDs
✓ Account takeover mimics legit user behaviour
✓ Synthetic identity has GOOD bureau scores
✓ 2.5% label noise — real world dirty labels
✓ Legit users sometimes look suspicious (VPN, late night)
✓ Fraudsters sometimes look clean (low velocity, day time)
✓ Noisy amounts — real txn amounts are never perfectly distributed
✓ No single feature separates fraud from legit cleanly
""")
