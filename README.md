# Financial Fraud Detection & Business Cost Optimization

An end-to-end machine learning project that builds a fraud detection model and optimizes it for **actual business profitability**, rather than just standard ML metrics. 

By calculating the true cost of False Positives using Customer Lifetime Value (LTV) and churn probability, this project demonstrates why optimizing for precision/recall alone can lose a bank money, and how threshold tuning can maximize net revenue.

## 📋 Executive Summary
* **Algorithm:** XGBoost with SMOTE (handling extreme class imbalance).
* **Dataset:** 400,000 transactions mimicking Indian fintech patterns (CIBIL scores, UPI, Aadhaar/PAN linkage, device fingerprinting).
* **Key Finding:** Standard bureau scores (CIBIL/CRIF) are insufficient for detecting fraud alone due to heavy distribution overlap. Behavioral velocity and device signals are far stronger predictors.
* **Business Impact:** Secured an actual net profit of **₹64.5 Million** against a total test transaction volume of **₹365.8 Million**. By shifting the model's operating threshold from the default 0.50 to a more conservative 0.65, the model sacrificed a small amount of raw recall but reduced costly false alarms by nearly 50%—effectively neutralizing the financial threat while minimizing customer churn.

## 🧠 The Problem with Standard Fraud Models
Most portfolio projects optimize for AUC or Recall. However, in retail banking, blocking a legitimate transaction (False Positive) creates severe customer friction. 

If a frustrated high-value customer churns, the bank loses their Lifetime Value (LTV). Therefore, catching a ₹500 fraud is actively harmful if it costs the bank a ₹1,25,000 LTV customer. This project bridges the gap between Data Science metrics and Risk Analytics realities.

## 📊 Exploratory Data Analysis (EDA) Insights
Extensive EDA revealed patterns consistent with real-world financial fraud:
1. **The Volume vs. Risk Paradox:** E-commerce drives the highest absolute volume of fraud, but Luxury Goods and Jewelry carry the highest *rate* of fraud (17.5% vs 10.6% average).
2. **Time-of-Day Anomalies:** Fraud peaks sharply between 2 AM - 4 AM (21.7% fraud rate), but a secondary spike occurs during business hours, mirroring "account takeover" patterns.
3. **Identity Over Credit:** CIBIL scores showed an insignificant 16-point gap between legitimate and fraudulent users. Conversely, device signals (Emulators/Rooted devices, Location Spoofing) and incomplete KYC (PAN/Aadhaar) were massive risk multipliers.

## ⚙️ Modeling Strategy
* **Data Preparation:** Encoded 20+ categorical features (Label Encoding for tree-based compatibility) and handled 63% missing data in `upi_id` by preserving the nulls as a distinct, valid "Non-UPI" class.
* **Class Imbalance:** Applied SMOTE (Synthetic Minority Oversampling Technique) solely on the training set to bring the 10.6% fraud minority class to a 50/50 balance.
* **Baseline vs. Tuned:** Trained a baseline XGBoost model, followed by `RandomizedSearchCV` on GPU to tune hyperparameters. Deeper trees (`max_depth=8`) and conservative node splitting (`min_child_weight=3`) improved precision and recall simultaneously.

## 💸 Financial Cost Analysis & Threshold Tuning
To find the actual optimal threshold, I modeled the exact financial cost of a False Alarm:
`False Alarm Cost = Ops Cost (₹500) + [Churn Probability (5%) * Customer LTV (₹1,25,000)]`
**True Cost of a False Alarm = ₹6,750**

I then evaluated the exact rupee value of the frauds caught by the model against the operational cost of the false alarms at various thresholds:

| Threshold | False Alarms | Cost of Friction | Actual Savings | Net Profit |
| :--- | :--- | :--- | :--- | :--- |
| 0.20 (Aggressive) | 3,381 | ₹22.8M | ₹71.0M | ₹39.8M |
| 0.50 (Default) | 597 | ₹4.0M | ₹68.9M | ₹63.4M |
| **0.65 (Optimal)** | **316** | **₹2.1M** | **₹67.4M** | **₹64.5M** |

### ⚖️ Scale & Context
To put these numbers into perspective: The total transaction volume processed during this test window was roughly **₹365.8 Million**. By optimizing the threshold for net revenue rather than raw recall, the model successfully secured **₹64.5 Million** in actual net profit, neutralizing the vast majority of the financial threat while keeping customer friction to an absolute minimum.

### 🐋 "Hunting Whales"
By auditing the exact transactions flagged at the 0.65 threshold, the model naturally optimized for high-value targets. While the average fraud in the dataset was ₹9,422, the average fraud caught by the model was **₹10,859**, proving the model successfully identified complex, large-sum fraud patterns rather than just low-value card testing bots.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab (T4 GPU Accelerated)

## 🚀 How to Run
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. The custom data generator script is included (`generator.py`). Run it to generate the 400k row dataset.
4. Open the Jupyter Notebook to step through the EDA, modeling, and financial analysis.
