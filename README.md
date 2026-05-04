# Financial Fraud Detection & Business Cost Optimization

An end-to-end machine learning project that builds a fraud detection model and optimizes it for **actual business profitability**, rather than just standard ML metrics. 

By calculating the true cost of False Positives using Customer Lifetime Value (LTV) and churn probability, this project demonstrates why optimizing for precision/recall alone can lose a bank money, and how threshold tuning can maximize net revenue.

## 📋 Executive Summary

* **Algorithm:** **XGBoost** classifier integrated with **SMOTE** (Synthetic Minority Over-sampling Technique) to address extreme class imbalance.
* **Model Performance (ROC-AUC):**
    * **Training AUC (Balanced CV):** **0.9842** — Established via 3-fold cross-validation during the hyperparameter optimization phase.
    * **Test AUC (Unseen):** **0.8877** — Demonstrating high model stability and strong generalization on imbalanced, real-world data distributions.
* **Optimization Results:** 
    * Executed **Hyperparameter Tuning** using `RandomizedSearchCV`, successfully outperforming the baseline model's test AUC of 0.8865.
    * Improved the **False Alarm Rate** from 1.0% to **0.58%** (at the recommended threshold), significantly reducing unnecessary customer friction while maintaining a high fraud catch rate.
* **Business Impact:** Secured an actual net profit of **₹67.4 Million** against a total test transaction volume of **₹365.8 Million**. By shifting the model's operating threshold from the default 0.50 to a more optimized **0.55**, the model achieved the perfect equilibrium between catching high-value fraud and preventing costly customer churn.

## 🧠 The Problem with Standard Fraud Models
Most projects optimize for AUC or Recall. However, in retail banking, blocking a legitimate transaction (False Positive) creates severe customer friction. 

If a frustrated high-value customer churns, the bank loses their Lifetime Value (LTV). Therefore, catching a ₹500 fraud is actively harmful if it costs the bank a ₹15,000 LTV customer. This project bridges the gap between Data Science metrics and Risk Analytics realities.

## 📊 Exploratory Data Analysis (EDA) Insights
Extensive EDA revealed patterns consistent with real-world financial fraud:
1. **The Volume vs. Risk Paradox:** E-commerce drives the highest absolute volume of fraud, but Luxury Goods and Jewelry carry the highest *rate* of fraud (17.5% vs 10.6% average).
   <img width="2063" height="792" alt="merchant-category-analysis(fraud rate and transaction vol by category)" src="https://github.com/user-attachments/assets/e312f570-c20d-4349-afd7-5d6e2c39637a" />

3. **Time-of-Day Anomalies:** Fraud peaks sharply between 2 AM - 4 AM (21.7% fraud rate), but a secondary spike occurs during business hours, mirroring "account takeover" patterns.
   <img width="2063" height="641" alt="time-of-day-fraud-analysis" src="https://github.com/user-attachments/assets/e74d72d3-37b6-4900-b9e9-1fcf527da770" />

5. **Identity Over Credit:** CIBIL scores showed an insignificant 16-point gap between legitimate and fraudulent users. Conversely, device signals (Emulators/Rooted devices, Location Spoofing) and incomplete KYC (PAN/Aadhaar) were massive risk multipliers.
   <img width="2323" height="1278" alt="bureau- -behavioral-score-distributions-fraud-vs-legit" src="https://github.com/user-attachments/assets/38f58bbf-476b-451d-adf1-318659a18177" />
   <img width="2323" height="1278" alt="device-digital-signal-analysis" src="https://github.com/user-attachments/assets/2806351e-8cf4-4f60-b55b-f9c28bfe78b1" />



## ⚙️ Modeling Strategy
* **Data Preparation:** Encoded 20+ categorical features (Label Encoding for tree-based compatibility) and handled 63% missing data in `upi_id` by preserving the nulls as a distinct, valid "Non-UPI" class.
* **Class Imbalance:** Applied SMOTE (Synthetic Minority Oversampling Technique) solely on the training set to bring the 10.6% fraud minority class to a 50/50 balance.
* **Baseline vs. Tuned:** Trained a baseline XGBoost model, followed by `RandomizedSearchCV` on GPU to tune hyperparameters. Deeper trees (`max_depth=8`) and conservative node splitting (`min_child_weight=3`) improved precision and recall simultaneously.

## 💸 Financial Cost Analysis & Threshold Tuning
To find the actual optimal threshold, I modeled the exact financial cost of a False Alarm using a more realistic churn impact:
`False Alarm Cost = Ops Cost (₹500) + [Churn Probability (15%) * Customer LTV (₹15,000)]`
**True Cost of a False Alarm = ₹2,750**

I then evaluated the exact rupee value of the frauds caught by the model against the operational cost of the false alarms at various thresholds:

| Threshold | False Alarms | Cost of Friction | Actual Savings | Net Profit |
| :--- | :--- | :--- | :--- | :--- |
| 0.20 (Aggressive) | 3,381 | ₹9.30M | ₹71.08M | ₹61.78M |
| 0.50 (Default) | 597 | ₹1.64M | ₹68.99M | ₹67.34M |
| **0.55 (Optimal)** | **464** | **₹1.28M** | **₹68.64M** | **₹67.36M** |
| 0.65 (Conservative)| 316 | ₹0.87M | ₹67.49M | ₹66.62M |

### ⚖️ Scale & Context
To put these numbers into perspective: The total transaction volume processed during this test window was roughly **₹365.8 Million**. By optimizing the threshold for net revenue rather than raw recall, the model successfully secured **₹67.36 Million** in actual net profit. This strategy prioritizes the bank's bottom line by aggressively lowering False Alarms as the cost of customer churn increases.

### 🐋 "Hunting Whales"
By auditing the exact transactions flagged at the 0.55 threshold, the model naturally optimized for high-value targets. While the average fraud in the dataset was ₹9,422, the average fraud caught by the model was **₹10,847**, proving the model successfully identified complex, large-sum fraud patterns rather than just low-value card testing bots.

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
