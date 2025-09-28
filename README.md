## Project Overview
The goal of this project is to **analyze customer purchase behavior**, **recommend products based on purchase history**, and **forecast sales** for an e-commerce company.

---

## 🔍 Proposed Solution

### 1️⃣ Data Understanding and Preprocessing
- Load and clean the e-commerce dataset.
- Prepare the data for **clustering**, **association rules**, and **sales forecasting**.

### 2️⃣ Customer Purchase Behavior

**👥 Customer Segmentation**  
- Group customers with similar purchase patterns using clustering algorithms: **K-Means, DBSCAN, Hierarchical Clustering**.

**📦 Association Rule Mining**  
- Identify relationships between products frequently purchased together using **Apriori** and **FP-Growth** algorithms.

**📊 Purchase Probability**  
- Calculate the likelihood of a customer buying recommended products based on mined rules.

### 3️⃣ Sales Forecasting
- Compute the expected cost of recommended products in a basket.  
- Predict future sales using **ARIMA** model.

---

## 🛠️ Technologies Used
- **Python** – Main programming language  
- **Pandas, NumPy** – Data manipulation  
- **MLxtend** – Apriori and FP-Growth for association rules  
- **Scikit-learn** – Clustering algorithms (K-Means, DBSCAN, Hierarchical)  
- **Statsmodels** – ARIMA forecasting model  
- **Jupyter Notebook** – Pipeline execution and visualization  

---

