# Customer Analytics & Segmentation Suite

Un'applicazione end-to-end di **Customer Intelligence** e **Unsupervised Machine Learning** sviluppata in Python e Streamlit, basata sul dataset *Online Retail II*.

## 📌 Features & Fondamenti Teorici

1. **Analisi delle Dipendenze**:
   - Confronto tra correlazione lineare di **Pearson ($r$)**, monotona di **Spearman ($\rho$)** e **Mutual Information ($I$)**.
   - Dimostrazione del limite della linearità su relazioni quadratiche ($Y = X^2$).

2. **Riduzione Dimensionale & Clustering**:
   - **PCA (Principal Component Analysis)**: Riduzione dello spazio delle feature RFM a 2 Componenti Principali (88.2% di varianza spiegata).
   - **K-Means Custom (NumPy)**: Algoritmo scritto da zero utilizzando la vettorializzazione NumPy, con benchmark rispetto all'implementazione di Scikit-Learn.
   - **DBSCAN**: Clustering basato sulla densità per la gestione e identificazione degli outlier ($k$-distance graph).
   - **Cluster Stability**: Validazione formale della stabilità tramite **Adjusted Rand Index (ARI = 0.996)** su subsampling dell'80%.

3. **Business Intelligence & Profilazione**:
   - Calcolo delle metriche **RFM (Recency, Frequency, Monetary)**.
   - Identificazione di 3 coorti di clienti (Standard, Inattivi, Top VIP / Wholesalers) e piani d'azione strategici.

---

## 🛠️ Tech Stack
- **Languages & Frameworks**: Python 3.11, Streamlit
- **Data & Math**: NumPy, Pandas, Scikit-Learn, SciPy
- **Data Visualization**: Matplotlib, Seaborn

---

## 🚀 Esecuzione Locale

```bash
# 1. Clona la repository
git clone [https://github.com/tuo-username/customer-segmentation-analytics.git](https://github.com/tuo-username/customer-segmentation-analytics.git)
cd customer-segmentation-analytics

# 2. Crea e attiva l'ambiente virtuale
python3 -m venv .venv
source .venv/bin/activate

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Avvia la dashboard
streamlit run app.py
