# Machine Learning Applications for Bitcoin Volatility Forecasting

A thesis research project that develops an **early-warning framework for future high-volatility conditions in the Bitcoin market** using technical indicators, investor sentiment, and macroeconomic variables.

The project formulates volatility forecasting as a **binary time-series classification problem** and compares three approaches:

1. **Random Forest** using sentiment and macroeconomic variables
2. **LSTM** using sequential technical indicators
3. **Hybrid LSTM–Random Forest** combining technical, sentiment, and macroeconomic information

The objective is **volatility risk monitoring and decision support**, not automated trading or price-direction prediction.

<img width="1122" height="1402" alt="Research Stage" src="https://github.com/user-attachments/assets/9a4f2956-0157-42ba-9d05-dfe4c8671c9e" />


---

## Research Objective

The study investigates whether machine-learning models can identify future Bitcoin market conditions in which realized volatility exceeds a predefined high-volatility threshold.

The main research questions are:

- Do investor sentiment and macroeconomic variables contain useful information for forecasting Bitcoin volatility?
- Can an LSTM capture temporal patterns in technical indicators?
- Does combining technical, sentiment, and macroeconomic information improve high-volatility detection?
- Which model provides the most useful trade-off between **recall** and **false alarms** under a risk-monitoring objective?

---

## Research Design

### Target Definition

The target is constructed as a binary variable:

\[
Y_t =
\begin{cases}
1, & \text{if future Bitcoin volatility exceeds the training-set 75th percentile} \\
0, & \text{otherwise}
\end{cases}
\]

This design treats high volatility as a risk event that should be detected in advance.

The volatility threshold is estimated using the **training set only** to reduce information leakage.

### Observation Period

- **Start:** March 2018
- **End:** December 2024
- **Frequency:** Daily
- **Total observations:** 2,477
- **Total features:** 38

---

## Data Sources

### Market Data

- Bitcoin price

### Investor Sentiment

- Fear & Greed Index

### Macroeconomic Variables

- Federal Funds Rate
- Consumer Price Index
- S&P 500 Index
- U.S. Dollar Index

### Technical Indicators

The technical feature set includes:

- Relative Strength Index
- Moving Average Convergence Divergence
- Bollinger Bands
- Average True Range
- Momentum indicators
- Trend indicators
- Volatility indicators

---

## Time-Series Data Split

The data is divided chronologically to preserve the temporal ordering of observations.

| Dataset | Samples | Period |
|---|---:|---|
| Train | 1,733 | 20 Mar 2018 – 16 Dec 2022 |
| Validation | 372 | 17 Dec 2022 – 23 Dec 2023 |
| Test | 372 | 24 Dec 2023 – 29 Dec 2024 |

No random shuffling is used.

The **validation set** is used for model selection and decision-threshold optimization. The **test set** remains strictly out of sample.

---

## Modeling Framework

### Model 1 — Random Forest

The Random Forest model evaluates the predictive contribution of sentiment and macroeconomic information.

**Input features**

- Fear & Greed Index
- S&P 500 return
- DXY return
- Federal Funds Rate
- CPI change

**Research purpose**

- Measure the standalone predictive value of sentiment and macroeconomic variables
- Estimate feature importance
- Provide an interpretable nonlinear benchmark

### Model 2 — LSTM

The LSTM model is designed to learn sequential patterns from technical indicators.

**Input features**

- 26 technical indicators
- Rolling sequence of 14 trading days

**Architecture**

- LSTM layer
- Dropout layer
- Dense layer
- Sigmoid output layer

**Research purpose**

- Capture temporal dependence
- Detect nonlinear patterns in technical signals
- Estimate the probability of a future high-volatility event

### Model 3 — Hybrid LSTM–Random Forest

The hybrid model combines the LSTM probability with sentiment and macroeconomic variables.

**Input features**

- LSTM probability output
- Fear & Greed Index
- S&P 500 return
- DXY return
- Federal Funds Rate
- CPI change

**Research purpose**

- Integrate technical sequence information with broader market conditions
- Evaluate whether cross-domain information improves risk-event detection
- Preserve feature-level interpretability through Random Forest importance scores

---

## Evaluation Strategy

### Primary Metric

**F2 Score**

The F2 Score assigns greater weight to recall than precision:

\[
F_2 = 5 \times \frac{\text{Precision} \times \text{Recall}}
{4 \times \text{Precision} + \text{Recall}}
\]

This metric is appropriate because the project prioritizes the detection of high-volatility events while still penalizing excessive false alarms.

### Supporting Metrics

- Recall
- Precision
- Accuracy
- Balanced Accuracy
- ROC-AUC
- Matthews Correlation Coefficient
- Confusion Matrix

### Threshold Selection

Classification thresholds are selected on the **validation set** using controlled F2 optimization.

The selected thresholds are then applied without modification to the out-of-sample test set.

---

## Out-of-Sample Results

| Model | Recall | Precision | F2 Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest | 70.7% | 20.3% | 0.472 | 0.490 |
| LSTM | 100.0% | ~20.9% | 0.562 | 0.560 |
| Hybrid LSTM–RF | 95.9% | ~21.0% | **0.563** | 0.511 |

### Performance Interpretation

- The **Hybrid LSTM–Random Forest** achieved the highest F2 Score at **0.563**.
- The improvement over the standalone LSTM was marginal: **0.563 versus 0.562**.
- The LSTM detected all high-volatility events in the test set but generated many false alarms.
- Precision remained close to **21%** across the LSTM-based models, indicating that high recall was achieved at the cost of low signal selectivity.
- The LSTM produced the strongest ROC-AUC at **0.560**, although this still represents only modest ranking ability.
- The Random Forest and Hybrid model produced ROC-AUC values close to 0.50, suggesting weak probability discrimination despite threshold-level recall performance.

These results indicate that the framework is more suitable as a **high-sensitivity warning system** than as a standalone trading signal.

---

## Feature Importance

### Random Forest

| Feature | Importance |
|---|---:|
| Fear & Greed Index | 42.0% |
| Federal Funds Rate | 28.5% |
| S&P 500 Return | 16.0% |
| DXY Return | 13.1% |
| CPI Change | 0.4% |

### Hybrid Model

| Feature | Importance |
|---|---:|
| LSTM Probability | 52.2% |
| Fear & Greed Index | 19.3% |
| Federal Funds Rate | 15.8% |
| DXY Return | 6.2% |
| S&P 500 Return | 6.2% |

---

## Key Findings

- **Investor sentiment** was the most influential non-technical variable in the Random Forest model.
- The **Federal Funds Rate** contributed meaningful information, suggesting a relationship between monetary conditions and Bitcoin volatility.
- The **LSTM probability** accounted for 52.2% of hybrid-model importance, indicating that technical sequence information was the dominant component.
- Technical indicators supported high recall, but they did not produce strong precision or ROC-AUC.
- Combining technical, sentiment, and macroeconomic information produced the highest F2 Score, but the improvement over the LSTM was small.
- The results support the use of machine learning as an **early-warning and risk-monitoring tool**, while also highlighting the limitations of using model predictions as direct investment signals.

---

## Practical Contributions

- Built an end-to-end time-series machine-learning pipeline
- Integrated technical, sentiment, and macroeconomic information
- Applied chronological train, validation, and test partitions
- Optimized classification thresholds using the validation set only
- Evaluated models with recall-sensitive and imbalance-aware metrics
- Developed an interpretable early-warning framework for cryptocurrency risk analysis
- Established a foundation for future regime-switching and alternative-data research

---

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r lib.txt
```

Open the main notebook:

```bash
jupyter notebook
```

---

## Methodological Limitations

- High recall was accompanied by low precision and frequent false alarms.
- ROC-AUC remained close to random for the Random Forest and Hybrid models.
- The 75th-percentile target definition is sample-dependent.
- Macroeconomic variables may be affected by publication timing and release-frequency mismatch.
- Daily aggregation may omit intraday volatility information.
- Feature importance does not imply causality.
- The test sample covers only one out-of-sample market period.
- Model performance may change across different volatility regimes.
- Threshold optimization may be sensitive to the validation period.
- The current framework does not include transaction costs or portfolio-level utility because it is not designed as an automated trading strategy.

---

## Future Research

Potential extensions include:

- Hidden Markov Models for volatility-regime detection
- GARCH-family benchmark models
- Realized-volatility forecasting
- Walk-forward validation
- Probability calibration
- Cost-sensitive learning
- Precision–recall curve optimization
- On-chain Bitcoin metrics
- News and social-media sentiment
- Volatility indices and derivatives-market variables
- SHAP-based model interpretation
- Alternative hybrid and ensemble architectures
- Testing across multiple cryptocurrencies and market regimes

---

## Research Positioning

This repository should be interpreted as an empirical study of **Bitcoin volatility risk classification**.

It does not claim that machine learning consistently predicts volatility with high precision. Instead, it evaluates the conditions under which technical, sentiment, and macroeconomic information may support a high-sensitivity early-warning system.

---

## Disclaimer

This project is intended for academic research and educational purposes only. It does not constitute financial advice, investment advice, or a recommendation to trade cryptocurrencies.
