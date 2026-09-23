# kenya-mortality-forecasting-lee-carter-vs-lstm
BSc Actuarial Science project comparing Lee-Carter and LSTM models for age-specific mortality forecasting in Kenya, with 2025-2030 projections (R, UN WPP 2024).
# Comparing Lee-Carter and LSTM Models for Age-Specific Mortality Forecasting in Kenya (2025-2030)

Undergraduate research project submitted in partial fulfilment of the requirements for the award of the **Bachelor of Science in Actuarial Science**, Daystar University, School of Science, Engineering and Health.

| | |
|---|---|
| **Author** | Zuber Haji (ADM 22-2333) |
| **Supervisor** | Professor Richard Simwa |
| **Institution** | Daystar University, Department of Science and Engineering |
| **Language / tools** | R, `keras3` (TensorFlow backend), `wpp2024` |

---

## Overview

Accurate age-specific mortality forecasts underpin life insurance and annuity pricing, pension planning, and public health resource allocation. Kenyan mortality forecasting has relied mainly on classical stochastic models such as Lee-Carter, while deep learning models such as Long Short-Term Memory (LSTM) networks have been reported globally to match or beat them. No prior study had benchmarked LSTM against Lee-Carter using Kenya's own age-specific mortality data.

This project fills that gap by fitting both models to the same Kenyan mortality panel, comparing them on a held-out test period, and using the better model to project mortality for 2025-2030.

## Research Questions

1. How effectively can an LSTM model forecast age-specific mortality rates in Kenya?
2. How does LSTM's forecasting accuracy compare with the Lee-Carter model on Kenyan data?
3. What are the projected age-specific mortality rates for Kenya for 2025-2030?

## Data

- **Source:** United Nations World Population Prospects (WPP) 2024, accessed through the R package `wpp2024` (Kenya, country code 404).
- **Coverage:** single-year ages 0-100, years 1990-2024, both sexes combined (a 101 x 35 age-by-year matrix, 3,535 observations).
- **Missing values:** none requiring imputation.

| Partition | Years | Use |
|---|---|---|
| Training | 1990-2019 | Estimate Lee-Carter parameters; train LSTM |
| Validation | 2020-2022 | LSTM hyperparameter tuning and monitoring |
| Test (hold-out) | 2023-2024 | Out-of-sample accuracy evaluation |

## Methodology

**Lee-Carter (benchmark)**
- Model: `log m(x,t) = a_x + b_x * k_t + ε_x,t`, with constraints Σb_x = 1 and Σk_t = 0.
- Parameters estimated by Singular Value Decomposition of the centred log-mortality matrix.
- The mortality index `k_t` is forecast with a random walk with drift.

**LSTM**
- Applied directly to min-max normalised mortality series using rolling input windows.
- Architecture reported: 1 LSTM layer, 32 units, window length 5, Adam optimiser, MSE loss, 50 epochs.
- Forecasts for 2025-2030 generated recursively, then transformed back to the original scale.

**Evaluation**
- Metrics: RMSE, MAE, MAPE.
- Statistical test: Diebold-Mariano test of equal predictive accuracy (H₀: E(d_t) = 0).

## Key Results

Hold-out period 2023-2024:

| Metric | LSTM | Lee-Carter |
|---|---|---|
| RMSE | 0.1084 | **0.0091** |
| MAE | 0.0744 | **0.0059** |
| MAPE (%) | 2,276.46 | **27.83** |

- **Diebold-Mariano statistic = -7.294, p < 0.001**: the null of equal accuracy is rejected in favour of Lee-Carter.
- Lee-Carter was adopted for the final 2025-2030 projections.
- Both models project a gradual **rise** in mortality over 2025-2030, largest in relative terms among children (Lee-Carter: age 5 about +2.08%, age 0 about +1.62%) and smallest around ages 40-60.
- The result diverges from much of the international literature but is consistent with Karani & Dongxiao (2025), the only other identified Kenyan comparison of a deep-learning model against a classical benchmark. A likely explanation is Kenya's short (35-year), structurally disrupted (HIV/AIDS era) mortality record, which favours the more parsimonious Lee-Carter structure.

> These findings should not be read as showing that LSTM is generally inferior to classical models. They apply to the Kenyan data and forecast horizon studied.

## Repository Contents

```
.
├── README.md
├── HAJI_MGAWA_FINAL_PROJECT.pdf        # Full research report
├── code/
│   └── Analysis_Code_For_Appendices.R  # Full R analysis pipeline
└── data/
    ├── Kenya_ASMR_Matrix_1990_2024_mxB.csv
    ├── Kenya_ASMR_Long_1990_2024_all_sexes.csv
    ├── Kenya_Population_Exposure_Matrix_1990_2024.csv
    ├── LeeCarter_Age_Parameters.txt
    ├── LeeCarter_Full_Forecast.txt
    └── LSTM_Forecast_2025_2030.txt
```

The supporting datasets and code are also available in the [shared Google Drive folder](https://drive.google.com/drive/folders/1sQsJ-Z7-orTaY1Npwb2yEZvrwltpOZ8I?usp=drive_link).

## Reproducing the Analysis

The analysis covers data preparation, historical trends, Lee-Carter estimation, LSTM training, forecasting, error metrics, the Diebold-Mariano test, and generation of tables and figures.

1. Install R (4.x recommended) and the required packages:

```r
   install.packages(c("wpp2024", "keras3", "tidyverse", "forecast"))
   # keras3 needs a TensorFlow backend:
   keras3::install_keras()
```

   Check the top of `Analysis_Code_For_Appendices.R` for the exact package list.

2. Place the data files where the script expects them (or let it pull the data from `wpp2024`).
3. Run the script:

```r
   source("code/Analysis_Code_For_Appendices.R")
```

LSTM training uses stochastic weight initialisation, so set a random seed for exact reproducibility. Results may vary slightly between runs.

## Limitations

- Short historical record (35 years), which likely disadvantages LSTM.
- WPP figures are modelled estimates, not fully observed vital registration data.
- LSTM is a "black box" compared with Lee-Carter's interpretable parameters.
- Both models rely on historical patterns and cannot anticipate future shocks (pandemics, policy changes).
- National-level, both-sexes analysis only; no county-level or cause-specific modelling.

## Citation

If you use or reference this work:

```
Haji, Z. (2026). Comparing Lee-Carter and Long Short-Term Memory Models for
Age-Specific Mortality Forecasting in Kenya: 2025-2030 Projections.
BSc Actuarial Science research project, Daystar University.
```

## Key References

- Lee, R. D., & Carter, L. R. (1992). Modeling and forecasting U.S. mortality. *JASA*, 87(419), 659-671.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253-263.
- Karani, A., & Dongxiao, R. (2025). Forecasting years lived in poor health in Kenya: ARIMA vs LSTM. *East African Journal of Health and Science*, 8(2), 211-222.
- United Nations, DESA, Population Division. (2024). *World Population Prospects 2024*.

The full reference list is in the report PDF.


## Contact

Zuber Haji, Daystar University
