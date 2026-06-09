# Volatility Regime Detection: From Machine Learning to Structural Dynamics

## 1. Introduction: The Limits of ML in Financial Causality

The initial phase of this project successfully utilized machine learning (XGBoost, LSTM) alongside Hidden Markov Models (HMM) to classify and predict volatility regimes in the stock market. However, rigorous ablation studies revealed a critical limitation: **Target Leakage and the Autoregressive Trap.**

When tasked with predicting a volatility regime, ML models predominantly reverse-engineer the autoregressive nature of the data (i.e., the best predictor of tomorrow's volatility is today's volatility). Consequently, the macroeconomic data was effectively ignored by the predictive models.

While ML is exceptional at point-in-time classification, it acts as a "black box" that fails to answer the core econometric question of this thesis: *When a specific economic shock occurs, how exactly does it propagate through different financial sectors over time?* To transition from mere correlation to **proven causality**, this project pivoted from predictive Machine Learning to **Structural Econometrics**, specifically utilizing Vector Autoregression (VAR) and Impulse Response Functions (IRFs).

## 2. Methodology

To accurately model how macroeconomic indicators drive sector volatility, we implemented a structural pipeline:

### A. Expectation Residuals (True Macro Surprises)

Markets are forward-looking; an inflation print of 3% does not move the market if Wall Street already expected 3%. Therefore, feeding raw macroeconomic data into a model is flawed. We utilize rolling **Autoregressive AR(1) models** on the FRED data to calculate an expected baseline. The model then subtracts the actual print from the expected print to isolate the **Expectation Residual** (The "Shock" or "Surprise").

### B. The Sparse Time-Series Panel

To prevent look-ahead bias, all macro surprises are strictly lagged by their exact historical publication delays (e.g., CPI is lagged ~11 trading days). By mapping these isolated shocks onto a daily trading calendar, we create a sparse time-series panel. This allows the model to observe sector volatility during "empty" days (no news) versus days where a shock hits the tape.

### C. Structural VAR & Cholesky Ordering

We model the ecosystem using a Vector Autoregression (VAR) system. Unlike isolated regressions, VAR assumes all variables are endogenous and interact simultaneously. Crucially, we enforce **Cholesky Ordering** (Macro Surprises $\rightarrow$ Sector Volatility). This mathematical restriction assumes that a macro print released at 8:30 AM can shock the stock market on the same day, but a stock market crash at 2:00 PM cannot retroactively alter the morning's macro print.

### D. Feature Selection: The Autoregressive Trap of GARCH

During methodology optimization, conditional volatility (EGARCH/GARCH) was tested as the primary target variable for the VAR system. However, GARCH formulas are inherently highly autoregressive. When fed into the VAR model, this autoregression absorbed the vast majority of the variance, effectively masking the impact of the exogenous macroeconomic shocks.

*Note: The Energy sector (`XLE`) was the sole exception, proving that physical commodity shocks are uniquely violent enough to break through the GARCH smoothing.*

To properly isolate and measure the impact of macroeconomic surprises across all sectors, the model utilizes **Squared Log Returns** (`Sq_Log_Return`). Squared returns preserve the raw, unexplained variance (the "noise") necessary for the VAR model to mathematically attribute sudden market reactions to external economic data.

## 3. Results Guide: How to Read the Notebook Outputs

When executing the `03_Structural_VAR_and_IRF.ipynb` notebook, the pipeline outputs four distinct analytical tools. Here is how to interpret them:

### I. Granger Causality Tests

* **What it is:** A statistical hypothesis test determining if one time series is useful in forecasting another.
* **How to read it:** Look for the p-values across Lags 1 through 5. Asterisks denote significance (`*` = 10%, `` = 5%, `***` = 1%). If `CPI_Surprise` shows a `***` at Lag 2, it is mathematical proof that an inflation shock has significant predictive power over the sector's volatility exactly two days later.

### II. VAR Equation Summary Table

* **What it is:** The algebraic coefficients of the fitted VAR model for the target sector.
* **How to read it:** * **Autoregression:** The `L(n).Sq_Log_Return` rows prove volatility clustering.
* **Coefficients:** A positive coefficient means the shock increases volatility; a negative coefficient means the shock suppresses volatility (calms the market).



### III. Impulse Response Functions (IRFs)

* **What it is:** A visual simulation. We mathematically inject a +1 Standard Deviation shock into a specific macro indicator at Day 0 and trace how the sector reacts over the following 15 days.
* **How to read it:** * The **Solid Blue Line** represents the change ($\Delta$) in volatility.
* The **Dotted Lines** are the 95% confidence intervals.
* If the blue line and *both* dotted lines rise above the zero-line, the shock caused a statistically proven increase in volatility.
* The point where the line crosses back down to zero represents the **"Half-Life"** of the shock (the exact day the market finished digesting the news).



### IV. Historical Alignment (Actual vs. Predicted)

* **What it is:** An overlay of the actual sector volatility (Grey) versus the volatility predicted strictly by the VAR model (Red).
* **How to read it:** This chart serves as a reality check. While linear VAR models historically underestimate the absolute *magnitude* of Black Swan crashes (fat tails), a successful model will perfectly align with the *timing* of historical regime shifts (e.g., 2008, 2020), proving the structural relationships hold true across decades.
