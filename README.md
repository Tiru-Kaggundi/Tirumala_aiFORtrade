AI4Trade Forecasting Challenge 2025

Team: Indian Westerlies (India)
Participant: Tirumala Venkatesh Kaggundi
1. Objective
I forecast bilateral monthly trade flows (USD) for the United States and China at the HS-6 level, aggregated to HS-4 for submission. The goal is to predict October 2025 trade values for each origin (USA / CHN), destination, and product (HS-4), separately for Exports and Imports, outperforming the naive “latest-month carry-forward” baseline under the symmetric Mean Absolute Percentage Error (sMAPE) metric.
2. Data and Forecast Horizon Policy
Source: Harmonized monthly trade data (OEC 2023–2025 CSV release) aggregated to national level. Data upto 2024 was provided and I downloaded monthly data from Trademap.org where I could get data upto July 2025 for USA and August 2025 for CHN. The same were weaved into provided data. 
Coverage: CHN → Top 30 export / 29 import partners; USA → Top 30 each (flow); partners ensure ≥ 20 countries with ≥ 200 HS-4 lines. The CHN import partners were selected to ensure that the partners usually export over 150 unique hs4 to CHN. 
Dual-Horizon Setup for training and testing:
China (CHN): last month Aug 2025 → horizon h=2 → target Oct 2025
USA (USA): last month Jul 2025 → horizon h=3 → target Oct 2025
Models are trained directly for y(t + h), not recursively.
3. Feature Engineering
All features are deterministic functions of the past, defined per (origin, destination, HS-6, flow):
Temporal: month_num, quarter, month_id (seasonality/trend)
Lags: 1, 2, 3, 6, 10 (China) / 9 (USA), 12
Moving averages: ma3, ma6, ma12
Volatility: roll_std6, pctchg1, pctchg3
Cross-flow: cross_flow_lag1, cross_flow_ma3, cross_flow_lag13
Macro drift: origin_total_exports_ma3, origin_total_imports_ma3, origin_total_trade_ma3
Global supply chain pressure and composite leading (OECD) indicators: scpi_ma2, cli_ma2 (shifted +1 month for CHN, +2 for USA)
Activity flags: was_trade_lag1, consec_zero_run
4. Modeling Architecture
I train four independent segment models: CHN Export (h=2), CHN Import (h=2), USA Export (h=3), USA Import (h=3). Each segment uses an ensemble of three gradient-boosted trees: XGBoost-Tweedie (power≈1.35), XGBoost-Log1p, and LightGBM-RMSE. These minimize RMSE or Tweedie deviance, evaluated via sMAPE.
5. Validation Strategy
Rolling time-split CV aligned to Horizon Policy (C1–C6 for CHN, U1–U5 for USA). Each fold trains ≤t, skips 1 month, validates on t+h. Fold weights emphasize late 2024–2025 folds close to Oct 2025; weighted mean HS-4 sMAPE used for model selection.
6. Ensemble Logic
For each segment, I form OOF predictions and compute fold-weighted sMAPE per model. Weights are inverse-sMAPE, regularized with prior (0.45 Tweedie / 0.25 Log1p / 0.30 LGBM) and a 0.70:0.30 blend. If a model wins on the latest folds, its weight receives +10% recency tilt. Weights normalized to sum=1, clipped ∈[0.15,0.65] to prevent dominance by one of them. 
7. Post-processing and Submission
Aggregate HS6→HS4, round to USD, filter top 20 destinations per (origin, flow) with ≥200 HS4 lines. Format: “Country1”,”Country2”,”ProductCode”,”TradeFlow”,”Value”. Output: submission_final.csv (UTF-8 plain text, quoted, 4-digit HS codes preserved, with row header).
8. Results & Reproducibility
Validation (2024H2–2025H1) shows ensemble HS4 sMAPE≈0.77 vs naive≈0.94. All pipelines executed in Python /Colab using pandas, numpy, lightgbm, xgboost. GPUs used for training wherever possible. I have uploaded all code to my github at this place: 
9. Conclusion
The dual-horizon, feature-rich ensemble is modelled to capture seasonality, macro and cross-flow dynamics. The direct (CHN h=2 / USA h=3) setup yields stable, interpretable forecasts that outperform naive baselines. The model ensemble used are usual Kaggle winners. All code, features, and weights reproducible under the shared schema. I hope the model is strong enough to predict accurately for October, while beating the naïve sMAPE. 
10. What further could be done?
Better hyperparameter tuning is needed if the model shows promise. 
