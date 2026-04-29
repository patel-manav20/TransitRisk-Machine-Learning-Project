# TransitRisk: Cost-Sensitive, Calibrated Next-Hour Delay Risk Forecasting for Urban Transit Operations

## Abstract
Urban transit systems regularly experience delay spillovers that degrade reliability and increase both passenger and operator burden. This report presents **TransitRisk**, an end-to-end machine learning system for predicting **next-hour elevated delay risk** at the **station-route-hour** level. The implemented pipeline includes synthetic data generation, preprocessing, temporal feature engineering, model training, calibration, threshold optimization, uncertainty estimation, and dashboard deployment. The dataset starts from approximately **1.25M trip events** and aggregates to approximately **153K station-route-hour records**. Seven candidate classifiers are evaluated, and calibrated XGBoost is selected for deployment. On held-out test data, the final model achieves **ROC-AUC 0.810**, **PR-AUC 0.766**, **F1 0.643**, and **Brier 0.169**. A cost-sensitive threshold policy with \(C_{FN}=5\), \(C_{FP}=1\) yields \(t_{cost}\approx0.163\). Conformal prediction is integrated to expose uncertainty in operator-facing views.

## I. Introduction
Transit delay is operationally costly because disruptions propagate across routes, stations, and time windows. Simple rule-based thresholds and short-horizon heuristics often fail under dynamic weather and demand conditions. TransitRisk addresses this gap with an end-to-end ML system that predicts whether a station-route pair will experience elevated delay in the next hour and converts risk estimates into actionable decisions.

**Contributions**
- End-to-end pipeline from synthetic data generation to deployed dashboard.
- Strict temporal split and leakage prevention.
- 38-feature design across temporal, lag, weather, and interaction categories.
- Calibration, cost-sensitive thresholding, and conformal uncertainty integration.
- Multi-tab decision support dashboard.

## II. Dataset Description
- Synthetic but structured data for reproducibility and realistic transit behavior.
- Scale: ~1.25M trip events, ~153K station-route-hour rows.
- Target: next-hour elevated delay classification (>= 5 min mean delay).
- Chronological split: train/val/test = 60/20/20.

## III. Data Preprocessing
Preprocessing includes timestamp normalization, schema checks, aggregation from trip-level to station-route-hour, missing-value handling, and leakage-safe temporal alignment. Features at time \(t\) use only present/past information, while labels are defined for \(t+1\).

## IV. Proposed Framework
Workflow: data generation -> aggregation -> feature engineering -> temporal split -> model training -> calibration -> threshold optimization -> conformal uncertainty -> evaluation -> dashboard.

Models evaluated: Logistic Regression, Naive Bayes, kNN, Decision Tree, Random Forest, SVM-RBF, and XGBoost. Final deployment uses calibrated XGBoost with \(t_{default}=0.50\), \(t_{F1}\approx0.345\), and \(t_{cost}\approx0.163\).

## V. Workflow Diagram
![Figure 1: End-to-End Workflow of TransitRisk](report/diagrams/workflow.png)

## VI. Experimental Design
Chronological validation is used to avoid temporal leakage. Metrics include ROC-AUC, PR-AUC, F1, and Brier score. PR-AUC is emphasized due to imbalance and alerting relevance.

## VII. Experimental Results
| Model | ROC-AUC | PR-AUC | F1 | Brier |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7601 | 0.6764 | 0.5998 | 0.1916 |
| Naive Bayes | 0.7083 | 0.5743 | 0.4887 | 0.2614 |
| kNN | 0.7757 | 0.7185 | 0.5837 | 0.1851 |
| Decision Tree | 0.7849 | 0.7249 | 0.5916 | 0.1782 |
| Random Forest | 0.8061 | 0.7618 | 0.6691 | 0.1766 |
| **XGBoost (Final)** | **0.8097** | **0.7662** | 0.6433 | **0.1691** |
| SVM-RBF | 0.7534 | 0.6994 | 0.5853 | 0.1916 |

XGBoost provides the strongest combined ranking and probability quality, making it most suitable for deployed risk scoring.

## VIII. System Architecture Diagram
![Figure 2: TransitRisk System Architecture](report/diagrams/architecture.png)

## IX. Work Distribution
- **Ayush**: synthetic data design, aggregation, preprocessing QA.
- **Manav**: model development, dashboard integration, deployment flow.
- **Parth**: calibration, thresholding, conformal uncertainty, reporting.

## X. Conclusion
TransitRisk demonstrates a full ML-to-operations pipeline for next-hour transit delay risk. The deployed system combines calibrated probabilistic forecasting, cost-aware decision policy, and uncertainty-aware interpretation in a practical dashboard.
