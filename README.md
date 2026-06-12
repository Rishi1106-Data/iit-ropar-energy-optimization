
# IIT Ropar Energy Optimization Pipeline

## Overview
This project analyzes and optimizes energy consumption data for IIT Ropar buildings. It compares machine learning models for energy prediction and evaluates optimization strategies to reduce electricity costs.

## Project Contents
- `iit_ropar_power_nov_feb.csv` – source energy dataset
- `model_comparison.csv` – ML model performance comparison
- `optimization_details.csv` – optimization results by building
- `prediction_results.png` – prediction visualization
- `optimization_results.png` – optimization visualization

## Suggested Pipeline
1. Load and clean energy consumption data.
2. Perform exploratory data analysis.
3. Train predictive models (Random Forest, Gradient Boosting, XGBoost).
4. Compare model performance using R², MAE, RMSE.
5. Run optimization scenarios.
6. Visualize predictions and savings.

## Results
Current model comparison:

| Model | R² |
|-------|----|
| Random Forest | 0.9903 |
| Gradient Boosting | 0.9874 |
| XGBoost | 0.9892 |

## Installation
```bash
git clone <your-repository-url>
cd IIT_Ropar_Energy_Pipeline
pip install -r requirements.txt
```

## Author
Rishiraj Karn
M.Sc. Data Science & Management
IIT Ropar & IIM Amritsar
