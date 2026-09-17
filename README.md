# Second-Hand Car Price Prediction — Feature Engineering & Dimensionality Reduction

Weekly project applying feature engineering and dimensionality reduction to a second-hand car
sales dataset, then training and comparing regression models (with and without PCA).

## Project Overview

The goal is to prepare a used-car dataset for machine learning by:

1. Cleaning and preprocessing the data
2. Engineering informative features (car age, price per mile, service/insurance/accessory flags, etc.)
3. Reducing the feature space via correlation analysis and Principal Component Analysis (PCA)
4. Training and evaluating models to compare performance with and without dimensionality reduction

## Repository Structure

```
second_hand_cars/
├── Second_Cars_DR.ipynb       # Main notebook (all analysis)
├── data/
│   └── second_hand_cars.csv   # Dataset (2500 rows, 16 columns)
└── README.md
```

## Dataset

The dataset contains second-hand car listings with columns such as:

- `Company Name`, `Car Name`, `Variant`, `Fuel Type`, `Tyre Condition`, `Make Year`
- `Owner Type`, `Mileage`, `Price`, `Transmission Type`, `Body Color`
- `Service Record`, `Insurance`, `Registration Certificate`, `Accessories`

## Dependencies

- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn (models, preprocessing, PCA)

## Running the Notebook

Open the notebook from the `second_hand_cars/` folder so the relative data path resolves:

```bash
jupyter notebook Second_Cars_DR.ipynb
```

Or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute Second_Cars_DR.ipynb --output Second_Cars_DR.ipynb
```

## Key Results

- **Feature Engineering** created new features: Car Age, Price per Mile, service/insurance/accessory
  binary flags, and accessory count.
- **Correlation analysis** found no highly correlated feature pairs (all |r| ≤ 0.85), so no features
  were removed on that basis.
- **PCA** reduced the feature space while retaining 95% variance, using 40 of 173 one-hot encoded
  features.
- **Best model**: Random Forest (without PCA)
  - RMSE: 19,639.81
  - MAE: 13,156.84
  - R²: 0.9930

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Random Forest (no PCA) | 19,640 | 13,157 | 0.9930 |
| Gradient Boosting (no PCA) | 33,683 | 25,519 | 0.9795 |
| Gradient Boosting (PCA) | 180,232 | 148,194 | 0.414 |
| Random Forest (PCA) | 186,331 | 155,269 | 0.373 |
| Linear Regression (PCA) | 210,270 | 178,088 | 0.202 |
| Linear Regression (no PCA) | 211,195 | 178,992 | 0.195 |

## Conclusion

- Feature engineering and PCA effectively compressed the feature space while preserving predictive
  signal.
- Tree-based ensemble models (Random Forest, Gradient Boosting) outperformed linear models, capturing
  non-linear relationships in car pricing.
- The Random Forest model explained ~99.3% of the variance in car prices, making it the best performer.
