# California House Price Prediction

Predicting the median house value of California districts from features such as
median income, house age and location, using scikit-learn's California Housing
dataset (1990 U.S. Census).

## Approach
1. Exploratory data analysis: distributions, correlations, geographic patterns
2. Outlier removal in AveRooms, AveBedrms, AveOccup and Population
3. Train/test split first, then scaling (scaler fit on train only to avoid data leakage)
4. Baseline Linear Regression, then Ridge, Lasso, feature engineering and PCA
5. Error analysis with predicted-vs-actual and residual plots

## Results
| Model | RMSE | R² |
|---|---|---|
| Linear Regression (baseline) | 0.6798 | 0.6539 |
| Ridge | 0.6798 | 0.6539 |
| Lasso | 0.6798 | 0.6540 |
| Linear Reg + new features | **0.6597** | **0.6741** |
| Linear Reg + PCA | 0.8006 | 0.5200 |

The target is in units of $100,000, so an RMSE of 0.66 is an average error of about $66,000.

## Key findings
- MedHouseVal is capped at 5.0 ($500,000), and MedInc (15) and HouseAge (52) are capped too. This limits how well a linear model can fit the data.
- AveRooms, AveBedrms, AveOccup and Population contain extreme outliers, which were removed.
- Engineered features (RoomsPerPerson, BedroomRatio) gave the best improvement.
- Ridge and Lasso did not help because the model was not overfitting.
- PCA hurt performance because it discards directions that matter for prediction.
- Median income is the strongest predictor of house value (correlation 0.69), and prices cluster geographically around the coast.

## How to run
Open `house_price_prediction.ipynb` in Google Colab or Jupyter, or run locally:

    pip install -r requirements.txt
    jupyter notebook house_price_prediction.ipynb

The dataset downloads automatically through `fetch_california_housing`.

## Next steps
- Try Random Forest and XGBoost
- Log-transform the target
- Add better location features (e.g., distance to the coast, LA and San Francisco)
