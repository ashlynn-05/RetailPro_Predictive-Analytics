# RetailPro Sales Performance Prediction

Predicting store-level sales for a consumer electronics retailer with linear regression and neural networks, then translating the results into a model RetailPro could actually adopt.

**Highlights:** 24 models tested · Ridge regression reached RMSE 197.6 (MAPE 0.57%) · Best MLP reached RMSE 1,195 (r = 0.96)

## Overview

RetailPro sells consumer electronics, home appliances and personal gadgets, and wants better sales forecasts to guide inventory, promotions and store planning. This project builds and compares predictive models on 2,000 store records with 21 sales-related features, then evaluates which model is most useful for the business.

The work was completed for **MIS373 – AI for Business** (Assignment 2, Task 1) at Deakin University.

> [!NOTE]
> The dataset is synthetic, supplied as part of the coursework. Results describe model behaviour on this dataset, not real RetailPro stores.

## Business questions

- Which store and market factors drive sales the most?
- How accurately can sales be predicted from operational data?
- Should RetailPro adopt a simple linear model or a more flexible neural network?

## Dataset

| | |
|---|---|
| Records | 2,000 stores |
| Features | 21 (19 numeric, 2 categorical) |
| Target | `Sales ($)`, roughly $21.5k to $40k |
| Missing values | None |

Features include store size, employees, advertising spend, product variety, seasonality, customer foot traffic, price index, competitor pricing, customer satisfaction, promotional frequency, online sales share, return rate, supply chain efficiency, economic index, market share, weather impact, opening hours, advertising medium mix, customer age group and store location type.

## Approach

1. **Preprocessing** – one-hot encoded `Store Location Type` and `Customer Demographics (Age Group)` (25 model inputs), 70/30 train–validation split (`random_state=2025`), mean imputation as a safeguard, and Min-Max scaling.
2. **Feature exploration** – correlation analysis against sales to identify key drivers and guide feature selection.
3. **Multiple linear regression (4 models)** – ordinary least squares on all features, OLS on features with |r| > 0.2, and Ridge regression with two alpha values.
4. **Multi-layer perceptrons (20 models)** – randomised search across 2, 3 and 4-layer networks, varying layer width, activation (ReLU, tanh, sigmoid), dropout, optimiser (Adam, RMSprop, Adagrad), learning rate, batch size and epochs, with early stopping.
5. **Evaluation** – MSE, RMSE, MAE, MAPE and correlation between predicted and actual sales on the validation set. RMSE and MAPE were the primary metrics: RMSE penalises large misses, and MAPE expresses error relative to sales in business terms.

## Key findings

**What drives sales.** Advertising spend shows the strongest correlation with sales (about 0.82), followed by price index and competitor pricing (about 0.49 to 0.50) and customer foot traffic (about 0.26). Opening hours, weather, seasonality, urban location and market share have almost no relationship with sales.

**Best linear model**

| Model | MSE | MAE | MAPE | RMSE | Correlation |
|---|---:|---:|---:|---:|---:|
| Ridge regression | 39,060.70 | 169.26 | 0.57% | 197.64 | 0.9984 |
| OLS, all features | 39,076.70 | 169.29 | 0.57% | 197.68 | 0.9984 |
| OLS, features with r > 0.2 | 62,815.70 | 203.53 | 0.68% | 250.63 | 0.9974 |

**Best neural network (MLP Model 3)** – one hidden layer of 120 ReLU units with 0.2 dropout, Adam optimiser (learning rate 0.001):

| MSE | MAE | MAPE | RMSE | Correlation |
|---:|---:|---:|---:|---:|
| 1,429,206.59 | 950.99 | 3.22% | 1,195.49 | 0.9603 |

MLP performance varied widely: the strongest configurations kept errors near 3 to 5% MAPE, while poorly tuned deeper networks failed to converge (MAPE above 80%).

> [!TIP]
> On this dataset, sales are close to a linear function of the features, so Ridge regression beats every neural network by a wide margin. The MLP was recommended for adoption because it can capture non-linear effects as real data gets messier, but the linear model is the stronger, simpler baseline here.

## Business recommendations

- Use the model to anticipate sales by store and flag stores whose actuals deviate sharply from predictions.
- Treat predictions as ranges and trend signals rather than exact dollar figures, matching how RetailPro already plans.
- Use it as a cross-check for sales teams' own forecasts.
- Keep data collection complete and current for the high-impact inputs: advertising spend, pricing and foot traffic.

## Limitations and next steps

- **Data volume** – 2,000 records is small for neural networks; more history across stores, periods and categories would improve generalisation.
- **Missing external factors** – supply chain disruptions, regional spending and other market signals are not captured.
- **Complexity vs. data size** – a 120-unit network risks learning noise; simpler architectures or stronger regularisation are worth testing.
- **Deployment** – cloud hosting (e.g. Google Cloud or AWS) would support scaling if data volume grows.

## Tech stack

Python · pandas · NumPy · scikit-learn · TensorFlow / Keras · Matplotlib · Google Colab

## Notebook

The full analysis, with code, outputs and charts, is in **[`RetailPro_Sales_Prediction.ipynb`](RetailPro_Sales_Prediction.ipynb)**. GitHub renders it directly, so you can read through every step without running anything.

| Section | What's inside |
|---|---|
| 1. Executive Summary | Business problem, chosen model and headline results |
| 2. Data Preprocessing | Encoding, train–validation split, imputation, scaling and correlation analysis |
| 3. AI Model Development | The 4 linear regression and 20 MLP configurations |
| 4. Experiments Report | Model comparison tables, evaluation of the chosen model, adoption, limitations and recommendations |

## Author

**Thi Thao Linh (Lynn) Le** – Bachelor of Business Analytics, Deakin University
