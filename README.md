# Soccer Match Outcome Prediction using Machine Learning

## Project Overview

This project aims to predict soccer match outcomes using match statistics and betting odds, combining traditional sports metrics with implied probabilities. We build models to predict two key targets: Full-Time Result (FTR): Win, Draw, or Loss & Over/Under 2.5 Goals (OU_2.5)

## Technology Used

Language: R

Packages: dplyr, ggplot2, caret, xgboost, randomForest, glmnet, tidyverse

## Data Source

All data was sourced from "https://www.football-data.co.uk/". We combined the Premier League data  with the Championship League data over multiple years to result in our combined_data.csv.

## Data Loading and Cleaning

The dataset match_data_eda.csv loaded and preprocessed.

Rows with missing values in critical betting columns (B365>2.5, B365<2.5) are dropped.

Categorical columns are converted to factors.

## Feature Engineering

1. Standardize Numerical Features

Center and scale betting odds and implied probability columns for consistent modeling.

2. Rolling Team History

Team-specific rolling stats computed separately for Home and Away games (last 10 matches):

Goals, Shots, Corners, Cards

## Derived metrics: goal differential, win rate

More recent matches are weighted more heavily.

## Dataset Merging

Home and Away rolling features are merged back into the full dataset.

Final dataset model_data includes match metadata, betting odds, and team form features.

## Train/Test Split

Chronological split based on date: training data before August 1, 2023; testing data from August 1, 2023 onward.

## Modeling: Full-Time Result (FTR)

Four models are trained using caret:

Logistic Regression : Baseline model for multiclass classification

LASSO: Performs feature selection

XGBoost: Gradient boosting model with 3-fold cross-validation

Random Forest: Bagged trees model with 5-fold cross-validation

## Evaluation

Predictions are made on the test set.

Accuracy is computed manually and compared via bar chart.

Confusion matrices show class-level performance.

## Modeling: Over/Under 2.5 Goals (OU_2.5)

Same pipeline as FTR is used, targeting the OU_2.5 column.

Logistic Regression, LASSO, XGBoost, and Random Forest models are trained and compared.

Bar chart summarizes accuracy across models.

## Outputs

Accuracy comparison plots for both FTR and OU models.

Confusion matrices for model diagnostics.
