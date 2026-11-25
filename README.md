# Corporate-Credit-Ratings
Predicting corporate credit ratings from financial ratios  Can a company's financial statements alone predict its credit rating? This project trains five machine learning classifiers on 25 accounting ratios to predict the ratings of US-listed companies, from AAA down to D.  Group project for FI505E Coding and Data Science for Finance (Project 14)
Data
Source: Corporate Credit Rating dataset on Kaggle (agewerc)
2,029 ratings of 593 US-listed companies, issued between 2010 and 2016
Ratings from five agencies: S&P, Moody's, Fitch, Egan-Jones and DBRS
25 financial ratios per rating, covering liquidity, profitability, leverage, efficiency and cash flow

Four original rating classes were almost empty (AAA had 7 ratings, CC 5, C 2, D 1), so the ratings are grouped into six ordered classes: AAA/AA, A, BBB, BB, B and CCC-D. The investment-grade line stays intact.

Method

Each company is rated several times, often with nearly identical ratios. A random train/test split lets a model recognise the company instead of learning credit risk. To prevent this, the data is split by company with StratifiedGroupKFold (stratified by rating, grouped by ticker). No company appears in both the training and test sets.

All preprocessing runs inside an imbalanced-learn pipeline, so it is fitted on training folds only:

Winsorise each ratio at the 1st and 99th percentiles of the training data, which caps obvious data errors (one firm's days sales outstanding was 115,962 days)
Standardise the ratios (SVM models only, since they are distance-based)
Oversample minority classes with SMOTE until every class matches the largest one
Tune hyperparameters with grouped 5-fold cross-validation, scored on macro F1

Five models are compared: Decision Tree, Random Forest, Gradient Boosting, Linear SVM and RBF SVM.

Results

Test set: 399 ratings from 119 companies the models never saw during training.

Model	Accuracy	Balanced accuracy	Macro F1	Within one class	Macro AUC
Random Forest	0.363	0.328	0.313	0.870	0.746
Gradient Boosting	0.341	0.296	0.290	0.832	0.681
Linear SVM	0.308	0.292	0.274	0.777	0.670
RBF SVM	0.308	0.268	0.253	0.799	0.672
Decision Tree	0.276	0.268	0.245	0.782	0.672
Always predict BBB	0.261	-	0.069	-	-

The Random Forest scored best on every metric. It placed 87% of predictions within one rating class of the true rating and separated investment grade from high yield correctly 78.7% of the time.

Findings
Financial ratios identify a firm's credit neighbourhood, but not its exact rating.
Cash generation and profitability matter most. By permutation importance on the test set, the top ratios were operating cash flow to sales, return on assets, free cash flow to operating cash flow, and gross margin.
Leakage is easy to miss. The same Random Forest scored 49.3% accuracy on a random split and 36.3% on the company-grouped split, so about 13 points of apparent accuracy came from recognising repeat companies.
Accuracy alone misleads on unbalanced data. Always predicting BBB scores 26% accuracy, so macro F1, balanced accuracy and within-one-class accuracy are better guides.
The agency matters. Adding the rating agency as a feature raised macro F1 from 0.29 to 0.34. In this sample, 74% of Egan-Jones ratings are investment grade, against 40% for S&P.
Tools

Python, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, seaborn, Google Colab

How to run
Download corporate_rating.csv from the Kaggle link above.
Open Project14_Corporate_Credit_Ratings.ipynb in Google Colab or Jupyter.
Put the CSV next to the notebook, or upload it when the notebook asks.
Run all cells.
