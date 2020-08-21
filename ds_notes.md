## Softwware

### SPARK

## Data Collection and Cleaning

### NaN
    
  * Missing data. Reason for being missing is not necessarily known. Can be imputed, have case dropped, be turned into a feature, etc.
  
### Class Imbalance
  
  * Supervised learning models rely on outcomes (targets) to be balanced in distritubion. 

## Feature Engineering

### Transformation
  
  * Normalization of data
  * Scaling of data
  * Interpretation of categorical predictors (features)
   
### Combination

### Dimensionality Reduction
  
  * Principal Components Analysis (PCA)

## Feature Selection

### Preprocessing

    * Can check correlation of outcome (target) and predictor (feature) using various statistical tests including:
        * Peason's Correlation
        * Linea Discriminant Analysis
        * ANOVA
        * Chi-Square
        
### Model-Based

    * Models such as LASSO and RIDGE have a built in feature selection due to the regularization in the algorithms.

## Linear Models

### Linear Regression

#### Assumptions

  * Linearity: Relationship between outcome and predictor (feature and target) is linear. Can check with a scatter plot.
  * Low Multicollinearity: Predictors (features) are not highly correlated with one another. Can check with a correlation plot.
  * Homoscedasticity: Residuals have constant variance. No pattern to errors. Can check with scatter plot of errors (residual plot).
  * Error Terms have Normal Distribution: Errors follow normal distribution. More important for low sample sizes. Can check with QQ plot.
  * Autocorrelation: Residuals are indepenent of one another. More important in time series data. Can check with Durbin-Watson test.

#### Variants

##### LASSO and RIDGE Regressions

  * LASSO penalizes sum of absolute values of coeffients (L1 penalty)
  * RIDGE penalizes sum of squared coefficients (L2 penalty)

  * Can be used to reduce model complexity and prevent over-fitting.
  * Puts a constraint on the predictor (feature) coefficients which can address multi-collinearity.
  * LASSO preferred when small number of significant predictors (features) and others close to zero.
  * RIDGE preferred when large number of significant predictors (features) of similar importance.
  
### Logistic Regression

#### Assumptions

  * Binary outcome (target).
  * Low Multicollinearity: Predictors (features) are not highly correlated with one another. Can check with a correlation plot.
  * Linear relationship between logit of outcome (target) and each predictor (feature).
  * No influential values (extreme or outlier) in predictor (feature).

## Machine Learning Models

    * Will fall into one of two camps, Supervised Learning and Unsupervised Learning.
        * Supervised Learning such as classification and regression will take observations, be trained on a set of the observations, and make predictions on test observations.
        * Unsupervised Learning such as Clustering will take unlabeled observations.

### Decision Tree 

### Random Forest

#### Advantages

  * Fast, accurate, and easy to implement
  * Can handle large amounts of predictors (features)
  * Provides estimates of importantce of predictors (features)
  * Can handle missing data

#### Limitations
  
   * Can overfit
   * Estimates of categorical predictors (features) with higher number of levels can be biased and variable importance is not always reliable.

### Assumptions

  * Non-parameteric so no formal assumptions. 
  
#### Interpretation

  * ROC AUC will show performance based on True Positive Rate and False Positive Rate. Range between 0 and 1 and having more area under curve will be better.
  * Can use confusion matrix. False positives and false negatives are both bad, but cetain outcomes (targets) will need one or the other to be minimized (cancer detection is example).
  * Feature importance can be plotted to see what predictors (features) are showing the most impact on the the outcome (target).
    * Feature importance in a random forest model will indicate sum of reduction in GINI Impurity. GINI Impurity is probability of incorrect classification.

### K-Means

### KNN

### Neural Network

f

## Model Evaluation


