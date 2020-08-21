# Outline

## Softwware

### SPARK

## Data Collection and Cleaning

### NaN
    
  * Missing data. Reason for being missing is not necessarily known. Can be imputed, have case dropped, be turned into a feature, etc.
  
### Class Imbalance
  
  * Supervised learning models rely on outcomes (targets) to be balanced in distritubion. 
  * Can collect more data
  * Can resample
  * Can try model that is not negatively impacted by this imbalance

## Feature Engineering

### Transformation
  
  * Normalization of data
  * Scaling of data
  * Interpretation of categorical predictors (features)
   
### Combination

  * Features may be more valuable and predictive when combined with other features (counts, averages, etc.)

### Dimensionality Reduction
  
  * Principal Components Analysis (PCA) will reduce predictors (features) using linear combinations
  * A covariance matrix of the preictor variables is generated
  * The sum of the diagnols of this covariance matrix is used to computed eigenvalues and eigenvectors
  * These eigenvectors then represent the principal components and the eigenvalues are used to find the proportion of the total variance explained by the components
  * A scree graph can show the proportion of variance explained by each additional eigenvalue
  * Once chosen, PCs can be interpreted based on which variables they are most correlated with in either direction

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

#### Advantages

  * Simple and easy to use/understand even with small amount of data
  * Helpful to start with

#### Disadvantages

  * Unstable and small change in data can cause large change in result
  * Relatively inaccurate compared to other models
  * Predictors (features) with high levels are favored
  * Can get complex

#### Interpretation

  * Can use accuracy, precision, recall, and F1 to evaluate model with classification

### Random Forest

#### Advantages

  * Fast, accurate, and easy to implement
  * Can handle large amounts of predictors (features)
  * Provides estimates of importantce of predictors (features)
  * Can handle missing data

#### Limitations
  
   * Can overfit
   * Estimates of categorical predictors (features) with higher number of levels can be biased and variable importance is not always reliable.

#### Assumptions

  * Non-parameteric so no formal assumptions. 
  
#### Interpretation

  * ROC AUC will show performance based on True Positive Rate and False Positive Rate. Range between 0 and 1 and having more area under curve will be better.
  * Can use confusion matrix. False positives and false negatives are both bad, but cetain outcomes (targets) will need one or the other to be minimized (cancer detection is example).
  * Feature importance can be plotted to see what predictors (features) are showing the most impact on the the outcome (target).
    * Feature importance in a random forest model will indicate sum of reduction in GINI Impurity. GINI Impurity is probability of incorrect classification.

### K-Means

  * Unsupervised learning 
  
#### Advantages

  * Simple and will converge
  * Scales with large data
  * Easily adaptable

#### Limitations

  * Have to choose K manually
  * Dependent on initial values
  * Different size and density of clusters can cause issues
  * Does not handle outliers well
  * Does not handle large numbers of predictors (features)

#### Assumptions

  * Clusters will be spherical
  * Clusters are of similar size

#### Interpretation

  * Select the cluster size (K) using elbow method (abrupt change in SSE)

### KNN

  * Non-parametric supervised learning technique used for classification
  
#### Advantages

  * Easy with minimal assumptions needed
  * Can be used for classification and regression
  * Can work for multic-lass problems (outcome has more than two values)

#### Limitations

  * Cost and time
  * Sensitive to scale of data
  * Issues when outcome (target) has certain values at low proportion (rare event target variable)
  * Issues when high number of predictors (features)
  
#### Assumptions

  * Predictors (features) are standardized to prevent one influencing the distance calculation.
  * Outliers are more important at low k-values as the distance calculation is more sensitive at these.
  
#### Interpretation

  * ROC can be used to evaluate what k-value to use. 
  * Can use accuracy, precision, recall, and F1 to evaluate model with classification
  
### Naive Bayes

#### Advantages
#### Limitations
#### Assumptions
#### Interpretation

## Model Evaluation

  * Confusion Matrix to see the True Negatives, True Positives, False Negatives, and False Positives
  * Accuracy of a model is the proportion of correctly predicted observations. The sum of True Positives and True Negatives divided by the number of observations. Higher is better.
  * Precision is the proportion of correctly predicted positive observations. The True Positives divided by all the Positives. 
  * The F1 score is the weighted average of both of these. 
  * Depending the model, outcome, and balance of outcomes, a model should prioritize some of these. F1 is good if the outcome is not balanced. If false positives and false negatives are similarly important, Accuracy is useful. If false positives and false negatives are different, then looking at both Accuracy and Precision is useful.
  
# Questions

## Bias vs. Variance Tradeoff

  * Bias will be simplifying assumptions made by model
  * Variance error will be amount estimates of outcome (target) will change if different data is used
  * High Bias, Low Variance Will be consistent, but inaccurate on average
  * High Variance, Low Bias will be accurate on average, but inconsistent
  * Need to make balance between underfitting by simplifying model too much and overfitting by memorizing noise instead of signal.
  
## Bayes Theorem

  * Bayes Theorem will give posterior probability of event given prior knowledge.
  * True Positive Rate divided by sum of False Positive Rate and True Positive Rate
  
## Type 1 vs. Type 2 Error

  * Type 1 Error is False Positive (Predict man is pregnant)
  * Type 2 Error is False Negative (Predict pregnant woman is not pregnant)
  
## Cross-Validation on Time-Series

  * Have to account for autocorrelation of data
  * Forward chain to model on past data and look at forward facing data
  
## Regression vs. Classification

  * Regression gives continous results while classification gives discrete values and categories (Is a name correlated with male/female vs. Is a name male or female)
  
## When to use ensemble model

  * Want to reduce overfitting and make model more robust
  * Combination and stacking of models
  
