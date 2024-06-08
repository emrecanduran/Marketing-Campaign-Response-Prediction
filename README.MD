<div align="center">

# Marketing Campaign Response Prediction Project

</div>

### CRISP-DM Methodology

# CRISP-DM for Student Project Using Logistic Regression

## 1. Business Understanding
**Objective:** Define project goals and requirements from a business perspective.

- **Identify objectives:** Determine the primary goal.
- **Assess situation:** Review resources and constraints.
- **Data mining goals:** Translate objectives into data mining goals.
- **Project plan:** Outline steps and timeline.

**Example:** Goal to predict student performance based on study habits.

## 2. Data Understanding
**Objective:** Gather and explore the data.

- **Collect data:** Acquire relevant data.
- **Describe data:** Examine structure and quantity.
- **Explore data:** Identify patterns and relationships.
- **Verify quality:** Address missing values and inconsistencies.

**Example:** Collect data on study hours, grades, and attendance.

## 3. Data Preparation
**Objective:** Prepare the final dataset for modeling.

- **Select data:** Choose relevant data.
- **Clean data:** Handle missing values and outliers.
- **Construct data:** Create new features if needed.
- **Integrate data:** Combine data from different sources.
- **Format data:** Normalize or scale features.

**Example:** Clean study hours data, create average study time per week.

## 4. Modeling
**Objective:** Apply Logistic Regression and evaluate its performance.

- **Select techniques:** Logistic Regression.
- **Test design:** Use train-test split.
- **Build models:** Train the Logistic Regression model.
- **Assess models:** Evaluate performance metrics.

**Example:** Use Logistic Regression to predict performance.

### Goals  
This project aims to develop a prediction model for the Marketing Department of a retail company to predict which customers are likely to respond to a marketing campaign based on information from a previous campaign. A response model can significantly enhance the efficiency of a marketing campaign by increasing responses or reducing expenses. Product manager Sarah striving to optimize the company marketing campaigns. With a keen eye on metrics like recall(>0.75) and F1 score, Sarah ensures their campaigns reach a broad audience (recall) while maintaining precision in targeting (F1 score > 0.5).

---
 
### Dataset Description   

| **Feature**          | **Description**                                                              |
|----------------------|------------------------------------------------------------------------------|
| **AcceptedCmp1**     | 1 if customer accepted the offer in the 1st campaign, 0 otherwise            |
| **AcceptedCmp2**     | 1 if customer accepted the offer in the 2nd campaign, 0 otherwise            |
| **AcceptedCmp3**     | 1 if customer accepted the offer in the 3rd campaign, 0 otherwise            |
| **AcceptedCmp4**     | 1 if customer accepted the offer in the 4th campaign, 0 otherwise            |
| **AcceptedCmp5**     | 1 if customer accepted the offer in the 5th campaign, 0 otherwise            |
| **Response (target)**| 1 if customer accepted the offer in the last campaign, 0 otherwise          |
| **Complain**         | 1 if customer complained in the last 2 years                                 |
| **DtCustomer**       | Date of customer’s enrollment with the company                               |
| **Education**        | Customer’s level of education                                                |
| **Marital**          | Customer’s marital status                                                    |
| **Kidhome**          | Number of small children in customer’s household                             |
| **Teenhome**         | Number of teenagers in customer’s household                                  |
| **Income**           | Customer’s yearly household income                                           |
| **MntFishProducts**  | Amount spent on fish products in the last 2 years                            |
| **MntMeatProducts**  | Amount spent on meat products in the last 2 years                            |
| **MntFruits**        | Amount spent on fruit products in the last 2 years                           |
| **MntSweetProducts** | Amount spent on sweet products in the last 2 years                           |
| **MntWines**         | Amount spent on wine products in the last 2 years                            |
| **MntGoldProds**     | Amount spent on gold products in the last 2 years                            |
| **NumDealsPurchases**| Number of purchases made with discount                                      |
| **NumCatalogPurchases** | Number of purchases made using a catalog                                   |
| **NumStorePurchases**| Number of purchases made directly in stores                                 |
| **NumWebPurchases**  | Number of purchases made through the company’s website                       |
| **NumWebVisitsMonth**| Number of visits to the company’s website in the last month                  |
| **Recency**          | Number of days since the last purchase                                       |


## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project has following libraries:
```python

# Data Manipulation and Analysis
import pandas as pd
import numpy as np
import re
import collections
from datetime import datetime

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

# Data Preprocessing
import imblearn
import sklearn
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

# Machine Learning Model

from sklearn.linear_model import LogisticRegression
from yellowbrick.model_selection import RFECV

# Model Selection and Evaluation
from sklearn.model_selection import GridSearchCV, train_test_split
from yellowbrick.model_selection import CVScores, LearningCurve
from yellowbrick.classifier import DiscriminationThreshold, ClassPredictionError, PrecisionRecallCurve, ROCAUC
from sklearn import metrics 
import statsmodels.api as sm

# Save the model
import pickle
```

### Versions   
```
pandas version: 2.1.4
numpy version: 1.23.5
matplotlib version: 3.8.0
seaborn version: 0.13.2
scikit-learn version: 1.3.2
```   

### Performance Metrics

**Train and Test Results:**

- **Accuracy:** Training set - 79.2%, Test set - 77.6%.
- **Precision:** Training set - 79.1%, Test set - 38.2%.
- **Recall:** Training and Test sets - 79.5%.
- **F1-score:** Training set - 79.3%, Test set - 51.6%.
- **Specificity:** Training set - 79.0%, Test set - 77.3%.
- **G-Mean:** Training set - 79.2%, Test set - 78.4%.

### Feature Importance

Top 5 Features:
1. total_amount_spent
2. total_campaigns_accepted
3. NumWebVisitsMonth
4. Recency
5. family_size

Significant features also include income_level and marital_status.

### Learning Curve

The learning curve shows the Logistic Regression model's performance on both training and cross-validation data, with convergence indicating consistent performance on unseen data, suggesting good generalization without significant underfitting or overfitting.

### ROC-AUC

ROC AUC of 0.85 for both class 0 and class 1 indicates good discriminative power. The micro-average ROC AUC confirms that the model performs well on average across all classes.

### Precision-Recall Curve

An average precision (AP) of 0.55 indicates moderate performance in balancing precision and recall across all thresholds.

### Confusion Matrix

The Logistic Regression model achieved the highest true positive rates (11.9%) and true negative rates (65.70%). Given the importance of recall (TPR) for business needs, this model performs well in identifying true positives.

## 5. Evaluation
**Objective:** Assess the model to ensure it meets business objectives.

- **Evaluate results:** Check against success criteria.
- **Review process:** Ensure alignment with goals.
- **Next steps:** Decide on deployment or further refinement.

**Example:** Compare predictions to actual performance, decide on model deployment.

### Overall Assessment

Tree-based models did not generalize well, showing high differences between train and test scores. The Logistic Regression model demonstrates strong predictive performance with low differences between train and test results, indicating good generalization. It achieved the best results with the simplest model and fewer features. The feature importance analysis highlights key factors influencing the target variable, including total_amount_spent, total_campaigns_accepted, and Recency.
