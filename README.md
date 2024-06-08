<div align="center">

# Marketing Campaign Response Prediction Project

</div>

### CRISP-DM Methodology

CRISP-DM is like our roadmap for exploring and understanding data. It breaks down complex data projects into manageable steps, starting from understanding what the business needs are, all the way to deploying our solutions. It's super helpful because it keeps us organized and ensures we're on track with our goals. Plus, it's not just for one specific industry, so we can apply it to all sorts of projects, which is awesome for learning and building our skills. It's basically our guide to navigating the world of data mining!

<p align="center">
  <img width="400" height="400" src="figures/CRISP-DM.png">
</p>

*    Business Understanding: determine business objectives; assess situation; determine data mining goals; produce project plan
*    Data Understanding: collect initial data; describe data; explore data; verify data quality
*    Data Preparation (generally, the most time-consuming phase): select data; clean data; construct data; integrate data; format data
*    Modeling: select modeling technique; generate test design; build model; assess model
*    Evaluation: evaluate results; review process; determine next steps
*    Deployment: plan deployment; plan monitoring and maintenance; produce final report; review project (deployment was not required for this project)
---

### Objective  
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

# Model
from sklearn.linear_model import LogisticRegression
from yellowbrick.model_selection import RFECV

# Model Selection and Evaluation
from sklearn.model_selection import GridSearchCV, train_test_split
from yellowbrick.model_selection import CVScores, LearningCurve
from yellowbrick.classifier import DiscriminationThreshold, ClassPredictionError, PrecisionRecallCurve, ROCAUC
from sklearn import metrics 
import statsmodels.api as sm

# Save the model for deployment
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

| Metric       | Training Set | Test Set   |
|--------------|--------------|------------|
| Accuracy     | 79.2%        | 77.6%      |
| Precision    | 79.1%        | 38.2%      |
| Recall       | 79.5%        | 79.5%      |
| F1-score     | 79.3%        | 51.6%      |
| Specificity  | 79.0%        | 77.3%      |
| G-Mean       | 79.2%        | 78.4%      |

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
