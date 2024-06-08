<div align="center">

# Marketing Campaign Response Prediction Project

</div>

### CRISP-DM Methodology

## 1. Business Understanding
**Objective:** To understand the project objectives and requirements from a business perspective and then convert this knowledge into a data mining problem definition and a preliminary plan.

- **Identify business objectives:** Define the primary goal of the project from the business perspective.
- **Assess situation:** Inventory of resources, constraints, assumptions, and other factors.
- **Determine data mining goals:** Translate business objectives into data mining goals.
- **Produce project plan:** Outline the steps and timeline to achieve the goals.

**Example:** For a retail company, the business objective might be to increase sales. A corresponding data mining goal could be to develop a model that predicts which products customers are likely to purchase.

## 2. Data Understanding
**Objective:** To collect, describe, and explore the data to gain insights and identify data quality issues.

- **Collect initial data:** Acquire the data listed in the project resources.
- **Describe data:** Examine the data's surface properties, such as structure, format, and quantity.
- **Explore data:** Perform exploratory data analysis (EDA) to identify patterns, trends, and relationships.
- **Verify data quality:** Assess data quality and address any issues such as missing values, outliers, or inconsistencies.

**Example:** For the retail company, this phase would involve collecting sales data, customer demographics, and transaction histories, then analyzing this data to identify patterns and anomalies.

## 3. Data Preparation
**Objective:** To construct the final dataset from the initial raw data for modeling.

- **Select data:** Choose the data relevant to the problem.
- **Clean data:** Handle missing values, noise, and outliers.
- **Construct data:** Derive new attributes, such as aggregating data or generating new features.
- **Integrate data:** Combine data from different sources if necessary.
- **Format data:** Reformat data for modeling, such as normalizing or scaling features.

**Example:** The retail company might clean the data by filling in missing values for customer demographics, create new features like total spending, and normalize the transaction amounts.

## 4. Modeling
**Objective:** To select and apply various modeling techniques and calibrate their parameters to optimal values.

- **Select modeling techniques:** Choose appropriate algorithms for the problem (e.g., decision trees, neural networks).
- **Generate test design:** Decide on the train-test split or cross-validation method.
- **Build models:** Train the models using the prepared data.
- **Assess models:** Evaluate models based on accuracy, precision, recall, etc.

**Example:** The retail company could try different models, such as logistic regression or random forests, to predict customer purchases, and evaluate these models using cross-validation.

## 5. Evaluation
**Objective:** To thoroughly evaluate the model(s) to ensure they meet the business objectives.

- **Evaluate results:** Assess the models against the business success criteria.
- **Review process:** Review the steps executed to ensure they align with the business objectives.
- **Determine next steps:** Decide whether to deploy the model, iterate with more data, or explore alternative models.

**Example:** The retail company would compare the model predictions with actual sales data to determine accuracy and decide whether the model's predictions are useful for increasing sales. If the model performs well, it might be deployed in the business operations; otherwise, further refinement may be needed.

## Summary
These phases ensure a structured approach to data science projects, aligning technical work with business goals and systematically transforming raw data into actionable insights and predictions. This methodology promotes a clear, repeatable process, increasing the likelihood of successful outcomes.

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
