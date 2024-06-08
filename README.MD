<div align="center">

# Marketing Campaign Response Prediction Project

</div>

### CRISP-DM Methodology

- Business understanding
- Data understanding
- Data preparation
- Modeling
- Evaluation

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
