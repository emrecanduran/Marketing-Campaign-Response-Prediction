## Marketing-Campaign-Classification

### Dataset description

- **`AcceptedCmp1`** - 1 if customer accepted the offer in the 1st campaign, 0 otherwise 
- **`AcceptedCmp2`** - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise 
- **`AcceptedCmp3`** - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise 
- **`AcceptedCmp4`** - 1 if customer accepted the offer in the 4th campaign, 0 otherwise 
- **`AcceptedCmp5`** - 1 if customer accepted the offer in the 5th campaign, 0 otherwise 
- **`Response (target)`** - 1 if customer accepted the offer in the last campaign, 0 otherwise 
- **`Complain`** - 1 if customer complained in the last 2 years
- **`DtCustomer`** - date of customer’s enrolment with the company
- **`Education`** - customer’s level of education
- **`Marital`** - customer’s marital status
- **`Kidhome`** - number of small children in customer’s household
- **`Teenhome`** - number of teenagers in customer’s household
- **`Income`** - customer’s yearly household income
- **`MntFishProducts`** - amount spent on fish products in the last 2 years
- **`MntMeatProducts`** - amount spent on meat products in the last 2 years
- **`MntFruits`** - amount spent on fruits products in the last 2 years
- **`MntSweetProducts`** - amount spent on sweet products in the last 2 years
- **`MntWines`** - amount spent on wine products in the last 2 years
- **`MntGoldProds`** - amount spent on gold products in the last 2 years
- **`NumDealsPurchases`** - number of purchases made with discount
- **`NumCatalogPurchases`** - number of purchases made using catalogue
- **`NumStorePurchases`** - number of purchases made directly in stores
- **`NumWebPurchases`** - number of purchases made through company’s web site
- **`NumWebVisitsMonth`** - number of visits to company’s web site in the last month
- **`Recency`** - number of days since the last purchase

### Overview
<p>It is asked to develop a prediction model in such a way that it will be possible for the Marketing Department of a retail company to predict which customers are likely to respond to a marketing campaing based on information from a previous campaign.</p>
<p>A response model can provide a significant boost to the efficiency of a marketing campaign by increasing responses or reducing expenses.</p>

Based on the evaluation metrics for both training and test sets, it appears that the best-performing model is the one with the following characteristics:

#### **Model**: Logistic Regression
##### **Performance Metrics**
*Train and Test results*:
- **Accucary:** 
        <p>*The model's accuracy on the training set is 79.2%, while on the test set, it is slightly lower at 77.6%. This indicates that the model performs fairly consistently across both datasets, with a slight drop in the test set, which is expected.*
- **Precision:** 
        <p>*Precision drops significantly from 79.1% on the training set to 38.2% on the test set. This suggests that the model is much better at correctly identifying positive instances on the training set than on the test set.*
- **Recall:**
        <p>*Recall remains constant at 79.5% for both the training and test sets. This consistency indicates that the model is equally effective at capturing true positive instances across both sets.*
- **F1-score:**
        <p>*The F1-score, which balances precision and recall, drops from 79.3% on the training set to 51.6% on the test set. This significant decrease aligns with the drop in precision, highlighting issues with the model's performance in identifying true positives accurately on the test set. This decrease in F1-score on the test set indicates a trade-off between precision and recall.*
- **Specificity:**
        <p>*Specificity, or the ability to correctly identify negative instances, is 79.0% on the training set and 77.3% on the test set. This minor decrease indicates the model's performance in recognizing negatives is relatively stable across both sets.*     
- **G-mean:**
        <p>*The geometric mean (G-Mean), which combines the true positive rate (recall) and the true negative rate (specificity), is 79.2% on the training set and 78.4% on the test set. This shows a balanced performance in identifying both positives and negatives, with only a slight reduction on the test set.*


##### **Feature Importance**
*Top 5 Features*:
- total_amount_spent
- total_campaigns_accepted
- NumWebVisitsMonth
- Recency
- family_size

*Significant Features*:
income_level and marital_status also show notable impacts on the target variable.

##### **Learning Curve**

The Logistic regression model's performance on both training and cross-validation data can be seen in the learning curve plot based on the weighted f1 score. The model performs better on the training set of data at first, but after an unusual start, the cross-validation scores improve slightly by following the same pattern with train scores. The two curves eventually converge at a specific point, signifying consistent performance on unseen data. This convergence shows that the model generalizes well without significant underfitting or overfitting. 

##### **ROC-AUC**
ROC AUC of 0.85 for both class 0 and class 1 indicates that the model has good discriminative power for both classes. The micro-average ROC AUC, if also around 0.85, confirms that the model performs well on average across all classes. This is the highest score among the models.


##### **Precision-Recall Curve**
This is a single-number summary of the precision-recall curve, calculated as the area under the PR curve (AUPRC). An AP of 0.55 means that, on average, the precision across different recall levels is 0.55. An average precision of 0.55 indicates a model with moderate performance in balancing precision and recall. A higher AP score indicates a better balance between precision and recall across all thresholds. AP with 0.55 is the best score among the other models. 


##### **Confusion Matrix**
With logistic regression model, we achieved the highest true positive rates(%11.9) and true negative rates(%65.70). Since the Ms. Sarah concerns about TPR(recall) than the other metrics, this model provides better results with respect to business needs. 

##### **Overall Assessment**
Tree-based models didn't generalize well in the data with the high difference between train and test scores. The Logistic Regression model demonstrates strong predictive performance with relatively low difference between train and test results, indicating a good generalization. It is achived to have best results possible with the most simplest model and less features. Only 7 features were created from 28 features. The feature importance analysis reveals several key factors influencing the target variable, including total_amount_spent, total_campaign_accepted, and Recency. 
