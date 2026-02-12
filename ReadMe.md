# Estimation of Obesity Levels Using Machine Learning

## Problem Statement

Obesity is a major public health concern influenced by eating habits, physical activity, and lifestyle factors. Early identification of obesity levels can help in preventive healthcare and awareness programs.


---

## Dataset Description

* **Dataset Name:** Estimation of Obesity Levels Based on Eating Habits and Physical Condition
* **Source:** UCI Machine Learning Repository
* **Number of Instances:** 2111
* **Number of Features:** 17
* **Target Variable:** `NObesity` (7 obesity categories)

### Obesity Classes / categories

* Insufficient_Weight
* Normal_Weight
* Overweight_Level_I
* Overweight_Level_II
* Obesity_Type_I
* Obesity_Type_II
* Obesity_Type_III

The dataset contains a mix of **numerical and categorical features** related to eating habits, physical activity, and lifestyle. No missing values are present.

---

## Models Used and Evaluation Metrics

The following six classification models were trained and evaluated on the same dataset:

1. Logistic Regression (Multinomial)
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier (Ensemble)
6. XGBoost Classifier (Ensemble)

### Evaluation Metrics

For each model, the following metrics were computed:

* **Accuracy**
* **AUC Score** (One-vs-Rest, Macro Average)
* **Precision** (Macro Average)
* **Recall** (Macro Average)
* **F1 Score** (Macro Average)
* **Matthews Correlation Coefficient (MCC)**

### Comparison Table

| ML Model            | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| ------------------- | -------- | --- | --------- | ------ | -------- | --- |
| Logistic Regression|	0.8747|	0.9834|	0.8707|	0.8724|	0.8708|	0.8539|
| Decision Tree|	0.9149|	0.9488|	0.9157|	0.9118|	0.9133|	0.9007|
| K-Nearest Neighbors|	0.8251|	0.965|	0.8235|	0.8187|	0.8131|	0.7981|
| Gaussian Naive Bayes|	0.5083|	0.8372|	0.5207|	0.5002|	0.4511|	0.4471|
| **Random Forest**|**0.9409**|**0.9949**|**0.9464**|**0.9395**|**0.9412**|**0.9316**|
| **XGBoost**|**0.9527**|**0.9965**|**0.9539**|**0.9512**|**0.9517**|**0.9451**|



> **Note:** AUC scores are computed using the One-vs-Rest (OvR) strategy with macro averaging due to the multiclass nature of the problem.

---

## Model Performance Observations

| ML Model            | Observation about model performance |
| ------------------- | -------- | 
| Logistic Regression  | Competitive linear baseline (Accuracy=0.875, macro-F1=0.871): Per-class performance is solid for some classes but worse where decision boundaries are nonlinear.  linear decision surfaces work when features are approximately linearly separable after scaling/encoding; they cannot capture complex interactions that tree ensembles do. May be good lightweight baseline and calibrated probabilities; combine with feature engineering if needed. |
| Decision Tree        | Good single-tree performance (Accuracy=0.915, macro-F1=0.913): Per-class scores indicate useful splits but higher variance vs ensembles.  A single tree can model nonlinearities but tends to over-fit particular branches and is sensitive to small data changes. May be useful for interpretation and quick prototyping; prefer ensembles for stable deployment. |
| K-Nearest Neighbors  | Moderate performance (Accuracy=0.825, macro-F1=0.813): Per-class results show reasonable detection of some labels but lower scores for overlapping classes.  KNN is sensitive to high-dimensional noise, class imbalance and local density differences despite scaling; it effectively memorizes local neighborhoods rather than learning global structure. May be consider only for small-scale or instance-based use-cases; Need to tune k or use metric learning / feature selection to improve it. |
| Naive Bayes          | Weakest overall (Accuracy=0.508, macro-F1=0.451): Extremely uneven per-class F1s (one class Obesity_Type_III:0.992 detected very well while all others are poor).  The strong conditional independence and Gaussian assumptions do not hold here — features are mixed categorical/continuous and classes overlap; NB therefore underfits complex dependencies. May be avoid NB as a main model for this dataset unless used with targeted feature transformations or class‑specific models.  |
| Random Forest        | Strong ensemble baseline (Accuracy=0.941, macro-F1=0.941): High per-class F1s and moderate dispersion show robust generalization.  Collection of many non-correlated trees reduces variance and is resilient to noisy or partially informative features. May be excellent trade-off of accuracy, latency and ability to interpret. |
| XGBoost              | Performer (Accuracy=0.953, macro-F1=0.952): Per-class F1s are consistently high with low dispersion, indicating reliable discrimination across labels.  gradient-boosted trees capture complex nonlinear interactions and handle heterogeneous features well; regularization and ensemble averaging reduce overfitting seen in single trees. May be prefer XGBoost when inference cost and feature stability permit. |

---


## Conclusion

This project demonstrates an end-to-end machine learning workflow, from data preprocessing and model evaluation to interactive visualization and cloud deployment. Ensemble methods, particularly XGBoost and Random Forest, show superior performance in predicting obesity levels from lifestyle and physical condition data.
