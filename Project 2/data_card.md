# data_card.md — Titanic Dataset Card (Cleaned Version)

## 1) Overview
- **Purpose**: Supervised learning for predicting `survived` on the Titanic.
- **Granularity**: One row per passenger.
- **Target**: `survived` (0 = No, 1 = Yes).
- **Source**: Kaggle Titanic dataset (historical passenger manifest).

## 2) Features (after preprocessing)
| Feature                   | Type       | Description / Processing |
|----------------------------|------------|---------------------------|
| survived                  | int (0/1)  | Target variable. |
| pclass                    | int (1–3)  | Passenger class; treated as ordinal. |
| age                       | float      | Median-imputed; `age_missing` indicator added. |
| fare                      | float      | Ticket fare; skew handled with `log(fare+1)` if needed. |
| adult_male                | int (0/1)  | Indicator for whether passenger was an adult male. |
| alone                     | int (0/1)  | Indicator if passenger traveled alone. |
| familySize                | float      | `sibsp + parch + 1`. |
| sex_male                  | bool       | One-hot encoding of gender. |
| embark_town_Queenstown    | bool       | Embarked from Queenstown. |
| embark_town_Southampton   | bool       | Embarked from Southampton (Cherbourg dropped as reference). |
| who_man                   | bool       | Passenger identified as a man. |
| who_woman                 | bool       | Passenger identified as a woman. |
| class_Second              | bool       | Passenger in second class. |
| class_Third               | bool       | Passenger in third class. |
| age_missing               | int (0/1)  | 1 if age originally missing, else 0. |

## 3) Preprocessing Summary
- **Missingness**: Age→median imputation with `age_missing` flag; Embarked→mode; Deck dropped due to high missingness. 
- **Outliers**: Fare outliers detected with IQR & Z-score; corrected using `log(fare+1)`.
- **Encoding**: One-hot encoding for categorical variables (`sex`, `embarked_town`, `class`, `who`).
- **Scaling**: MinMax or Standard scaling applied to numerical features (`age`, `fare`) if required.
- **Feature Selection**: Compared correlation (numeric), Chi-square (categorical), and model-based importance (RandomForest).

## 4) Splits & Evaluation
- **Split**: Train/Validation with `stratify=survived`.
- **Metrics**: Accuracy, Precision/Recall, ROC-AUC; confusion matrix to examine minority-class errors.

## 5) Population vs Sample
The dataset reflects **historical Titanic passengers (1912)**. Results generalize only to **similar early 20th-century maritime settings**, not to modern contexts or broader populations.

## 6) Biases & Limitations (Short Bias Note)
The Titanic dataset encodes **historical inequalities**. Survival strongly depended on **gender, age, and class**, reflecting lifeboat policies and social norms rather than intrinsic merit. Third-class passengers and certain groups were underrepresented among survivors, while women and children were prioritized. As a result, trained models reproduce these historical patterns, not generalizable truths. In addition, missing data (e.g., Deck) and incomplete passenger records may bias outcomes. This sample is not representative of maritime passengers overall or modern safety protocols. Findings should be interpreted as **pattern recognition of a past event**, not predictive for contemporary or individual use.

## 7) Intended Use
- **Intended**: Education and demonstration of ML preprocessing, feature engineering, and evaluation.  
- **Not intended**: Deployment for real-world decision-making, risk scoring, or applications affecting people.

## 8) Versioning
- **File**: `clean_titanic_v1.csv`
- **Docs**: This `data_card.md` and `decision_log.md`. Maintain changelog when preprocessing is updated.
