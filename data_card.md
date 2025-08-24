# data_card.md — Titanic Dataset Card (Cleaned Version)

## 1) Overview
- **Purpose**: Supervised learning for predicting `Survived` on the Titanic.
- **Granularity**: One row per passenger.
- **Target**: `Survived` (0 = No, 1 = Yes).
- **Source**: Kaggle Titanic dataset (historical passenger manifest).

## 2) Features (after preprocessing)
| Feature        | Type       | Description / Processing |
|----------------|------------|---------------------------|
| Pclass         | int (1–3)  | Passenger class; left as numeric (treated as ordinal). |
| Sex_*          | one-hot    | From `Sex`; reference dropped to avoid multicollinearity. |
| Age            | float      | Median-imputed; optionally `Age_was_missing` indicator. |
| SibSp          | int        | Siblings/spouse count. |
| Parch          | int        | Parents/children count. |
| FamilySize     | int        | `SibSp + Parch + 1`. |
| IsAlone        | int (0/1)  | 1 if `FamilySize==1`, else 0. |
| Fare_log       | float      | `log(Fare + 1)` to reduce skew/outliers. |
| Embarked_*     | one-hot    | Mode-imputed; reference dropped. |
| Title_*        | one-hot    | Extracted from Name; rare titles binned. |
| (optional) Deck / HasCabin | categorical/indicator | Derived from Cabin when present; Cabin dropped due to high missingness. |

## 3) Preprocessing Summary
- **Missingness**: Age→median, Embarked→mode, Cabin→dropped (sparse) with optional engineered summaries. 
- **Outliers**: Fare detected via IQR & z-score; corrected with `log(Fare+1)`.
- **Encoding**: One-hot for nominal categories (`Sex`, `Embarked`, `Title`).
- **Scaling**: MinMax or Standard scaling applied to numeric features as needed.
- **Feature Selection**: Correlation (numerical), Chi-square (categorical), and RandomForest importances used to compare and select stable features.

## 4) Splits & Evaluation
- **Split**: Train/Validation with `stratify=Survived`.
- **Metrics**: Accuracy, Precision/Recall, ROC-AUC; use confusion matrix to inspect minority errors.

## 5) Population vs Sample
- Dataset reflects **historical passengers** on the RMS Titanic (1912). Findings generalize to **similar contexts** (shipboard evacuation under early 20th-century norms), not universally to modern scenarios.

## 6) Biases & Limitations (Short Bias Note)
The Titanic dataset encodes the **social and structural conditions** of 1912 travel. Survival depended strongly on **sex, age, and class**, reflecting lifeboat policies and social norms rather than intrinsic merit. Third-class and certain nationalities were underrepresented among survivors, and children and women were prioritized. These patterns can yield models that appear highly accurate but simply reproduce historical inequities. Because the data come from a specific event with incomplete records (e.g., missing cabins) and potential reporting errors, estimates may be biased. The sample is not representative of broader maritime populations or modern safety practices, so deploying such a model beyond educational settings risks **misinterpretation and unfair inferences**. Results should be framed as **historical pattern recognition**, not causal conclusions or predictors of individual worth or modern outcomes.

## 7) Intended Use
- **Intended**: Education, demonstration of ML preprocessing and evaluation.
- **Out of scope**: Real-world deployment, risk scoring, or any decision affecting people.

## 8) Versioning
- **File**: `clean_titanic_v1.csv`
- **Docs**: This `data_card.md` and `decision_log.md`. Keep a changelog when preprocessing choices change.