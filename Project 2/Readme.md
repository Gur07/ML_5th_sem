# Titanic Dataset - Preprocessing and Data Card

## 📌 Original Dataset Schema
The Titanic dataset (as provided by Seaborn/Kaggle) contains the following columns:

| Column       | Non-Null Count | Dtype   |
|--------------|----------------|---------|
| survived     | 891 non-null   | int64   |
| pclass       | 891 non-null   | int64   |
| sex          | 891 non-null   | object  |
| age          | 714 non-null   | float64 |
| sibsp        | 891 non-null   | int64   |
| parch        | 891 non-null   | int64   |
| fare         | 891 non-null   | float64 |
| embarked     | 889 non-null   | object  |
| class        | 891 non-null   | object  |
| who          | 891 non-null   | object  |
| adult_male   | 891 non-null   | bool    |
| deck         | 203 non-null   | object  |
| embark_town  | 889 non-null   | object  |
| alive        | 891 non-null   | object  |
| alone        | 891 non-null   | bool    |

---

## 📌 Final Preprocessed Dataset Schema
After preprocessing and feature engineering, the dataset schema was transformed to:

| Column                   | Non-Null Count | Dtype   |
|---------------------------|----------------|---------|
| survived                 | 891 non-null   | int64   |
| pclass                   | 891 non-null   | int64   |
| age                      | 891 non-null   | float64 |
| fare                     | 891 non-null   | float64 |
| adult_male               | 891 non-null   | int64   |
| alone                    | 891 non-null   | int64   |
| familySize               | 891 non-null   | float64 |
| sex_male                 | 891 non-null   | bool    |
| embark_town_Queenstown   | 891 non-null   | bool    |
| embark_town_Southampton  | 891 non-null   | bool    |
| who_man                  | 891 non-null   | bool    |
| who_woman                | 891 non-null   | bool    |
| class_Second             | 891 non-null   | bool    |
| class_Third              | 891 non-null   | bool    |
| age_missing              | 891 non-null   | bool    |

---

## ⚖️ Bias & Representativeness Note
The Titanic dataset, while widely used for machine learning demonstrations, is not fully representative of broader populations. It reflects the passenger demographics of a transatlantic voyage in 1912, with strong biases toward European and North American travelers, and a clear overrepresentation of wealthier social classes. Women, children, and higher-class passengers had higher survival rates due to evacuation policies, which may introduce gender and socioeconomic biases in predictive modeling. Additionally, missing values in features like **age**, **deck**, and **embark_town** reflect uneven record-keeping, often skewed by passenger status. Therefore, models trained on this dataset capture historical and social inequalities specific to this event and should not be generalized to modern survival prediction contexts.

---

## 📂 Usage
This dataset (with preprocessing) is prepared for machine learning tasks such as:
- Binary classification (survival prediction).
- Feature engineering practice.
- Bias and fairness analysis.

