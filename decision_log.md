# decision_log.md — Titanic Preprocessing Decisions

**Project goal:** Prepare a clean, ML-ready Titanic dataset for predicting `Survived`, with every preprocessing choice tied to sound statistical and distributional reasoning that supports generalization.

---

## 1) Data Ingestion
- **Sources**: CSV (primary), JSON/Excel (supported for robustness). 
- **Action**: Loaded with `pandas.read_*` and verified schema using `.info()` and `.describe()`.
- **Reasoning**: Early schema checks surface type mismatches and missingness patterns, preventing silent data leakage or type coercion that can distort distributions.

---

## 2) Missing Values
### 2.1 Age
- **Action**: Impute with **median** age; add `Age_was_missing` indicator if needed.
- **Reasoning**: Age is **right-skewed**; the **median** is robust to outliers and preserves the central tendency without pulling toward high-fare, older extremes. The indicator preserves any predictive signal in the missingness mechanism (MNAR/MAR).

### 2.2 Embarked
- **Action**: Impute with **mode** (most frequent value).
- **Reasoning**: Embarked is nominal categorical. Mode imputation preserves the **empirical class proportions**, minimizing distortion of the marginal distribution and avoiding arbitrary category inflation.

### 2.3 Cabin
- **Action**: **Drop** due to high missingness; optionally engineer `Deck` from first letter when present and add `HasCabin` indicator.
- **Reasoning**: Extremely sparse features can inject noise and overfit. If retained, reduce dimension to stable summary (Deck) and track presence with an indicator so the model can exploit any strong signal without over-parameterization.

---

## 3) Outliers (Fare)
- **Detection**: 
  - **IQR rule**: values outside `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` flagged.
  - **Z-score**: absolute z-score > 3 flagged.
- **Treatment**: **Log-transform `Fare_log = log(Fare + 1)`**.
- **Reasoning**: Fare is heavy-tailed; log transform compresses extreme values, stabilizes variance, and makes linear decision boundaries more appropriate. Winsorization was considered but discarded for baseline to keep the **rank information** intact while reducing skew.

---

## 4) Categorical Encoding
- **Action**: One-hot encode `Sex`, `Embarked`, and engineered `Title` from `Name`; use `drop_first=True` to avoid perfect multicollinearity.
- **Reasoning**: One-hot preserves **category-level contrasts** without imposing ordinality. Dropping a reference level ensures model identifiability in linear models and reduces redundant dimensions.

---

## 5) Feature Engineering
- **Title** (from `Name`): standardize rare titles into `Rare` bucket.
  - **Reasoning**: Titles correlate with **social status and age/gender patterns**, capturing survival priors beyond raw `Sex`/`Age`.
- **FamilySize = SibSp + Parch + 1**; **IsAlone = 1{FamilySize==1}**.
  - **Reasoning**: Captures **social grouping effects** that influence access to lifeboats. `IsAlone` is a non-linear transformation that often exhibits stronger association than raw counts.
- **Optional**: `Pclass*Sex` interaction if using linear models.
  - **Reasoning**: Allows class effect to vary by gender, reflecting stratified rescue dynamics.

---

## 6) Scaling / Normalization
- **Action**: Scale numeric features (`Age`, `Fare_log`, `FamilySize`) with **MinMaxScaler** for tree-agnostic baselines or **StandardScaler** for linear models.
- **Reasoning**: Scaling improves numerical stability and optimizers’ convergence; tree models are scale-invariant but consistent scaling aids cross-model comparability.

---

## 7) Feature Selection
- **Statistical**: 
  - **Numerical**: Pearson correlation with target.
  - **Categorical**: **Chi-square** test on one-hot features.
- **Model-based**: **RandomForest feature importance**.
- **Decision rule**: Keep features consistently strong across methods; review discordant features for redundancy or instability.
- **Reasoning**: Combining univariate (correlation/chi2) with multivariate (model-based) guards against spurious associations and highlights interactions captured by trees.

---

## 8) Data Splitting & Leakage
- **Action**: Train/validation split with `stratify=Survived` and `random_state` fixed. Fit imputers/scalers **only on train**, then apply to validation/test.
- **Reasoning**: Stratification preserves class balance; fitting transformations on train only prevents **target leakage** and preserves out-of-sample validity.

---

## 9) Generalization & Bias
- **Population vs Sample**: The dataset is a **sample of Titanic passengers**, not modern maritime populations. Model learns **historical, context-specific** rescue patterns.
- **Sampling/Selection Bias**: Over-representation of certain classes/sex/ages among survivors introduces class-conditional imbalance; report metrics beyond accuracy (e.g., recall, ROC-AUC).

---

## 10) Reproducibility
- **Seeds set**; versions pinned; code kept simple and linear. Outputs saved as `clean_titanic_v1.csv` with a clear data card.