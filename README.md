#Classification with Logistic Regression

## Objective
Build a binary classifier using logistic regression to predict breast cancer diagnosis (Malignant vs. Benign).

## Dataset
The dataset `data.csv` contains Breast Cancer Wisconsin (Diagnostic) Data. The target variable is `diagnosis`, where `M` (Malignant) is mapped to `1` and `B` (Benign) is mapped to `0`. There are multiple continuous features extracted from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## Methodology
1. **Data Preprocessing**:
   - Irrelevant columns like `id` and any completely empty columns were dropped.
   - The target variable `diagnosis` was mapped to a binary format.
2. **Train/Test Split**: 
   - The data was divided into 80% for training and 20% for testing.
   - Stratified sampling was applied to maintain an even class distribution across sets.
3. **Feature Scaling**: 
   - Features were standardized to have a mean of 0 and a standard deviation of 1 using Scikit-Learn's `StandardScaler`. This scale normalization is crucial for Logistic Regression to converge appropriately.
4. **Modeling**: 
   - A `LogisticRegression` model was fitted on the scaled training data.
5. **Evaluation**:
   - The model was assessed on the test set using a variety of robust metrics yielding the following results:
     - **Precision**: ~0.9750
     - **Recall**: ~0.9286
     - **ROC-AUC**: ~0.9960

## Sigmoid Function & Threshold Tuning
Following standard evaluation, an experiment tuning the model prediction threshold was documented.

### The Sigmoid Function
The Logistic Regression model uses the sigmoid function `σ(z) = 1 / (1 + e^-z)` to map raw continuous predictor scores (`z`) into probabilities between 0 and 1. This mapped output fundamentally represents the probability that a data point belongs to the positive class (i.e. in this context, the likelihood of a tumor being Malignant).

### Tuning the Threshold
By default, the classification model employs a decision threshold of `0.5`. Altering this threshold adjusts the intrinsic tradeoff between Precision and Recall:
- **Decreasing threshold (e.g. to 0.3)**: Increases Recall, successfully identifying more true positive cases. This is extremely important in medical modeling where identifying illness is prioritized over having false alarms (lower Precision).
- **Increasing threshold (e.g. to 0.7)**: Increases Precision, guaranteeing higher confidence for every positive case predicted, but significantly diminishes Recall as it acts overly conservative and may miss patients.

## Output Assets
The following metrics and visual representations are saved within the project:
- `metrics.txt`: Summarized details on final evaluation scores and threshold tests.
- `confusion_matrix.png`: Breakdown of model identification success indicating True Positives/Negatives and False Positives/Negatives.
- `roc_curve.png`: A plotted True Positive Rate versus False Positive Rate summarizing ROC-AUC performance visually.
