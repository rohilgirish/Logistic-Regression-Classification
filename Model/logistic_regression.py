import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay

def main():
    # 1. Load dataset
    df = pd.read_csv('data.csv')
    
    # Drop columns that are completely null or irrelevant, like 'id'
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Target variable 'diagnosis': M (Malignant) -> 1, B (Benign) -> 0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    X = df.drop(columns=['diagnosis'])
    
    # Check for NaN columns (e.g. Unnamed: 32 which is common in this dataset) and drop them
    X = X.dropna(axis=1, how='all')
    y = df['diagnosis']

    # 2. Train/test split and standardize features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Fit a Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # 4. Evaluate with confusion matrix, precision, recall, ROC-AUC
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Write evaluation metrics and threshold tuning analysis to a text file
    with open('metrics.txt', 'w') as f:
        f.write("Logistic Regression Model Evaluation (Threshold = 0.5):\n")
        f.write("-" * 55 + "\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"ROC-AUC:   {roc_auc:.4f}\n\n")

        # 5. Tune threshold and explain sigmoid function
        f.write("5. Sigmoid Function Explanation\n")
        f.write("-------------------------------\n")
        f.write("Logistic regression uses the sigmoid function: sigma(z) = 1 / (1 + e^-z)\n")
        f.write("This function maps raw continuous scores (z) into probabilities between 0 and 1.\n")
        f.write("The output represents the probability that the observation belongs to the positive class (Malignant).\n\n")

        f.write("Threshold Tuning Experiment\n")
        f.write("---------------------------\n")
        for threshold in [0.3, 0.5, 0.7]:
            custom_preds = (y_pred_proba >= threshold).astype(int)
            prec = precision_score(y_test, custom_preds)
            rec = recall_score(y_test, custom_preds)
            f.write(f"Threshold = {threshold:.1f} | Precision = {prec:.4f}, Recall = {rec:.4f}\n")
            
        f.write("\nConclusion on Tuning threshold:\n")
        f.write("- Lowering the threshold (e.g. to 0.3) increases recall (catches more true positives) but decreases precision (more false positives).\n")
        f.write("  In medical domains like cancer prediction, high recall is often priority to ensure no patient who is sick is missed.\n")
        f.write("- Raising the threshold (e.g. to 0.7) increases precision (more confident in positive predictions) but decreases recall (might miss true positives).\n")

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign (0)', 'Malignant (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Threshold 0.5)')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png', bbox_inches='tight')
    plt.close()

    print("Execution completed successfully.")
    print(f"Metrics: Prec={precision:.4f}, Rec={recall:.4f}, AUC={roc_auc:.4f}")

if __name__ == "__main__":
    main()
