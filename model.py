import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pyexpat import features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/creditcard.csv")
print (df.head())
df.describe()
df.shape
               # Percentage of missing values in each column
round(100 * (df.isnull().sum()/len(df)),2).sort_values(ascending=False)
                # Percentage of missing values in each row
round(100 * (df.isnull().sum(axis=1)/len(df)),2).sort_values(ascending=False)
                     # Checking duplicates
df_d=df.copy()
df_d.drop_duplicates(subset=None, inplace=True)
df.shape
df_d.shape #Duplicates are found
## Assigning removed duplicate(df_d) to  original data (df)
df=df_d
df.shape
del df_d
                           # EDA
df.info()

def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(16,16))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='indigo')
        ax.set_title(feature+" Distribution",color='black')
        ax.set_yscale('log')
    fig.tight_layout()
    plt.show()
draw_histograms(df,df.columns,8,4)

df.Class.value_counts()
 #Bar Plot
ax=sns.countplot(x='Class',data=df, color='r');
ax.set_yscale('log')
 #Correlation Matrix
plt.figure(figsize = (50,20))
sns.heatmap(df.corr(), annot = True, cmap="viridis")
plt.show()

   #LOGISTIC REGRESSION MODEL
df.info()
# Dropping non-important feature 'Time'
estimators=[ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

X1 = df[estimators]
y = df['Class']

Col=X1.columns[:-1]
Col

X = sm.add_constant(X1)
reg_logit = sm.Logit(y,X)
results_logit = reg_logit.fit()
results_logit.summary()

#Feature Selection: Backward elimination (P-value approach)
""" Takes in the dataframe, the dependent variable and a list of column names, runs the regression repeatedly eleminating feature with the highest
    P-value above alpha one at a time and returns the regression summary with all p-values below alpha"""


import statsmodels.api as sm

def back_feature_elem(X, y, Col, alpha=0.05):
    Col_list = list(Col)  # convert Index to list

    while True:
        X_1 = sm.add_constant(X[Col_list])
        model = sm.Logit(y, X_1).fit(disp=False)
        p_values = model.pvalues.iloc[1:]  # exclude intercept

        max_p = p_values.max()
        feature_to_remove = p_values.idxmax()

        if max_p > alpha:
            Col_list.remove(feature_to_remove)  # remove feature with highest p-value
        else:
            break

    # return both the final model and the selected columns
    final_model = sm.Logit(y, sm.add_constant(X[Col_list])).fit(disp=False)
    return Col_list, final_model

# use it
selected_features, final_model = back_feature_elem(X, df["Class"], Col, alpha=0.05)

print("Selected Features:", selected_features)
print(final_model.summary())

# Interpretation: Odds ratios + Confidence Intervals
params = np.exp(final_model.params)
conf = np.exp(final_model.conf_int())
conf['OR'] = params
pvalue = round(final_model.pvalues, 3)
conf['pvalue'] = pvalue
conf.columns = ['CI 95%(2.5%)', 'CI 95%(97.5%)', 'Odds Ratio', 'pvalue']

print(conf)

# Select features and target
new_features = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
                     'V20','V21', 'V22', 'V23', 'V25', 'V26', 'V27','Class']]

x = new_features.iloc[:, :-1]   # Features
y = new_features.iloc[:, -1]    # Target

# Train-test split (with stratification to balance class distribution)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)
print("Class distribution in training set:\n", y_train.value_counts(normalize=True))
print("Class distribution in test set:\n", y_test.value_counts(normalize=True))

                       # Logistic Regression Model
# --------------------------
# Class weight helps deal with imbalance
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(x_train, y_train)

y_pred_lr = log_reg.predict(x_test)
y_prob_lr = log_reg.predict_proba(x_test)[:,1]

print("=== Logistic Regression Results ===")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob_lr))

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_lr)
# Save trained model
joblib.dump(log_reg, "log_reg.pkl")

# Save feature names used in training
joblib.dump(x_train.columns.tolist(), "model_features.pkl")

# Save model accuracy
joblib.dump(accuracy, "model_accuracy.pkl")

print("Model and features saved successfully!")

# Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, log_reg.predict_proba(x_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={roc_auc:.2f})")
plt.plot([0,1], [0,1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()



