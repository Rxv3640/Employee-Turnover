import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline

df = pd.read_csv('HR_comma_sep.csv') '''Read dataset'''

df = df.dropna() '''Remove na values from dataset'''

correlation_matrix = df.select_dtypes(include='number').corr() '''Find the correlation between features in the dataset'''

'''sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

sns.histplot(df['satisfaction_level'])
plt.show()

sns.histplot(df['last_evaluation'])
plt.show()

sns.histplot(df['average_montly_hours'])
plt.show()

sns.barplot(x='sales', y='number_project', data=df, hue='left')
plt.legend()
plt.show()'''

X = df[['satisfaction_level', 'last_evaluation', 'left']].values

wcss = []

for i in range(1, 3):
  model = KMeans(n_clusters = i, n_init = 10, init = 'k-means++', random_state = 42)
  model.fit(X)
  wcss.append(model.inertia_)
plt.plot(range(1,3), wcss)
plt.title('K-means clustering of employees')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

df_categorical = df.select_dtypes(exclude='number')
df_numerical = df.select_dtypes(include='number')

df_encoded = pd.get_dummies(df_categorical)

df_new = pd.concat([df_numerical, df_encoded], axis=1)

X = df_new.drop('satisfaction_level', axis=1)
X = df_new.drop('last_evaluation', axis=1)

label = LabelEncoder()

y = df_new['satisfaction_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

y_train = label.fit_transform(y_train)

smote = SMOTE(random_state=123)
X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)

'''model = LogisticRegression(random_state=123)

k_fold_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5)

model = RandomForestClassifier(random_state=123)

k_fold_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5)

model = GradientBoostingClassifier(random_state=123)

k_fold_scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5)'''

pipeline = Pipeline([
  (('log_reg', LogisticRegression(random_state=123)),
    ('forest_classifier', RandomForestClassifier(random_state=123)),
  ('gradient_classifier', GradientBoostingClassifier(random_state=123)))
])

pipeline.fit(X_train_smote, y_train_smote)

y_pred_train_log_reg = pipeline.predict(X_train_smote)
y_pred_test_log_reg = pipeline.predict(X_test)

print(classification_report(y_test, y_pred_test_log_reg))

y_pred_test_prob_log_reg = pipeline.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_prob_log_reg)

roc_auc = auc(fpr,tpr)

youden_j = tpr - fpr
optimal_threshold_index = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_threshold_index]

plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred_test_log_reg)
print('Confusion matrix')
print(conf_matrix)


