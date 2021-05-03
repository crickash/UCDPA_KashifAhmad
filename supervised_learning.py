import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix, roc_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.decomposition import PCA


# Use pandas to read the TSV file into a DataFrame
df = pd.read_csv("dataset_shopping.csv")

print (df.shape)
print("============================================================")
print (df.head())
print("============================================================")
print(df.info())
print("============================================================")
print(df.describe())
print("============================================================")


print(df.isnull().sum())
print("============================================================")



label_distribution = sns.countplot(x="purchase", data=df)
plt.show()


print(df['purchase'].value_counts())
print(df['visitor'].value_counts())
print(df['month'].value_counts())
print("============================================================")

print(df.duplicated().value_counts())
print("============================================================")



df.drop_duplicates(keep = False, inplace = True)
df.reset_index()
print(df.shape)
print("============================================================")


print(df.isnull().values.any())


le = LabelEncoder()
df.month = le.fit_transform(df.month)
df.visitor = le.fit_transform(df.visitor)

print(df.head())


plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True)
plt.title("Correlation")
plt.show()


# Create arrays for the features and the response variable
y = df['purchase'].values
X = df.drop('purchase', axis=1).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)


# --------------------KNN----------------------
# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier()

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# --------------------KNN----------------------


# --------------------Linear Regressions----------------------
# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

print(reg_all.score(X_test, y_test))
# --------------------Linear Regressions----------------------



# --------------------Logistic Regressions----------------------
# Create the classifier: logreg
logreg = LogisticRegression(solver='liblinear')

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# --------------------Logistic Regressions----------------------



# --------------------SVC----------------------
# Create the pipeline: pipeline
svc = SVC()

# Fit the pipeline to the train set
svc.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = svc.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# --------------------SVC----------------------



def scaled_comparison(model):
    steps = [('scaler', StandardScaler()),
         ('model', model)]

    # Create the pipeline: pipeline
    pipeline = Pipeline(steps)

    # Fit the pipeline to the training set: knn_scaled
    scaled = pipeline.fit(X_train, y_train)


    # Compute and print metrics
    print('Accuracy for {} with Scaling: {}'.format(model, scaled.score(X_test, y_test)))
    return




def comparison(model):
    # Instantiate and fit a k-NN classifier to the unscaled data
    accuracy = model.fit(X_train, y_train)

    # Compute and print metrics
    print('Accuracy for {} with accuracy score: {}'.format(model, accuracy.score(X_test, y_test)))
    return



knn = KNeighborsClassifier()
lin_reg = LinearRegression()
log_reg = LogisticRegression(solver='liblinear')
svc = SVC()
dt = DecisionTreeClassifier()


models = [knn, lin_reg, log_reg, svc, dt]


for model in models:
    comparison(model)

print("============================================================")

for model in models:
    scaled_comparison(model)