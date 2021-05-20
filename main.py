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
from sklearn.ensemble import RandomForestClassifier

# Use pandas to read the TSV file into a DataFrame
df = pd.read_csv("dataset_shopping.csv")

# Examine the dataset
# print (df.shape)
# print (df.head())
# print(df.info())
# print(df.describe())


# check for missing entries/null values and view non-numeric features
# print(df.isnull().sum())
# print(df['purchase'].value_counts())
# print(df['visitor'].value_counts())
# print(df['month'].value_counts())
# print(df.duplicated().value_counts())

# drop duplicated and reset dataset index
df.drop_duplicates(keep=False, inplace=True)
df.reset_index()

# Label encode non-numeric features/columns
le = LabelEncoder()
df.month = le.fit_transform(df.month)
df.visitor = le.fit_transform(df.visitor)
# print(df.head())


# ========================START SUPERVISED LEARNING========================

# Set up training and test
y = df['purchase'].values
X = df.drop('purchase', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ========================START CLASSIFIERS========================

# --------------------KNN----------------------
# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier()

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# --------------------KNN----------------------


# --------------------Linear Regressions----------------------
# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# print(reg_all.score(X_test, y_test))
# --------------------Linear Regressions----------------------


# --------------------Logistic Regressions----------------------
# Create the classifier: logreg
logreg = LogisticRegression(solver='liblinear')

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# --------------------Logistic Regressions----------------------


# --------------------SVC----------------------
# Create the classifier: SVC
svc = SVC()

# Fit the classifier to the training data
svc.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = svc.predict(X_test)

# Generate the confusion matrix and classification report
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# --------------------SVC----------------------


# --------------------DECISION TREE----------------------
# Create the classifier: DecisionTreeClassifier
dt = DecisionTreeClassifier()

# Fit the classifier to the training data
dt.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = dt.predict(X_test)

# Generate the confusion matrix and classification report
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
# --------------------DECISION TREE----------------------


# --------------------RandomForestClassifier----------------------
# Create the classifier: RandomForestClassifier
rf = RandomForestClassifier()

# Fit the classifier to the training data
rf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = rf.predict(X_test)

# Generate the confusion matrix and classification report
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# --------------------RandomForestClassifier----------------------

# ========================END CLASSIFIERS========================


def comparison(model):
    # Instantiate and fit a k-NN classifier to the unscaled data
    accuracy = model.fit(X_train, y_train)

    # Compute and print metrics
    # print('Accuracy for {} with accuracy score: {}'.format(model, accuracy.score(X_test, y_test)))
    return


def scaled_comparison(model):
    steps = [('scaler', StandardScaler()),
             ('model', model)]

    # Create the pipeline: pipeline
    pipeline = Pipeline(steps)

    # Fit the pipeline to the training set: knn_scaled
    scaled = pipeline.fit(X_train, y_train)

    # Compute and print metrics
    # print('Accuracy for {} with Scaling: {}'.format(model, scaled.score(X_test, y_test)))
    return


knn = KNeighborsClassifier()
lin_reg = LinearRegression()
log_reg = LogisticRegression(solver='liblinear')
svc = SVC()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

models = [knn, lin_reg, log_reg, svc, dt, rf]

for model in models:
    comparison(model)

for model in models:
    scaled_comparison(model)

# ========================START HYPERPARAMETER TUNING========================
# Commented out as running it can take a long time

# --------------------KNN----------------------
# Specify the hyperparameter space
parameters = {"n_neighbors": range(1, 10),
              "weights": ["uniform", "distance"],
              "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]}

# Instantiate the GridSearchCV object: gs
gs = GridSearchCV(KNeighborsClassifier(), parameters)

# Fit to the training set
# gs.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
# y_pred = gs.predict(X_test)

# Compute and print metrics
# print("Accuracy: {}".format(gs.score(X_test, y_test)))
# print(classification_report(y_test, y_pred))
# print("Tuned Model Parameters: {}".format(gs.best_params_))
# --------------------KNN----------------------


# --------------------Logistic Regressions----------------------
# Specify the hyperparameter space
parameters = {"penalty": ['l1', 'l2', 'elasticnet', 'none'],
              "multi_class": ['auto', 'ovr', 'multinomial'],
              "solver": ['lbfgs', 'liblinear', 'sag', 'saga', 'newton-cg']}

# Instantiate the GridSearchCV object: gs
gs = GridSearchCV(LogisticRegression(), parameters)

# Fit to the training set
# gs.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
# y_pred = gs.predict(X_test)

# Compute and print metrics
# print("Accuracy: {}".format(gs.score(X_test, y_test)))
# print(classification_report(y_test, y_pred))
# print("Tuned Model Parameters: {}".format(gs.best_params_))
# --------------------Logistic Regressions----------------------


# --------------------DECISION TREE----------------------
# Specify the hyperparameter space
parameters = {"max_features": ["auto", "sqrt", "log2"],
              "min_samples_leaf": range(1, 9),
              "criterion": ["gini", "entropy"],
              "splitter": ["best", "random"]}

# Instantiate the GridSearchCV object: gs
gs = GridSearchCV(DecisionTreeClassifier(), parameters)

# Fit to the training set
# gs.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
# y_pred = gs.predict(X_test)

# Compute and print metrics
# print("Accuracy: {}".format(gs.score(X_test, y_test)))
# print(classification_report(y_test, y_pred))
# print("Tuned Model Parameters: {}".format(gs.best_params_))
# --------------------DECISION TREE----------------------


# --------------------SVC----------------------
# Specify the hyperparameter space
parameters = {"C": range(1, 10),
              "kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
              "gamma": ['auto', 'scale']}

# Instantiate the GridSearchCV object: gs
gs = GridSearchCV(SVC(), parameters)

# Fit to the training set
# gs.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
# y_pred = gs.predict(X_test)

# Compute and print metrics
# print("Accuracy: {}".format(gs.score(X_test, y_test)))
# print(classification_report(y_test, y_pred))
# print("Tuned Model Parameters: {}".format(gs.best_params_))
# --------------------SVC----------------------


# --------------------RandomForestClassifier----------------------
# Setup the parameters and distributions to sample from: param_dist
parameters = {"n_estimators": [1, 10, 25, 50, 70, 100],
              "max_features": ["auto", "sqrt", "log2"],
              "min_samples_leaf": range(1, 9),
              "criterion": ["gini", "entropy"],
              "class_weight": ["balanced", "balanced_subsample"]}

# Instantiate the GridSearchCV object: gs
gs = GridSearchCV(RandomForestClassifier(), parameters)

# Fit to the training set
# gs.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
# y_pred = gs.predict(X_test)

# Compute and print metrics
# print("Accuracy: {}".format(gs.score(X_test, y_test)))
# print(classification_report(y_test, y_pred))
# print("Tuned Model Parameters: {}".format(gs.best_params_))
# --------------------RandomForestClassifier----------------------

# ========================END HYPERPARAMETER TUNING========================


#========================VISUALIZATIONS========================

# Label Distribution
sns.countplot(x="purchase", data=df)
#plt.show()


# Extract dataset of just True purchase values
purchase_true = df[df["purchase"] == True].reset_index()

# Count Plot of purchases by month
sns.countplot(x="month", data=purchase_true)
#plt.show()


# Correlation Heatmap
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True)
plt.title("Correlation")
#plt.show()



# Compare bounce_rate against exit_Rate on the complete dataset
sns.scatterplot(data=df, x="bounce_rate", y="exit_rate")
#plt.show()

# # Compare bounce_rate against exit_Rate on only the True purchase dataset
sns.scatterplot(data=purchase_true, x="bounce_rate", y="exit_rate")
#plt.show()



# Compute accuracy score against the number of neighbors for the KNN Classifier
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)


# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
#plt.show()