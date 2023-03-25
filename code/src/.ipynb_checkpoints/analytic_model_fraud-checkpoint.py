''' Libraries '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preparation.preparation import load_file_card
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import dataframe_image as dfi

''' Defined Load File '''
model_data = load_file_card().copy()


''' Preparation data for run analytic model '''
x = model_data.drop("fraud", axis = 1).values
y = model_data["fraud"].values

smote = SMOTE(random_state=39)
non_fraud_over, fraud_over = smote.fit_resample(x, y)

non_fraud_over_df = pd.DataFrame(non_fraud_over, columns=["distance_from_home", "distance_from_last_transaction",
    "ratio_to_median_purchase_price", "repeat_retailer", "used_chip",
    "used_pin_number", "online_order"])
non_fraud_over_df

non_fraud_over_df["fraud"] = fraud_over
df3 = non_fraud_over_df

feature_columns = ["distance_from_home", "distance_from_last_transaction",
"ratio_to_median_purchase_price", "repeat_retailer", "used_chip", "used_pin_number", "online_order"]

X_smote = df3[feature_columns]
y_smote = df3.fraud

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.2, random_state=39)

def run():
    graph_raw_data()
    status_fraud()
    model()
 
def graph_raw_data():
    ''' Exploration Data Analysis raw data '''

    for column in [0, 1, 2]:
        model_data.iloc[:, column] = np.log10(load_file_card().iloc[:, column])

    numerical_columns = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price"]
    for column in numerical_columns:
        plt.figure()
        plot = model_data[column]
        sns.histplot(plot, bins=10, kde=True)
        plt.savefig(f"./src/static/EDA_{column}.png")


def status_fraud():
    html1 = f"No Frauds:  {round(model_data['fraud'].value_counts()[0]/len(model_data) * 100,2)} % of the dataset"
    html2 = f"Frauds:  {round(model_data['fraud'].value_counts()[1]/len(model_data) * 100,2)} % of the dataset"
    return html1, html2

def model():
    
  ''' Model Logistic Regresion '''
  logreg = LogisticRegression(max_iter=200)
  logreg.fit(X_train_smote, y_train_smote)

  y_pred_logreg_smote = logreg.predict(X_test_smote)
  print("Accuracy of logistic regression classifier on test set: {:.5f}".format(logreg.score(X_test_smote, y_test_smote)))

  ''' Confusion matrix Logistic Regresion '''
  confusion_matrix_logreg = confusion_matrix(y_test_smote, y_pred_logreg_smote)
  print(f"Confusion matrix Logistic Regression: {confusion_matrix_logreg}")

  plt.figure(figsize = (12, 6))

  sns.heatmap(confusion_matrix_logreg, annot = True, cmap = "hot")
  plt.xlabel("Prediction")
  plt.ylabel("Reality")
  plt.savefig("./src/static/confusion_matrix_reg_log.png")

  ''' Classification report Logistic Regression '''
  print(f"classification report RL: {classification_report(y_test_smote, y_pred_logreg_smote, digits=6)}")

  ''' Model Decision Tree '''
  clf = DecisionTreeClassifier()
  clf = clf.fit(X_train_smote,y_train_smote)
  y_pred_dectree_smote = clf.predict(X_test_smote)

  print(f"Accuracy Decision Tree: {metrics.accuracy_score(y_test_smote, y_pred_dectree_smote)}")

  ''' Confusion matrix Decision Tree '''
  confusion_matrix_decision = confusion_matrix(y_test_smote, y_pred_dectree_smote)
  confusion_matrix_decision

  plt.figure(figsize = (12, 6))

  sns.heatmap(confusion_matrix_decision, annot = True, cmap = "hot")
  plt.xlabel("Prediction")
  plt.ylabel("Reality")
  plt.savefig("./src/static/confusion_matrix_decision_tree.png")

  ''' Classification report Decision Tree '''
  print(f"Classification report Decision Tree: {classification_report(y_test_smote, y_pred_dectree_smote, digits=6)}")


  ''' Selection of best model '''

  models = ["Logistic Regression", "Decision Tree"]

  false_positives = []

  false_positives.append(confusion_matrix_logreg[0, 1])
  false_positives.append(confusion_matrix_decision[0, 1])


  false_negatives =[]

  false_negatives.append(confusion_matrix_logreg[1, 0])
  false_negatives.append(confusion_matrix_decision[1, 0])
 

  accuracy = [logreg.score(X_test_smote, y_test_smote), metrics.accuracy_score(y_test_smote, y_pred_dectree_smote)]

  model_comp = []

  model_comp = pd.DataFrame(models, columns=["Model"])
  model_comp["False Negatives"] = false_negatives
  model_comp["False Positives"] = false_positives
  model_comp["Accuracy"] = accuracy
  dfi.export(model_comp, './src/static/best_model.png', table_conversion='matplotlib')
  print(model_comp)

  return model_comp

if __name__ == "__main__":
    run()