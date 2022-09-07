import pickle
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, url_for
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv(
        r"{}/Data/spam.csv".format(os.getcwd()), encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    # Features and Labels
    df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
    df['message'] = df['v2']
    df.drop(['v1', 'v2'], axis=1, inplace=True)
    X = df['message']
    y = df['label']

    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    # XGboost
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)

    svm_predict = svm.SVC(decision_function_shape='ovo')
    svm_predict.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()

    return render_template('result.html', output={"message": data[0], "knn": neigh.predict(vect), "svm": svm_predict.predict(vect), "rf": rf.predict(vect), "dt": dt.predict(vect), "nb": clf.predict(vect)})


if __name__ == '__main__':

    app.run(debug=True)
