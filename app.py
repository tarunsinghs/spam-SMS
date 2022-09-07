import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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
    xgb.score(X_test, y_test)

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    neigh.score(X_test, y_test)

    svm_predict = svm.SVC(decision_function_shape='ovr')
    svm_predict.fit(X_train, y_train)
    svm_predict.score(X_test, y_test)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    rf.score(X_test, y_test)

    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    dt.score(X_test, y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()

        final_output = {
            "message": data[0],
            "knn": neigh.predict(vect),
            "svm": svm_predict.predict(vect),
            "rf": rf.predict(vect),
            "dt": dt.predict(vect),
            "nb": clf.predict(vect)
        }

        print(final_output)

    return render_template('result.html', output=final_output)


if __name__ == '__main__':

    app.run(debug=True)
