import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

dlt = pd.read_csv("dengue_labels_train.csv", encoding="utf-8")
dft = pd.read_csv("dengue_features_train.csv",  encoding="utf-8")

df = dlt.merge(dft)
#print(df)

#print(df.head())

df = df.fillna(df.mean())

df = df.drop("week_start_date", axis=1)

y = df["total_cases"]
X = df.drop("total_cases", axis=1)

X = pd.get_dummies(X, columns = ["city", "year", "weekofyear"])


kf = KFold(n_splits=5, shuffle = True, random_state = 8)

def hparam_tuning(model, X, y, param_grid):
    return GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error').fit(X,y).best_params_


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


#sel = VarianceThreshold(threshold=(.5 * (1 - .5)))
#X = sel.fit_transform(X)

pvalues = f_regression(X,y)
array = list(pvalues[1])
X = pd.DataFrame(X)
#print(X.head())


def removHighPValueFeature(array,X,threshold):
	col = 0
	for v in array:
		if v > threshold:
			col = int(array.index(v,col))
			X = X.drop(col, axis=1)
	return X		
			
X = removHighPValueFeature(array,X,0.01)
#print(X.head())

def run(model, X, y):
    errors = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        MAE = mean_absolute_error(y_test, y_pred)
        errors.append(MAE)

    return errors

param_grid = {#'bootstrap': [True, False],
 'max_depth': [2, 4, 8, 16, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [2, 4, 8, 16]}

#best_params = hparam_tuning(RandomForestRegressor(), X, y, param_grid)
#print(best_params)

models_reg = [LinearRegression(), Lasso(), Ridge(), KNeighborsRegressor(), DecisionTreeRegressor(), SVR(), 
          GradientBoostingRegressor(), RandomForestRegressor(), 
          AdaBoostRegressor(), MLPRegressor()]
models_clf = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), SVC(), 
          GradientBoostingClassifier(), RandomForestClassifier(), 
          AdaBoostClassifier(), MLPClassifier()]
models_clf = [GaussianNB(), MultinomialNB()]

for model in models_reg:
    errors = run(model, X ,y)
    print(type(model).__name__)
    print("%.2f\n" %np.mean(errors))

