import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline
import matplotlib as mpl 
mpl.rcParams['figure.dpi'] = 400 
import graphviz 

df_orig = pd.read_excel('/content/default_of_credit_card_clients.xls')

df_zero_mask = df_orig == 0

feature_zero_mask = df_zero_mask.iloc[:,1:].all(axis=1)

sum(feature_zero_mask)

df_clean = df_orig.loc[~feature_zero_mask,:].copy()

df_clean['EDUCATION'].replace(to_replace=[0, 5, 6], value=4, inplace=True)

df_clean['MARRIAGE'].replace(to_replace=0, value=3, inplace=True)

missing_pay_1_mask = df_clean['PAY_1'] == 'Not available'

df_missing_pay_1 = df_clean.loc[missing_pay_1_mask,:].copy()

df = pd.read_csv('/content/cleaned_data.csv')

features_response = df.columns.tolist()

items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']

features_response = [item for item in features_response if item not in items_to_remove]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(df[features_response[:-1]].values, df['default payment next month'].values,
test_size=0.2, random_state=24)

np.median(X_train[:,4])

np.random.seed(seed=1)
fill_values = [0, np.random.choice(X_train[:,4], size=(3021,), replace=True)]

fill_strategy = ['mode', 'random']

from sklearn.model_selection import KFold
k_folds = KFold(n_splits=4, shuffle=True, random_state=1)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth=9,
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
random_state=4, verbose=1, warm_start=False, class_weight=None)

for counter in range(len(fill_values)):

    df_fill_pay_1_filled = df_missing_pay_1.copy()
    df_fill_pay_1_filled['PAY_1'] = fill_values[counter]
    X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
    train_test_split(
        df_fill_pay_1_filled[features_response[:-1]].values,
        df_fill_pay_1_filled['default payment next month'].values,
    test_size=0.2, random_state=24)
   
    X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
    y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
    
    imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')
    
    test_score = imputation_compare_cv['test_score']
    

pay_1_df = df.copy()

features_for_imputation = pay_1_df.columns.tolist()

items_to_remove_2 = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university', 'default payment next month', 'PAY_1']

features_for_imputation = [item for item in features_for_imputation if item not in items_to_remove_2]

X_impute_train, X_impute_test, y_impute_train, y_impute_test = \
train_test_split(
    pay_1_df[features_for_imputation].values,
    pay_1_df['PAY_1'].values,
test_size=0.2, random_state=24)

rf_impute_params = {'max_depth':[3, 6, 9, 12],
             'n_estimators':[10, 50, 100, 200]}

from sklearn.model_selection import GridSearchCV

cv_rf_impute = GridSearchCV(rf, param_grid=rf_impute_params, scoring='accuracy',
                            n_jobs=-1, iid=False, refit=True,
                            cv=4, verbose=2, error_score=np.nan, return_train_score=True)

cv_rf_impute.fit(X_impute_train, y_impute_train)

impute_df = pd.DataFrame(cv_rf_impute.cv_results_)
impute_df

cv_rf_impute.best_params_

cv_rf_impute.best_score_

pay_1_value_counts = pay_1_df['PAY_1'].value_counts().sort_index()

pay_1_value_counts/pay_1_value_counts.sum()

y_impute_predict = cv_rf_impute.predict(X_impute_test)

from sklearn import metrics

metrics.accuracy_score(y_impute_test, y_impute_predict)

X_impute_all = pay_1_df[features_for_imputation].values
y_impute_all = pay_1_df['PAY_1'].values

rf_impute = RandomForestClassifier(n_estimators=100, max_depth=12)

rf_impute

rf_impute.fit(X_impute_all, y_impute_all)

df_fill_pay_1_model = df_missing_pay_1.copy()

df_fill_pay_1_model['PAY_1'].head()

df_fill_pay_1_model['PAY_1'] = rf_impute.predict(df_fill_pay_1_model[features_for_imputation].values)

df_fill_pay_1_model['PAY_1'].head()

df_fill_pay_1_model['PAY_1'].value_counts().sort_index()

X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)

X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)

imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')

imputation_compare_cv['test_score']
np.mean(imputation_compare_cv['test_score'])
np.std(imputation_compare_cv['test_score'])

df_fill_pay_1_model['PAY_1'] = np.zeros_like(df_fill_pay_1_model['PAY_1'].values)

df_fill_pay_1_model['PAY_1'].unique()

X_fill_pay_1_train, X_fill_pay_1_test, y_fill_pay_1_train, y_fill_pay_1_test = \
train_test_split(
    df_fill_pay_1_model[features_response[:-1]].values,
    df_fill_pay_1_model['default payment next month'].values,
test_size=0.2, random_state=24)

X_train_all = np.concatenate((X_train, X_fill_pay_1_train), axis=0)
X_test_all = np.concatenate((X_test, X_fill_pay_1_test), axis=0)
y_train_all = np.concatenate((y_train, y_fill_pay_1_train), axis=0)
y_test_all = np.concatenate((y_test, y_fill_pay_1_test), axis=0)

imputation_compare_cv = cross_validate(rf, X_train_all, y_train_all, scoring='roc_auc',
                                       cv=k_folds, n_jobs=-1, verbose=1,
                                       return_train_score=True, return_estimator=True,
                                       error_score='raise-deprecating')


rf.fit(X_train_all, y_train_all)

y_test_all_predict_proba = rf.predict_proba(X_test_all)

pickle.dump(rf, open('Credit_Card.pickle.dat', 'wb'))
