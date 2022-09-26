# -----------------------------------------------------------
# This is the project class.
# Here we clean the data, impute missing data and train our models
# 2022 Lars Kristian Dugstad, Mainz, Germany
# email lkdugstad@googlemail.com
# -----------------------------------------------------------
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree, metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
import joblib
import itertools


class DataPreparer:

    def __init__(self, feature_select='tree'):
        self.df = self.load_data()

    def load_data(self):
        '''
        df1 = pd.read_csv('data/1year.csv')
        df1['time'] = 5
        df2 = pd.read_csv('data/2year.csv')
        df2['time'] = 4
        '''
        df3 = pd.read_csv('data/3year.csv')
        df3['time'] = 3
        df4 = pd.read_csv('data/4year.csv')
        df4['time'] = 2
        df5 = pd.read_csv('data/5year.csv')
        df5['time'] = 1
        df = pd.concat([df3, df4, df5])
        df['class'] = pd.to_numeric(df['class'].apply(lambda x: re.search('\d+', x).group(0)))
        df['is_bankrupt_after_years'] = df['class'] * df['time']
        df = df.drop(['class', 'time'], axis=1)
        return df

    def impute_missing_values(self, showBeforeAndAfter=False):
        if showBeforeAndAfter:
            print(self.df.isnull().sum())
        self.df = pd.DataFrame(SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(self.df),
                               columns=self.df.columns)
        if showBeforeAndAfter:
            print(self.df.isnull().sum())

    def synthetic_minority_oversampling(self):
        oversample = BorderlineSMOTE()
        X = self.df.drop('is_bankrupt_after_years', axis=1)
        y = self.df['is_bankrupt_after_years']
        X_new, y_new = oversample.fit_resample(X, y)
        X_new['is_bankrupt_after_years'] = y_new
        self.df = X_new
        # self.X_train, self.y_train = oversample.fit_resample(self.X_train, self.y_train)
        # self.X_test, self.y_test = oversample.fit_resample(self.X_test, self.y_test)

    def feature_selection(self, feature_select):
        X = self.df.drop('is_bankrupt_after_years', axis=1)
        y = self.df.is_bankrupt_after_years
        if feature_select == 'tree':
            from sklearn.ensemble import ExtraTreesClassifier
            from sklearn.feature_selection import SelectFromModel
            clf = ExtraTreesClassifier(n_estimators=50, random_state=42)
            clf = clf.fit(X, y)
            model = SelectFromModel(clf, prefit=True)
            dfNew = pd.DataFrame(model.transform(X), columns=X.columns[model.get_support()])

        if feature_select == 'kbest':
            # ANOVA feature selection for numeric input and categorical output
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import f_classif
            fs = SelectKBest(score_func=f_classif, k=20)
            X_selected = fs.fit_transform(X, y)
            dfNew = pd.DataFrame(X_selected, columns=X.columns[fs.get_support()])
        dfNew['is_bankrupt_after_years'] = y
        self.df = dfNew

    def min_max_scale(self):
        X = self.df.drop('is_bankrupt_after_years', axis=1)
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(X)
        df = pd.DataFrame(x_scaled)
        df['is_bankrupt_after_years'] = self.df['is_bankrupt_after_years']
        self.df = df

    def heatmap(self, method):
        # plt.figure(figsize=(12, 10), dpi=100)
        df = self.df
        df['class'] = df['is_bankrupt_after_years']
        df = df.drop('is_bankrupt_after_years', axis=1)
        corr_matrix = df.corr(method=method)
        sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns, cmap='RdYlGn',
                    center=0, annot_kws={'fontsize': 10}, fmt='.2f')
        plt.title('Kendall heatmap after selection', fontsize=22, color='black')
        plt.xlabel('Feature 2')
        plt.ylabel('Feature 1')
        #plt.show()
        plt.savefig('images/CorrelationHeatmapAfter.png', bbox_inches='tight')
        plt.show()


class Model:

    def __init__(self, df, model='RandomForest', test_split=0.2):
        self.df = df
        self.model = model
        X = self.df.drop('is_bankrupt_after_years', axis=1)
        y = self.df['is_bankrupt_after_years']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_split, stratify=y,
                                                                                random_state=0)
        self.predict()
        self.get_values()

    def predict(self):
        if self.model == 'DecisionTree':  # 0.9084627799072019
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(self.X_train, self.y_train)
            self.y_pred = clf.predict(self.X_test)
            self.clf = clf
        if self.model == 'RandomForest':  # 0.9807343151099456
            clf = RandomForestClassifier(max_depth=100, random_state=0)
            clf = clf.fit(self.X_train, self.y_train)
            #self.y_pred = clf.predict_proba(self.X_test)
            self.y_pred = clf.predict(self.X_test)
            print(self.y_pred)
            self.classifier = clf
        if self.model == 'AdaBoost':  # 0.49399838612063746
            clf = AdaBoostClassifier(n_estimators=100, random_state=0)
            clf = clf.fit(self.X_train, self.y_train)
            self.y_pred = clf.predict(self.X_test)
        if self.model == 'KNeighbors':  # 0.7816219487593302
            clf = KNeighborsClassifier(n_neighbors=3)
            clf = clf.fit(self.X_train, self.y_train)
            self.y_pred = clf.predict(self.X_test)
        if self.model == 'XGBoost':  # 0.9425055477103087
            model = XGBClassifier(eval_metric="mlogloss")
            model.fit(self.X_train, self.y_train)
            self.y_pred = model.predict_proba(self.X_test)
            self.y_pred = [np.argmax(ele) for ele in self.y_pred]
            self.classifier = model

    def save_model_for_serving(self):
        joblib.dump(self.classifier, 'Bankruptcy_' + self.model + '.joblib')

    def get_values(self):
        print(self.model)
        print(accuracy_score(self.y_test, self.y_pred))
        print(metrics.classification_report(self.y_test, self.y_pred, digits=4))

    def confusion_matrix(self, percent=False):
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)
        if percent:
            sns.heatmap(conf_matrix / np.sum(conf_matrix), annot=True, fmt='.2%', cmap='YlGn')
            plt.title('Confusion matrix percentage', fontsize=22, color='black')
            plt.xlabel('Predicted category')
            plt.ylabel('True category')
            plt.yticks(rotation=0)
            plt.savefig('images/ConfusionMatrixPercentage.png', bbox_inches='tight')
            plt.show()
        else:
            sns.heatmap(conf_matrix, annot=True, cmap='YlGn', fmt='g')
            plt.title('Confusion matrix values', fontsize=22, color='black')
            plt.xlabel('Predicted category')
            plt.ylabel('True category')
            plt.yticks(rotation=0)
            plt.savefig('images/ConfusionMatrixValues.png', bbox_inches='tight')
            plt.show()

    def save_model_for_serving(self):
        joblib.dump(self.classifier, 'Bankruptcy_' + self.model + '.joblib')


if __name__ == '__main__':
    data = DataPreparer()
    data.impute_missing_values(showBeforeAndAfter=False)
    data.feature_selection(feature_select='tree')
    data.synthetic_minority_oversampling()
    model = Model(df=data.df, model='XGBoost', test_split=0.2)
    model.confusion_matrix(percent=True)
    model.confusion_matrix(percent=False)


