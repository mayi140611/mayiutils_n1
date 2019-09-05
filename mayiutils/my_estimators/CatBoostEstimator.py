#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: CatBoostEstimator.py
@time: 2019-09-05 16:07
"""
import pandas as pd
from catboost import Pool


class CatBoostEstimator:
    def __init__(self, params, cat_cols):
        """

        :param params:
            params = {
                'iterations': iterations,
                # learning_rate: 0.1,
                #     'custom_metric': 'AUC',
                'custom_metric': custom_metric,
                'loss_function': 'CrossEntropy'
            }
        """
        self._params = params
        self._cat_cols = cat_cols
        from catboost import CatBoostClassifier
        self._model = CatBoostClassifier(**self._params)

    def model_fn(self):
        return self._model

    def train(self, input_fn, plot=True, verbose=True, show_features_importance=False):

        features, labels = input_fn()
        train_data = Pool(data=features,
                          label=labels,
                          cat_features=self._cat_cols)
        self.model_fn().fit(train_data, plot, verbose)
        if show_features_importance:
            df_features_importance = pd.DataFrame({'name': self._model.feature_names_,
                                                   'value': self._model.feature_importances_})
            df_features_importance = df_features_importance.sort_values('value', ascending=False)

            df_features_importance.reset_index(drop=True, inplace=True)
            print(df_features_importance.head(5))
            import matplotlib.pyplot as plt
            fea_ = df_features_importance.sort_values('value')[df_features_importance.value > 0].value
            fea_name = df_features_importance.sort_values('value')[df_features_importance.value > 0].name
            plt.figure(figsize=(10, 20))
            plt.barh(fea_name, fea_, height=0.5)
            plt.show()
            return df_features_importance

    def evaluate(self, input_fn):
        from sklearn.metrics import classification_report
        df_val, y_val = input_fn()
        test_data = Pool(data=df_val,
                         cat_features=self._cat_cols)
        print(pd.Series(self._model.predict(test_data)).value_counts())
        print(classification_report(y_val, self._model.predict(test_data)))
        dfr = pd.DataFrame(y_val)
        dfr.columns = ['true_label']
        y_test_hat = self._model.predict_proba(test_data)[:, 1]
        dfr['score'] = y_test_hat
        dfr['predict_label'] = self._model.predict(test_data)
        dfr = dfr.sort_values('score', ascending=False)
        dfr['order'] = range(1, dfr.shape[0] + 1)
        print(dfr[dfr.true_label == 1])
        return dfr

    def predict(self, input_fn):
        df_test = input_fn()
        test_data = Pool(data=df_test,
                         cat_features=self._cat_cols)
        r = self._model.predict(test_data)
        print(pd.Series(r).value_counts())
        return r

