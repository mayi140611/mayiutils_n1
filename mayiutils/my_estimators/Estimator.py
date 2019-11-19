#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: Estimator.py
@time: 2019-09-05 15:14
"""


class Estimator(object):
    """Base Estimator class.
    参考 https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/estimator.py
    """
    def __init__(self, params=None):
        """Constructs an `Estimator` instance.
        """
        self._params = params

    def model_fn(self):
        return ""

    def train(self,
              input_fn,
              steps=None):
        """
        Trains a model given training data `input_fn`.
        :param input_fn: A function that provides input data for training as minibatches.
        :param steps: Number of steps for which to train the model.
        :return:
        """
        pass

    def evaluate(self, input_fn,
                 steps=None):
        """
        Evaluates the model given evaluation data `input_fn`.
    For each step, calls `input_fn`, which returns one batch of data.
        :param input_fn:
        :param steps:
        :param hooks:
        :param checkpoint_path:
        :param name:
        :return:
        """
        pass

    def predict(self,
                input_fn,
                predict_keys=None,
                hooks=None,
                checkpoint_path=None,
                yield_single_examples=True):
        """
        Yields predictions for given features.
        :param input_fn:
        :param predict_keys:
        :param hooks:
        :param checkpoint_path:
        :param yield_single_examples:
        :return:
        """
        pass

    def explain(self):
        """模型解释"""
        pass


class CatBoostClassifierEstimator(Estimator):
    def __init__(self, params):
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
        super().__init__(params=params)
        self._model = self.model_fn()

    def model_fn(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**self._params)

    def train(self, input_fn, plot=True, verbose=True, show_features_importance=False):
        from catboost import Pool
        import pandas as pd
        features, labels, cat_cols = input_fn()
        train_data = Pool(data=features,
                          label=labels,
                          cat_features=cat_cols)
        self._model.fit(train_data, plot=plot, verbose=verbose)
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
        from catboost import Pool
        import pandas as pd
        df_val, y_val, cat_cols = input_fn()
        test_data = Pool(data=df_val,
                         cat_features=cat_cols)
        r = self._model.predict(test_data)
        print(pd.Series(r).value_counts())
        print(classification_report(y_val, r))
        dfr = pd.DataFrame(y_val)
        dfr.columns = ['true_label']
        y_test_hat = self._model.predict_proba(test_data)[:, 1]
        dfr['score'] = y_test_hat
        dfr['predict_label'] = r
        dfr = dfr.sort_values('score', ascending=False)
        dfr['order'] = range(1, dfr.shape[0] + 1)
        print(dfr[dfr.true_label == 1])
        return dfr

    def predict(self, input_fn, explain=True):
        from catboost import Pool
        import pandas as pd
        df_predict, cat_cols, id_col_name = input_fn()  # id_col_name: 预测的标识列名称
        test_data = Pool(data=df_predict,
                         cat_features=cat_cols)
        dfr = pd.DataFrame(df_predict[id_col_name])
        y_test_hat = self._model.predict_proba(test_data)[:, 1]
        dfr['score'] = y_test_hat
        dfr['predict_label'] = self._model.predict(test_data)
        s = dfr['predict_label'].value_counts()
        print(s)
        print(f'su sample num：{s.loc[1] if 1 in s else 0}')
        if explain:
            rr = self.explain(df_predict, cat_cols, id_col_name, dfr)
            return dfr, rr
        return dfr

    def explain(self, df_predict, cat_cols, id_col_name, dfr):
        """模型解释"""
        from catboost import Pool
        import pandas as pd
        test_data = Pool(data=df_predict, cat_features=cat_cols)
        shap_values = self._model.get_feature_importance(test_data, type='ShapValues')
        dfs = pd.DataFrame(shap_values[:, :-1], columns=df_predict.columns, index=df_predict[id_col_name])
        dfs_T = dfs.T
        ss = []
        for i in range(dfs_T.shape[1]):
            ss.append(dfs_T.iloc[:, i].copy().sort_values(ascending=False).iloc[:5])
        count = 0
        rr = []
        for line in dfr[dfr.predict_label == 1].itertuples():
            rr.append({"id": line[id_col_name], "FS_SC_SCORE": round(line.score, 2),
                       "EXPLAIN": ','.join(
                           [f'{i[0]}:{round(i[1], 2)}' for i in list(zip(ss[count].index, ss[count].values))])})
            count += 1
        print(rr)
        return rr


if __name__ == '__main__':
    params = {
        'iterations': 10,
        # learning_rate: 0.1,
        #     'custom_metric': 'AUC',
        'custom_metric': 'F1',
        'loss_function': 'CrossEntropy'
    }
    c = CatBoostClassifierEstimator(params=params)
    print(c._params)
    print(c._model)




