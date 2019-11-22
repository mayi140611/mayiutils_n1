#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: Estimator.py
@time: 2019-09-05 15:14
"""


class DataProcessor:
    def load_data(self, cat_cols, num_cols=None, date_cols=None, mode='train'):
        """
        载入数据，
        准备label，
        及train_set, val_set, pridect_set的通用操作
        划分train_set，val_set
        :param mode: 'train' | 'predict'
        :return:
        """
        (train_x, train_y), (val_x, val_y), test_x = None
        if mode == "train":
            return (train_x, train_y), (val_x, val_y)
        elif mode == "predict":
            return test_x

    def train_input_fn(self):
        """
        A function that provides input data for training as minibatches.
        :return:
        """
        pass

    def val_input_fn(self, features, labels, batch_size=None):
        pass

    def test_input_fn(self, features, batch_size=None):
        pass


class DataProcessor4CBCEstimator(DataProcessor):
    """
    DataProcessor for CatBoostClassifierEstimator
    """
    def load_data(self, df, cat_cols, num_cols=None, date_cols=None, mode='train', split=0.25, random_state=14):
        for i in date_cols:
            try:
                df.loc[:, f'{i}_day'] = df[i].dt.day
                df.loc[:, f'{i}_month'] = df[i].dt.month
                df.loc[:, f'{i}_weekday'] = df[i].dt.weekday
            except Exception as e:
                print(i, e)
                df.loc[:, f'{i}_day'] = 0
                df.loc[:, f'{i}_month'] = 0
                df.loc[:, f'{i}_weekday'] = 0
            cat_cols.extend([f'{i}_day', f'{i}_month', f'{i}_weekday'])
        import itertools
        for s, e in itertools.combinations(date_cols, 2):
            try:
                df.loc[:, f'{e}-{s}'] = (df[e] - df[s]).dt.days
            except Exception as ee:
                print(s, e, ee)
                df.loc[:, f'{e}-{s}'] = 0
            num_cols.append(f'{e}-{s}')
        df = df[cat_cols + num_cols]
        cat_cols.remove('label')
        df.loc[:, cat_cols] = df[cat_cols].fillna('<UNK>')
        df.loc[:, cat_cols] = df[cat_cols].applymap(str)
        if mode == 'train':
            from sklearn.model_selection import train_test_split
            df_train, df_val, y_train, y_val = train_test_split(df.drop(columns=['label']), df['label'],
                                                                test_size=split, random_state=random_state,
                                                                stratify=df['label'])
            return (df_train, df_val), (y_train, y_val), cat_cols

    def train_input_fn(self, features, labels, cat_cols, batch_size=None):
        return features, labels, cat_cols

    def val_input_fn(self, features, labels, cat_cols, batch_size=None):
        return features, labels, cat_cols


class Estimator(object):
    """Base Estimator class.
    参考 https://github.com/tensorflow/estimator/blob/master/tensorflow_estimator/python/estimator/estimator.py
    """
    def __init__(self, params: dict=None, data_processor: DataProcessor=None):
        """
        Constructs an `Estimator` instance.
        :param params:
        :param data_processor:
        """
        self._params = params
        self._data_processor = data_processor

    def model_fn(self):
        return ""

    def train(self,
              steps=None):
        """
        Trains a model given training data `input_fn`.
        :param input_fn:
        :param steps: Number of steps for which to train the model.
        :return:
        """
        pass

    def evaluate(self, steps=None):
        """
        Evaluates the model given evaluation data `input_fn`.
    For each step, calls `input_fn`, which returns one batch of data.
        :param steps:
        :param hooks:
        :param checkpoint_path:
        :param name:
        :return:
        """
        pass

    def predict(self,
                predict_keys=None,
                hooks=None,
                checkpoint_path=None,
                yield_single_examples=True):
        """
        Yields predictions for given features.
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
    def __init__(self, params, data_processor):
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
        super().__init__(params=params, data_processor=data_processor)
        self._model = self.model_fn()

    def model_fn(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**self._params)

    def train(self, plot=True, verbose=True, show_features_importance=False):
        from catboost import Pool
        import pandas as pd
        features, labels, cat_cols = self._data_processor.train_input_fn()
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

    def evaluate(self):
        from sklearn.metrics import classification_report
        from catboost import Pool
        import pandas as pd
        df_val, y_val, cat_cols = self._data_processor.val_input_fn()
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

    def predict(self, explain=True):
        from catboost import Pool
        import pandas as pd
        df_predict, cat_cols, id_col_name = self._data_processor.test_input_fn()  # id_col_name: 预测的标识列名称
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


class XGBoostRegressorEstimator(Estimator):
    def __init__(self, params, data_processor):
        """

        :param params:
            params = {
                'max_depth': 3,
                learning_rate: 0.1,
                'n_estimators': 100,
                'objective': 'reg:linear',
                'booster': 'gbtree'
            }
        """
        super().__init__(params=params)
        self._model = self.model_fn()

    def model_fn(self):
        from xgboost import XGBRegressor
        return XGBRegressor(**self._params)

    def train(self, input_fn, plot=True, verbose=True, show_features_importance=False):
        import pandas as pd
        features, ys, feature_names = input_fn()
        self._model.fit(features, ys)
        if show_features_importance:
            df_features_importance = pd.DataFrame({'name': feature_names,
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
        from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
        import pandas as pd
        import numpy as np
        features, ys = input_fn()  # test_data: np.array
        r = self._model.predict(features)
        dfr = pd.DataFrame(ys)
        dfr.columns = ['y_true']
        dfr['y_predict'] = r
        print(f'mae: {mae(ys, r)}')
        print(f'mse: {mse(ys, r)}')
        print(f'rmse: {np.sqrt(mse(ys, r))}')
        return dfr

    def predict(self, input_fn):
        import pandas as pd
        df_predict, id_col_name = input_fn()  # id_col_name: 预测的标识列名称
        dfr = pd.DataFrame(df_predict[id_col_name])
        r = self._model.predict(df_predict.values)

        dfr['y_predict'] = r
        return dfr


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




