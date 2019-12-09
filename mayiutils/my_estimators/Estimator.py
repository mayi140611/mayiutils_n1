#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: Estimator.py
@time: 2019-09-05 15:14
"""


class DataProcessor:
    def __init__(self, features=None, labels=None, cat_cols=None, split=0.1, random_state=14):
        self._features = features
        self._labels = labels
        self._cat_cols = cat_cols
        from sklearn.model_selection import train_test_split
        self._features_train, self._features_val, self._labels_train, self._labels_val = train_test_split(features, labels,
                                                                                                          test_size=split, random_state=random_state,
                                                                                                          stratify=labels)

    def cv_input_fn(self):
        return self._features, self._labels, self._cat_cols

    def train_input_fn(self):
        return self._features_train, self._labels_train, self._cat_cols

    def eval_input_fn(self):
        return self._features_val, self._labels_val, self._cat_cols

    def test_input_fn(self, features, batch_size=None):
        pass


class FastTextDataProcessor(DataProcessor):
    def __init__(self, features=None, labels=None, split=0.1, random_state=14):
        """
        注意：这里的输入都是没有经过分词处理的短句
        :param features: Series
        :param labels: Series
        :param cat_cols:
        :param split:
        :param random_state:
        """
        super().__init__(features=features, labels=labels, split=split, random_state=random_state)

    def train_input_fn(self, dest_train_file_path='data_gen/trainset.txt'):
        """

        :param dest_train_file_path:
        :return:
        """
        train_set = self._features_train.map(lambda x: ' '.join(list(x))) + ' __label__' + self._labels_train.map(
            str) + '\n'
        with open(dest_train_file_path, 'w', encoding='utf8') as f:
            f.writelines(train_set.tolist())

    def eval_input_fn(self, dest_val_file_path='data_gen/valset.txt'):
        val_set = self._features_val.map(lambda x: ' '.join(list(x))) + ' __label__' + self._labels_val.map(
            str) + '\n'
        with open(dest_val_file_path, 'w', encoding='utf8') as f:
            f.writelines(val_set.tolist())

    def test_input_fn(self, features):
        """

        :param features: Series
        :return:
        """
        return features.map(lambda x: ' '.join(list(x)))



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

    def test(self,
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

    def baseline(self):
        """
        一个使用默认参数的模型工作流：[cv,] train, val, test
        :return:
        """
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass


class ClsEstimator(Estimator):
    """Classifier Estimator class.
    """
    def __init__(self, params: dict=None, data_processor: DataProcessor=None, model_path: str=None):
        """
        Constructs an `Estimator` instance.
        :param params:
        :param data_processor:
        :param model_path:  模型的路径
        """
        self._params = params
        self._data_processor = data_processor
        self._model_path = model_path

    def model_fn(self):
        raise NotImplementedError()

    def train(self,
              steps=None):
        """
        :param steps: Number of steps for which to train the model.
        :return:
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def train_eval(self):
        """
        train & evaluate
        :return:
        """
        pass

    def cross_val(self):
        """

        :return:
        """
        raise NotImplementedError()

    def test(self,
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
        raise NotImplementedError()

    def explain(self):
        """模型解释"""
        raise NotImplementedError()


class CatBoostClsEstimator(ClsEstimator):
    def __init__(self, params=None, data_processor=None, model_path=None):
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
        super().__init__(params=params, data_processor=data_processor, model_path=model_path)
        if self._model_path:
            self.load_model(self._model_path)
        else:
            self._model = self.model_fn()

    def model_fn(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**self._params) if self._params else CatBoostClassifier()

    def train(self, plot=True, verbose=True, show_features_importance=True, init_model=None):
        """

        :param plot:
        :param verbose:
        :param show_features_importance:
        :param init_model:
            CatBoost class or string, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string, load initial model from the path specified by this string.
        :return:
        """
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

    def evaluate(self, df_val=None, y_val=None, cat_cols=None):
        from sklearn.metrics import classification_report
        from catboost import Pool
        import pandas as pd
        if not df_val:
            df_val, y_val, cat_cols = self._data_processor.eval_input_fn()
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

    def train_eval(self, plot=True, verbose=True, show_features_importance=True, init_model=None, use_best_model: bool=True, early_stopping_rounds: int=None):
        """

        :param plot:
        :param verbose:
        :param show_features_importance:
        :param init_model:
            CatBoost class or string, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string, load initial model from the path specified by this string.
        :param use_best_model:
        :param early_stopping_rounds:
        :return:
        """
        from catboost import Pool
        import pandas as pd
        features, labels, cat_cols = self._data_processor.train_input_fn()
        train_data = Pool(data=features,
                          label=labels,
                          cat_features=cat_cols)
        df_val, y_val, cat_cols = self._data_processor.eval_input_fn()
        val_data = Pool(data=df_val, label=y_val, cat_features=cat_cols)
        self._model.fit(train_data, eval_set=val_data, plot=plot, verbose=verbose)
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

    def cross_val(self, nfold=3, shuffle=True, stratified=None, plot=True, partition_random_seed: int=14):
        """

        :param nfold:
        :param shuffle:
        :param stratified:
        :param plot:
        :param partition_random_seed:
        :return:
            cv results : pandas.core.frame.DataFrame with cross-validation results
            columns are: test-error-mean  test-error-std  train-error-mean  train-error-std
        """
        from catboost import Pool, cv
        import numpy as np
        features, labels, cat_cols = self._data_processor.cv_input_fn()
        cv_data = Pool(data=features,
                          label=labels,
                          cat_features=cat_cols)
        cv_result = cv(cv_data, self._params, nfold=nfold, shuffle=shuffle, stratified=stratified, plot=plot, partition_random_seed=partition_random_seed)
        print('Best validation {} score: {:.2f}±{:.2f} on step {}'.format(
            self._params['custom_metric'],
            np.max(cv_result[f'test-{self._params["custom_metric"]}-mean']),
            cv_result[f'test-{self._params["custom_metric"]}-std'][np.argmax(cv_result[f'test-{self._params["custom_metric"]}-mean'])],
            np.argmax(cv_result[f'test-{self._params["custom_metric"]}-mean'])
        ))
        print('Precise validation {} score: {}'.format(self._params['custom_metric'], np.max(cv_result[f'test-{self._params["custom_metric"]}-mean'])))
        return cv_result

    def baseline(self):
        import numpy as np
        cv_result = self.cross_val()
        self._params.update({
            'iterations': np.argmax(cv_result[f'test-{self._params["custom_metric"]}-mean']),
        })
        print(self._params)
        df_features_importance = self.train_eval()
        dfr = self.evaluate()
        return df_features_importance, dfr

    def test(self, df_test=None, cat_cols=None, explain=True):
        from catboost import Pool
        import pandas as pd
        if df_test is None:
            df_test, cat_cols = self._data_processor.test_input_fn()
        test_data = Pool(data=df_test,
                         cat_features=cat_cols)
        dfr = pd.DataFrame(df_test.index)
        y_test_hat = self._model.predict_proba(test_data)[:, 1]
        dfr['score'] = y_test_hat
        dfr['test_label'] = self._model.predict(test_data)
        s = dfr['test_label'].value_counts()
        print(s)
        print(f'su sample num：{s.loc[1] if 1 in s else 0}')
        if explain:
            rr = self.explain(df_test, cat_cols, dfr)
            return dfr, rr
        return dfr

    def explain(self, df_test, cat_cols, dfr):
        """模型解释"""
        from catboost import Pool
        import pandas as pd
        test_data = Pool(data=df_test, cat_features=cat_cols)
        shap_values = self._model.get_feature_importance(test_data, type='ShapValues')
        dfs = pd.DataFrame(shap_values[:, :-1], columns=df_test.columns, index=df_test.index)
        dfs_T = dfs.T
        ss = []
        for i in range(dfs_T.shape[1]):
            ss.append(dfs_T.iloc[:, i].copy().sort_values(ascending=False).iloc[:5])
        count = 0
        rr = []
        for line in dfr[dfr.test_label == 1].itertuples():
            rr.append({"id": line[0], "SCORE": round(line.score, 2),
                       "EXPLAIN": ','.join(
                           [f'{i[0]}:{round(i[1], 2)}' for i in list(zip(ss[count].index, ss[count].values))])})
            count += 1
        print(rr)
        return rr

    def save_model(self, model_path):
        """

        :param model_path: 'catboost_model.dump'
        :return:
        """
        self._model.save_model(model_path)

    def load_model(self, model_path):
        self._model = self.model_fn().load_model(model_path)


class XGBoostClsEstimator(ClsEstimator):
    def __init__(self, params=None, data_processor=None, model_path=None):
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
        super().__init__(params=params, data_processor=data_processor, model_path=model_path)
        if self._model_path:
            self.load_model(self._model_path)
        else:
            self._model = self.model_fn()

    def model_fn(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**self._params) if self._params else CatBoostClassifier()

    def train(self, plot=True, verbose=True, show_features_importance=True, init_model=None):
        """

        :param plot:
        :param verbose:
        :param show_features_importance:
        :param init_model:
            CatBoost class or string, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string, load initial model from the path specified by this string.
        :return:
        """
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

    def evaluate(self, df_val=None, y_val=None, cat_cols=None):
        from sklearn.metrics import classification_report
        from catboost import Pool
        import pandas as pd
        if not df_val:
            df_val, y_val, cat_cols = self._data_processor.eval_input_fn()
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

    def train_eval(self, plot=True, verbose=True, show_features_importance=True, init_model=None, use_best_model: bool=True, early_stopping_rounds: int=None):
        """

        :param plot:
        :param verbose:
        :param show_features_importance:
        :param init_model:
            CatBoost class or string, [default=None]
            Continue training starting from the existing model.
            If this parameter is a string, load initial model from the path specified by this string.
        :param use_best_model:
        :param early_stopping_rounds:
        :return:
        """
        from catboost import Pool
        import pandas as pd
        features, labels, cat_cols = self._data_processor.train_input_fn()
        train_data = Pool(data=features,
                          label=labels,
                          cat_features=cat_cols)
        df_val, y_val, cat_cols = self._data_processor.eval_input_fn()
        val_data = Pool(data=df_val, label=y_val, cat_features=cat_cols)
        self._model.fit(train_data, eval_set=val_data, plot=plot, verbose=verbose)
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

    def cross_val(self, nfold=3, shuffle=True, stratified=None, plot=True, partition_random_seed: int=14):
        """

        :param nfold:
        :param shuffle:
        :param stratified:
        :param plot:
        :param partition_random_seed:
        :return:
            cv results : pandas.core.frame.DataFrame with cross-validation results
            columns are: test-error-mean  test-error-std  train-error-mean  train-error-std
        """
        from catboost import Pool, cv
        import numpy as np
        features, labels, cat_cols = self._data_processor.cv_input_fn()
        cv_data = Pool(data=features,
                          label=labels,
                          cat_features=cat_cols)
        cv_result = cv(cv_data, self._params, nfold=nfold, shuffle=shuffle, stratified=stratified, plot=plot, partition_random_seed=partition_random_seed)
        print('Best validation {} score: {:.2f}±{:.2f} on step {}'.format(
            self._params['custom_metric'],
            np.max(cv_result[f'test-{self._params["custom_metric"]}-mean']),
            cv_result[f'test-{self._params["custom_metric"]}-std'][np.argmax(cv_result[f'test-{self._params["custom_metric"]}-mean'])],
            np.argmax(cv_result[f'test-{self._params["custom_metric"]}-mean'])
        ))
        print('Precise validation {} score: {}'.format(self._params['custom_metric'], np.max(cv_result[f'test-{self._params["custom_metric"]}-mean'])))
        return cv_result

    def baseline(self):
        import numpy as np
        cv_result = self.cross_val()
        self._params.update({
            'iterations': np.argmax(cv_result[f'test-{self._params["custom_metric"]}-mean']),
        })
        print(self._params)
        df_features_importance = self.train_eval()
        dfr = self.evaluate()
        return df_features_importance, dfr

    def test(self, df_test=None, cat_cols=None, explain=True):
        from catboost import Pool
        import pandas as pd
        if df_test is None:
            df_test, cat_cols = self._data_processor.test_input_fn()
        test_data = Pool(data=df_test,
                         cat_features=cat_cols)
        dfr = pd.DataFrame(df_test.index)
        y_test_hat = self._model.predict_proba(test_data)[:, 1]
        dfr['score'] = y_test_hat
        dfr['test_label'] = self._model.predict(test_data)
        s = dfr['test_label'].value_counts()
        print(s)
        print(f'su sample num：{s.loc[1] if 1 in s else 0}')
        if explain:
            rr = self.explain(df_test, cat_cols, dfr)
            return dfr, rr
        return dfr

    def explain(self, df_test, cat_cols, dfr):
        """模型解释"""
        from catboost import Pool
        import pandas as pd
        test_data = Pool(data=df_test, cat_features=cat_cols)
        shap_values = self._model.get_feature_importance(test_data, type='ShapValues')
        dfs = pd.DataFrame(shap_values[:, :-1], columns=df_test.columns, index=df_test.index)
        dfs_T = dfs.T
        ss = []
        for i in range(dfs_T.shape[1]):
            ss.append(dfs_T.iloc[:, i].copy().sort_values(ascending=False).iloc[:5])
        count = 0
        rr = []
        for line in dfr[dfr.test_label == 1].itertuples():
            rr.append({"id": line[0], "SCORE": round(line.score, 2),
                       "EXPLAIN": ','.join(
                           [f'{i[0]}:{round(i[1], 2)}' for i in list(zip(ss[count].index, ss[count].values))])})
            count += 1
        print(rr)
        return rr

    def save_model(self, model_path):
        """

        :param model_path: 'catboost_model.dump'
        :return:
        """
        self._model.save_model(model_path)

    def load_model(self, model_path):
        self._model = self.model_fn().load_model(model_path)


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
        r = self._model.test(features)
        dfr = pd.DataFrame(ys)
        dfr.columns = ['y_true']
        dfr['y_test'] = r
        print(f'mae: {mae(ys, r)}')
        print(f'mse: {mse(ys, r)}')
        print(f'rmse: {np.sqrt(mse(ys, r))}')
        return dfr

    def test(self, input_fn):
        import pandas as pd
        df_test, id_col_name = input_fn()  # id_col_name: 预测的标识列名称
        dfr = pd.DataFrame(df_test[id_col_name])
        r = self._model.test(df_test.values)

        dfr['y_test'] = r
        return dfr


class FastTextClsEstimator(ClsEstimator):
    """
    facebook fasttext==0.9.1
    """
    def __init__(self, params=None, data_processor=None, model_path=None):
        """

        :param params:
        """
        super().__init__(params=params, data_processor=data_processor, model_path=model_path)
        if self._model_path:
            self.load_model(self._model_path)
        else:
            self._model = None

    def model_fn(self):
        import fasttext
        return fasttext

    def train(self, train_file_path):
        """

        :param train_file_path:
        :return:
        """
        self._model = self.model_fn.train_supervised(train_file_path)
        return self._model

    def _print_results(self, N, p, r):
        print("N\t" + str(N))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))

    def evaluate(self, path, val_file_path, mode='simple', k=1):
        """
        To evaluate our model by computing the precision at 1 (P@1) and the recall on a test set, we use the test function
        :param path:
        :param val_file_path:
        :param mode:
            simple: 返回整体的精确率和召回率
            detail: 返回每个label的精确率和召回率和F1
        :param k:
            By default, predict returns only one label : the one with the highest probability.
            You can also predict more than one label by specifying the parameter k
        :return:
            N, p, r = 测试样本数，精确率，召回率
        """
        N, p, r = self._model.test(val_file_path, k)
        self._print_results(N, p, r)
        if mode == 'detail':
            r_dict = self._model.test_label(val_file_path, k)
            print(r_dict)
            return r_dict
        return N, p, r

    def test(self, test_data: list, k=1):
        """
        Given a string, get a list of labels and a list of
        corresponding probabilities. k controls the number
        of returned labels. A choice of 5, will return the 5
        most probable labels. By default this returns only
        the most likely label and probability.
        :param test_data:
        :param k:
        :return:
        """
        return self._model.predict(test_data)

    def save_model(self, model_path):
        """

        :param model_path: 'model_item.bin'
        :return:
        """
        self._model.save_model(model_path)

    def load_model(self, model_path):
        self._model = self.model_fn().load_model(model_path)


if __name__ == '__main__':
    params = {
        'iterations': 10,
        # learning_rate: 0.1,
        #     'custom_metric': 'AUC',
        'custom_metric': 'F1',
        'loss_function': 'CrossEntropy'
    }
    c = CatBoostClsEstimator(params=params)
    print(c._params)
    print(c._model)




