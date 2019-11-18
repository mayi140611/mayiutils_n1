#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: DataProcessor.py
@time: 2019-11-18 15:18

为Estimator准备输入
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

    def train_input_fn(self, features, labels, batch_size=None):
        pass

    def val_input_fn(self, features, labels, batch_size=None):
        pass

    def predict_input_fn(self, features, batch_size=None):
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