#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@file: DNNClassifier.py
@time: 2019-09-05 15:49
"""
from mayiutils.my_estimators.Estimator import Estimator


class DNNClassifier(Estimator):
    """
    A classifier for TensorFlow DNN models.
    https://github.com/tensorflow/estimator/blob/ef31ca605f5565751261436058b62ebad1c06de2/tensorflow_estimator/python/estimator/canned/dnn.py#L774
    """
    def __init__(
            self,
            hidden_units,
            feature_columns,
            model_dir=None,
            n_classes=2,
            weight_column=None,
            label_vocabulary=None,
            optimizer='Adagrad',
            activation_fn=nn.relu,
            dropout=None,
            config=None,
            warm_start_from=None,
            loss_reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            batch_norm=False,
    ):
        pass





