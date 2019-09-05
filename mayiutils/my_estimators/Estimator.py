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
    Estimator class to train and evaluate models.
  The `Estimator` object wraps a model which is specified by a `model_fn`,
  which, given inputs and a number of other parameters, returns the ops
  necessary to perform training, evaluation, or predictions.
  All outputs (checkpoints, event files, etc.) are written to `model_dir`, or a
  subdirectory thereof. If `model_dir` is not set, a temporary directory is
  used.
    The `config` argument can be passed `tf.estimator.RunConfig` object containing
  information about the execution environment. It is passed on to the
  `model_fn`, if the `model_fn` has a parameter named "config" (and input
  functions in the same manner). If the `config` parameter is not passed, it is
  instantiated by the `Estimator`. Not passing config means that defaults useful
  for local execution are used. `Estimator` makes config available to the model
  (for instance, to allow specialization based on the number of workers
  available), and also uses some of its fields to control internals, especially
  regarding checkpointing.

    The `params` argument contains hyperparameters. It is passed to the
  `model_fn`, if the `model_fn` has a parameter named "params", and to the input
  functions in the same manner. `Estimator` only passes params along, it does
  not inspect it. The structure of `params` is therefore entirely up to the
  developer.
  None of `Estimator`'s methods can be overridden in subclasses (its
  constructor enforces this). Subclasses should use `model_fn` to configure
  the base class, and may add methods implementing specialized functionality.
    """

    def __init__(self, model_fn, model_dir=None, config=None, params=None,
               warm_start_from=None):
        """Constructs an `Estimator` instance.
        See [estimators](https://tensorflow.org/guide/estimators) for more
        information.
        To warm-start an `Estimator`:
        ```python
        estimator = tf.estimator.DNNClassifier(
            feature_columns=[categorical_feature_a_emb, categorical_feature_b_emb],
            hidden_units=[1024, 512, 256],
            warm_start_from="/path/to/checkpoint/dir")
        ```
        For more details on warm-start configuration, see
        `tf.estimator.WarmStartSettings`.
        Args:
          model_fn: Model function. Follows the signature:
            * Args:
              * `features`: This is the first item returned from the `input_fn`
                     passed to `train`, `evaluate`, and `predict`. This should be a
                     single `tf.Tensor` or `dict` of same.
              * `labels`: This is the second item returned from the `input_fn`
                     passed to `train`, `evaluate`, and `predict`. This should be a
                     single `tf.Tensor` or `dict` of same (for multi-head models).
                     If mode is `tf.estimator.ModeKeys.PREDICT`, `labels=None` will
                     be passed. If the `model_fn`'s signature does not accept
                     `mode`, the `model_fn` must still be able to handle
                     `labels=None`.
              * `mode`: Optional. Specifies if this is training, evaluation or
                     prediction. See `tf.estimator.ModeKeys`.
              * `params`: Optional `dict` of hyperparameters.  Will receive what
                     is passed to Estimator in `params` parameter. This allows
                     to configure Estimators from hyper parameter tuning.
              * `config`: Optional `estimator.RunConfig` object. Will receive what
                     is passed to Estimator as its `config` parameter, or a default
                     value. Allows setting up things in your `model_fn` based on
                     configuration such as `num_ps_replicas`, or `model_dir`.
            * Returns:
              `tf.estimator.EstimatorSpec`
          model_dir: Directory to save model parameters, graph and etc. This can
            also be used to load checkpoints from the directory into an estimator to
            continue training a previously saved model. If `PathLike` object, the
            path will be resolved. If `None`, the model_dir in `config` will be used
            if set. If both are set, they must be same. If both are `None`, a
            temporary directory will be used.
          config: `estimator.RunConfig` configuration object.
          params: `dict` of hyper parameters that will be passed into `model_fn`.
                  Keys are names of parameters, values are basic python types.
          warm_start_from: Optional string filepath to a checkpoint or SavedModel to
                           warm-start from, or a `tf.estimator.WarmStartSettings`
                           object to fully configure warm-starting.  If the string
                           filepath is provided instead of a
                           `tf.estimator.WarmStartSettings`, then all variables are
                           warm-started, and it is assumed that vocabularies
                           and `tf.Tensor` names are unchanged.
        Raises:
          ValueError: parameters of `model_fn` don't match `params`.
          ValueError: if this is called via a subclass and if that class overrides
            a member of `Estimator`.
        """
    def model_fn(self, features, labels, mode, config):
        """
        模型函数
        :param features:
        :param labels:
        :param mode:
        :param config:
        :return:
        """
        pass

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













