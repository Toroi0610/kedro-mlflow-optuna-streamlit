# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

import pickle
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

import optuna
from optuna.integration.mlflow import MLflowCallback


mlflc = MLflowCallback(
    tracking_uri="./mlruns",
    metric_name='mean squared error',
)


def score(y_true: np.ndarray, y_pred:np.ndarray) -> float:
    """[summary]

    Args:
        y_true (np.ndarray): target value
        y_pred (np.ndarray): Predictive value

    Returns:
        float: score
    """
    return mean_absolute_error(y_true, y_pred)


def training(train_x: pd.DataFrame,
             train_y: pd.DataFrame,
             parameters: Dict[str, Any]
):
    """[summary]

    Args:
        train_x (pd.DataFrame): [description]
        train_y (pd.DataFrame): [description]
        model ([type]): [description]
        parameters (Dict[str, Any]): [description]
    """

    model_dir = "./data/06_models/"
    def objective(trial):
        model = create_lightgbm_model_from_suggested_params(trial)
        score = fitting(train_x, train_y, model, parameters)
        # Save a trained model to a file.
        with open(model_dir+'lgbm_{}.pickle'.format(trial.number), 'wb') as fout:
            pickle.dump(model, fout)
        return score

    study = optuna.load_study(study_name=parameters["study_name"],
                              storage=parameters['database_path'])
    study.optimize(objective, n_trials=1, callbacks=[mlflc])

    # Load best model
    with open(model_dir+'lgbm_{}.pickle'.format(study.best_trial.number), 'rb') as fin:
        best_model = pickle.load(fin)

    return best_model


def fitting(
    train_x: pd.DataFrame, train_y: pd.DataFrame,
    model, parameters: Dict[str, Any]
) -> np.ndarray:
    """[summary]

    Args:
        train_x (pd.DataFrame): training input
        train_y (pd.DataFrame): training target
        model ([type]): regressor model
        parameters (Dict[str, Any]): training configuration

    Returns:
        np.ndarray: score
    """

    fold = KFold(n_splits=parameters['folds'], random_state=parameters['random_state'])

    ### run model with kfold
    mses_valid = []
    for k, (train_index, valid_index) in enumerate(fold.split(train_x, train_y)):
        #print(train_index)
        X_train, X_valid = train_x.iloc[train_index], train_x.iloc[valid_index]
        y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]

        model = model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

        y_train_pred = model.predict(X_train)
        y_valid_pred = model.predict(X_valid)

        mse_train = score(y_train, y_train_pred)
        mse_valid = score(y_valid, y_valid_pred)
        print('Fold {n_folds}: train SCORE is {train: .3f} valid SCORE is {valid: .3f}'.format(n_folds=k+1, train=mse_train, valid=mse_valid))
        mses_valid.append(mse_valid)

    return np.mean(mses_valid)


# def create_lightgbm_model(params):
#     model = lgb.LGBMRegressor()
#     return model.set_params(**params)


def create_lightgbm_model_from_suggested_params(trial) -> "Model":
    """Node for creating a lightgbm model.

    Args:
        trial: optuna.trial

    Returns:
        "Model": model which is created suggested params
    """

    lgb_params = {
        'metric'               : 'mae',
        'num_iterations'       : trial.suggest_int('num_iterations', 100, 1000),
        'num_leaves'           : trial.suggest_int('num_leaves', 3, 10),
        'learning_rate'        : trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'max_depth'            : trial.suggest_int('max_depth', 3, 10),
        'n_estimators'         : trial.suggest_int('n_estimators', 100, 1000)
        }

    model = lgb.LGBMRegressor(**lgb_params)

    return model


def predict_lightgbm_model(
    model, test_x: pd.DataFrame, test_id: pd.Series,
):
    test_pred = model.predict(test_x)
    test_id["SalePrice"] = test_pred
    test_id.to_csv("submition.csv", index=False)
    return test_id


def train_model(
    train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]
) -> np.ndarray:
    """Node for training a simple multi-class logistic regression model. The
    number of training iterations as well as the learning rate are taken from
    conf/project/parameters.yml. All of the data as well as the parameters
    will be provided to this function at the time of execution.
    """
    num_iter = parameters["example_num_train_iter"]
    lr = parameters["example_learning_rate"]
    X = train_x.to_numpy()
    Y = train_y.to_numpy()

    # Add bias to the features
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)

    weights = []
    # Train one model for each class in Y
    for k in range(Y.shape[1]):
        # Initialise weights
        theta = np.zeros(X.shape[1])
        y = Y[:, k]
        for _ in range(num_iter):
            z = np.dot(X, theta)
            h = _sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            theta -= lr * gradient
        # Save the weights for each model
        weights.append(theta)

    # Return a joint multi-class model with weights for all classes
    return np.vstack(weights).transpose()


def predict(model: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set.
    """
    X = test_x.to_numpy()

    # Add bias to the features
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)

    # Predict "probabilities" for each class
    result = _sigmoid(np.dot(X, model))

    # Return the index of the class with max probability for all samples
    return np.argmax(result, axis=1)


def report_accuracy(predictions: np.ndarray, test_y: pd.DataFrame) -> None:
    """Node for reporting the accuracy of the predictions performed by the
    previous node. Notice that this function has no outputs, except logging.
    """
    # Get true class index
    target = np.argmax(test_y.to_numpy(), axis=1)
    # Calculate accuracy of predictions
    accuracy = np.sum(predictions == target) / target.shape[0]
    # Log the accuracy of the model
    log = logging.getLogger(__name__)
    log.info("Model accuracy on test set: %0.2f%%", accuracy * 100)


def _sigmoid(z):
    """A helper sigmoid function used by the training and the scoring nodes."""
    return 1 / (1 + np.exp(-z))
