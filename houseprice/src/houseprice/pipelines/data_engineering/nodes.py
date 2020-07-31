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

PLEASE DELETE THIS FILE ONCE YOU START WORKING ON YOUR OWN PROJECT!
"""

from typing import Any, Dict

import pandas as pd
import featuretools as ft
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder


# def split_data(data: pd.DataFrame, example_test_data_ratio: float) -> Dict[str, Any]:
#     """Node for splitting the classical Iris data set into training and test
#     sets, each split into features and labels.
#     The split ratio parameter is taken from conf/project/parameters.yml.
#     The data and the parameters will be loaded and provided to your function
#     automatically when the pipeline is executed and it is time to run this node.
#     """
#     data.columns = [
#         "sepal_length",
#         "sepal_width",
#         "petal_length",
#         "petal_width",
#         "target",
#     ]
#     classes = sorted(data["target"].unique())
#     # One-hot encoding for the target variable
#     data = pd.get_dummies(data, columns=["target"], prefix="", prefix_sep="")

#     # Shuffle all the data
#     data = data.sample(frac=1).reset_index(drop=True)

#     # Split to training and testing data
#     n = data.shape[0]
#     n_test = int(n * example_test_data_ratio)
#     training_data = data.iloc[n_test:, :].reset_index(drop=True)
#     test_data = data.iloc[:n_test, :].reset_index(drop=True)

#     # Split the data to features and labels
#     train_data_x = training_data.loc[:, "sepal_length":"petal_width"]
#     train_data_y = training_data[classes]
#     test_data_x = test_data.loc[:, "sepal_length":"petal_width"]
#     test_data_y = test_data[classes]

#     # When returning many variables, it is a good practice to give them names:
#     return dict(
#         train_x=train_data_x,
#         train_y=train_data_y,
#         test_x=test_data_x,
#         test_y=test_data_y,
#     )


def standard_scaler(training_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
    pass


def add_features(
    data: pd.DataFrame
) -> (pd.DataFrame):
    pass


def to_categorical(
    training_data: pd.DataFrame, test_data: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame):

    categorical_columns_list = list(training_data.columns[training_data.dtypes==object])
    ce_be = BinaryEncoder(cols=categorical_columns_list, handle_unknown="inpute")
    training_data_ce_binary = ce_be.fit_transform(training_data)
    test_data_ce_binary = ce_be.transform(test_data)

    return dict(train_data_categorical=training_data_ce_binary,
                test_data_categorical=test_data_ce_binary)


def get_ordinal_mapping(obj):
    """Ordinal Encodingの対応をpd.DataFrameで返す
    param: obj : category_encodersのインスタンス
    return pd.DataFrame
    """
    listx = list()
    for x in obj.category_mapping:
        listx.extend([tuple([x['col']])+ i for i in x['mapping']])
    df_ord_map = pd.DataFrame(listx,columns=['column','label','ord_num'])
    return df_ord_map


def load_data(training_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:

    """load data for train_data, test_data

    Args:
        training_data (pd.DataFrame): train.csv
        test_data (pd.DataFrame): test.csv

    Returns:
        Dict[str, Any]: Dictionary to pipeline.py
    """

    input_columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
                     'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
                     'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                     'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                     'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                     'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                     'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
                     'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
                     'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
                     'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
                     'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
                     'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
                     'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
                     'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
                     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
                     'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
                     'SaleCondition']

    target = "SalePrice"
    train_y = training_data.loc[:, target]
    train_x = training_data.loc[:, input_columns]
    test_id = test_data[["Id"]]

    # When returning many variables, it is a good practice to give them names:
    return dict(
        train_x=train_x,
        train_y=train_y,
        test_id=test_id,
        test_x=test_data.loc[:, input_columns]
    )