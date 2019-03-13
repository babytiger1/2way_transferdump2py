from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

import numpy as np
import tensorflow as tf

# fix ImportError: No mudule named lib.*
import sys
import xgb_model_zzr

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
from lib.joint import BuildClassifier

# wide columns
categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
categorical_column_with_vocabulary_file = tf.feature_column.categorical_column_with_vocabulary_file
crossed_column = tf.feature_column.crossed_column
bucketized_column = tf.feature_column.bucketized_column
# deep columns
embedding_column = tf.feature_column.embedding_column
indicator_column = tf.feature_column.indicator_column
numeric_column = tf.feature_column.numeric_column


def normalizer_fn_builder(scaler, normalization_params):
    """normalizer_fn builder"""
    if scaler == 'min_max':
        if normalization_params[0] == normalization_params[1]:
            return lambda x: x - x
        else:
            # return lambda x: 0 if x == 0 else 0
            return lambda x: (x - normalization_params[0]) / (normalization_params[1] - normalization_params[0])
    elif scaler == 'standard':
        return lambda x: (x - normalization_params[0]) / normalization_params[1]
    else:
        return lambda x: tf.log1p(x - normalization_params[0]) / tf.log1p(normalization_params[1] - normalization_params[0])

def transform_XGB_Model(x,allFeature):
	dictDate = dict(zip(x,allFeature))
	res = xgb_model_zzr.xgb_predict(dictDate)
	return res

def transform2cat：
	allFeature = []
	catColumn = []
	conColumn = []
	for col in catColumn:
		allFeature.append( categorical_column_with_identity(col) )
	for col in conColumn:
		allFeature.append( numeric_column(col) )

    normalizer_fn = transform_XGB_Model(allFeature)

	resFeature = numeric_column(allFeature,
            default_value=0,
            dtype=tf.float32,
            normalizer_fn=lambda x:transform_XGB_Model(x,allFeature) ))
