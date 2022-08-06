"""Unearthed Training Template"""
"Making a change"

import sys, pickle, logging, argparse
from io import StringIO

from os import getenv
from os.path import abspath, join

import numpy as np
import pandas as pd


import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.model_selection import StratifiedKFold


from preprocess import preprocess
from ensemble_model import EnsembleModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Work around for a SageMaker path issue
# (see https://github.com/aws/sagemaker-python-sdk/issues/648)
# WARNING - removing this may cause the submission process to fail
if abspath("/opt/ml/code") not in sys.path:
    sys.path.append(abspath("/opt/ml/code"))

def train(args):
    """Train

    Your model code goes here.
    """
    logger.info("calling training function")

    # If you require any particular preprocessing to create features then this
    # *must* be contained in the preprocessing function for the Unearthed pipeline
    # apply it to the private data
    df = preprocess(join(args.data_dir, "public.csv.gz"), False)
    cols = df.columns.to_list()
    cols.remove('ZN_PPM')

    # quantile loss models for baseline
    lower_light = LGBMRegressor(objective="quantile", alpha=1-0.95, n_jobs = -1)
    lower_light.fit(df[cols], df['ZN_PPM'])

    upper_light = LGBMRegressor(objective = 'quantile', alpha = 0.95, n_jobs = -1)
    upper_light.fit(df[cols], df['ZN_PPM'])

    models = [lower_light, upper_light]
    
    
    df['bin'] = (df['ZN_PPM'].rank(pct= True) * 100).astype(int)
        
    
    N_folds = 20
    seed = 16
    
    df['fold'] = -1
    skf = StratifiedKFold(n_splits=N_folds, shuffle=True, random_state=seed)
    for f, (_, idxs) in enumerate(skf.split(df, df['bin'])):
        df.loc[idxs, 'fold'] = f
        
        
    
        
    for F in range(N_folds):
        
        print('FOLD : ', F)
        
        #train = df[df['fold'] != F].reset_index(drop=True)
        #valid = df[df['fold'] == F].reset_index(drop=True)
        
        
        callbacks = [lgb.early_stopping(100, verbose=0), lgb.log_evaluation(period=25)]

        
        model = LGBMRegressor(boosting_type = 'gbdt', 
                              num_leaves = 255, 
                              n_estimators = 1500, 
                              objective = 'regression_l1',
                              n_jobs = -1, 
                              random_state = 16 + F, 
                              verbosity = -1)
                              
        #model.fit(train[cols], train['ZN_PPM'], 
        #         eval_set = [(train[cols], train['ZN_PPM']), (valid[cols], valid['ZN_PPM'])],
        #         eval_names = ['train', 'valid'],
        #         eval_metric = 'mae',
        #         callbacks=callbacks)
        
        model.fit(df[cols], df['ZN_PPM'])
        
        models.append(model)
    
    
    
    

    # save the model to disk
    save_model(EnsembleModel(models), args.model_dir)


def save_model(model, model_dir):
    """Save model to a binary file.

    This function must write the model to disk in a format that can
    be loaded from the model_fn.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info(f"saving model to {model_dir}")
    with open(join(model_dir, "model.pkl"), "wb") as model_file:
        pickle.dump(model, model_file)


def model_fn(model_dir):
    """Load model from binary file.

    This function loads the model from disk. It is called by SageMaker.

    WARNING - modifying this function may case the submission process to fail.
    """
    logger.info("loading model")
    with open(join(model_dir, "model.pkl"), "rb") as file:
        return pickle.load(file)


def input_fn(input_data, content_type):
    """Take request data and de-serialize the data into an object for prediction.

    In the Unearthed submission pipeline the data is passed as "text/csv". This
    function reads the CSV into a Pandas dataframe ready to be passed to the model.

    WARNING - modifying this function may cause the submission process to fail.
    """
    logger.info("receiving preprocessed input")

    # this call must result in a dataframe or nparray that matches your model
    input = pd.read_csv(StringIO(input_data), index_col=0, parse_dates=True)
    logger.info(f"preprocessed input has shape {input.shape}")
    return input


if __name__ == "__main__":
    """Training Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed train" command.

    WARNING - modifying this function may cause the submission process to fail.

    The main function must call preprocess, arrange th
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default=getenv("SM_MODEL_DIR", "/opt/ml/models")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    train(parser.parse_args())
