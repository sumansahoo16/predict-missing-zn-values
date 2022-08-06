"""Preprocess Template.

This script will be invoked in two ways during the Unearthed scoring pipeline:
 - first during model training on the 'public' dataset
 - secondly during generation of predictions on the 'private' dataset
"""
import argparse
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

target_columns = [
    'ZN_PPM'
]

def preprocess(data_file, drop_targets):
    """Apply preprocessing and featurization steps to each file in the data directory.

    Your preprocessing and feature generation goes here.
    """
    logger.info(f"running preprocess on {data_file}")
    
    float32_cols = {'AG_PPM': np.float32,
 'AL_PPM': np.float32,
 'AL2O3_PPM': np.float32,
 'AS_PPM': np.float32,
 'AU_PPM': np.float32,
 'B_PPM': np.float32,
 'BA_PPM': np.float32,
 'BE_PPM': np.float32,
 'BI_PPM': np.float32,
 'BR_PPM': np.float32,
 'C_PPM': np.float32,
 'CA_PPM': np.float32,
 'CAO_PPM': np.float32,
 'CD_PPM': np.float32,
 'CE_PPM': np.float32,
 'CO_PPM': np.float32,
 'CR_PPM': np.float32,
 'CS_PPM': np.float32,
 'CU_PPM': np.float32,
 'DY_PPM': np.float32,
 'ER_PPM': np.float32,
 'EU_PPM': np.float32,
 'F_PPM': np.float32,
 'FE_PPM': np.float32,
 'FE2O3_PPM': np.float32,
 'GA_PPM': np.float32,
 'GD_PPM': np.float32,
 'GE_PPM': np.float32,
 'HF_PPM': np.float32,
 'HG_PPM': np.float32,
 'HO_PPM': np.float32,
 'I_PPM': np.float32,
 'IN_PPM': np.float32,
 'IR_PPM': np.float32,
 'K_PPM': np.float32,
 'K2O_PPM': np.float32,
 'LA_PPM': np.float32,
 'LI_PPM': np.float32,
 'LU_PPM': np.float32,
 'MG_PPM': np.float32,
 'MGO_PPM': np.float32,
 'MN_PPM': np.float32,
 'MNO_PPM': np.float32,
 'MO_PPM': np.float32,
 'NA_PPM': np.float32,
 'NA2O_PPM': np.float32,
 'NB_PPM': np.float32,
 'ND_PPM': np.float32,
 'NI_PPM': np.float32,
 'P_PPM': np.float32,
 'P2O5_PPM': np.float32,
 'PB_PPM': np.float32,
 'PD_PPM': np.float32,
 'PR_PPM': np.float32,
 'PT_PPM': np.float32,
 'RB_PPM': np.float32,
 'RE_PPM': np.float32,
 'RH_PPM': np.float32,
 'RU_PPM': np.float32,
 'S_PPM': np.float32,
 'SB_PPM': np.float32,
 'SC_PPM': np.float32,
 'SE_PPM': np.float32,
 'SI_PPM': np.float32,
 'SIO2_PPM': np.float32,
 'SM_PPM': np.float32,
 'SN_PPM': np.float32,
 'SR_PPM': np.float32,
 'TA_PPM': np.float32,
 'TB_PPM': np.float32,
 'TE_PPM': np.float32,
 'TH_PPM': np.float32,
 'TI_PPM': np.float32,
 'TIO2_PPM': np.float32,
 'TL_PPM': np.float32,
 'TM_PPM': np.float32,
 'U_PPM': np.float32,
 'V_PPM': np.float32,
 'W_PPM': np.float32,
 'Y_PPM': np.float32,
 'YB_PPM': np.float32,
 'ZR_PPM': np.float32}

    # read the data file
    df = pd.read_csv(data_file, parse_dates=True, engine='c', dtype=float32_cols)
    logger.info(f"data read from {data_file} has shape of {df.shape}")

    # add preprocessing here

    # Optionally drop target columns depending on the context.
    try:
        if drop_targets:
            df.drop(columns=target_columns, inplace=True)
    except KeyError:
        pass
        
    DROP = [ 'BR_PPM',
             'C_PPM',
             'ER_PPM',
             'F_PPM',
             'FE2O3_PPM',
             'I_PPM',
             'IR_PPM',
             'LU_PPM',
             'NA2O_PPM',
             'RH_PPM',
             'RU_PPM',
             'SM_PPM',
             'TB_PPM',
             'TM_PPM',
             'YB_PPM']

    df.drop(columns=DROP, inplace=True)
    
    if 'Unnamed: 0' in df.columns : df.drop(columns=['Unnamed: 0'], inplace=True)
    
    cols = df.columns[:-1].to_list()
    df['min'] = df[cols].min(axis = 1)
    df['max'] = df[cols].max(axis = 1)
    df['sum'] = df[cols].sum(axis = 1)
    df['avg'] = df[cols].isna().sum(axis = 1 )
    df['avg'] = df['sum'] / df['avg']

    logger.info(f"data after preprocessing has shape of {df.shape}")
    return df


if __name__ == "__main__":
    """Preprocess Main

    The main function is called by both Unearthed's SageMaker pipeline and the
    Unearthed CLI's "unearthed preprocess" command.

    WARNING - modifying this file may cause the submission process to fail.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="/opt/ml/processing/input/public/public.csv.gz"
    )
    parser.add_argument(
        "--output", type=str, default="/opt/ml/processing/output/preprocess/public.csv"
    )
    args, _ = parser.parse_known_args()

    # call preprocessing on private data
    df = preprocess(args.input, True)

    logger.info(f"preprocessed result shape is {df.shape}")

    # write to the output location
    df.to_csv(args.output)
