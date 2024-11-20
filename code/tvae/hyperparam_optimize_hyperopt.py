import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL
from hyperopt.mongoexp import MongoTrials
from collections import Counter
from sdv.evaluation.single_table import evaluate_quality

# Load data
df = pd.read_csv(
    "../../data/Angoss Knowledge Seeker - carclaims.txt/carclaims_original.csv"
)
# Drop row with missing data
df.drop(df[df["DayOfWeekClaimed"] == "0"].index, inplace=True)
# Drop ID column
df.drop(columns="PolicyNumber", inplace=True)

# Train-test split
carclaims_train, carclaims_test = train_test_split(df, test_size=0.2, random_state=141)

# Load SDV metadata
metadata = Metadata.load_from_json(filepath="carclaims_metadata.json")


space = {
    'tvae_epochs': hp.quniform('tvae_epochs', 20000, 30000, 1),
    'tvae_compress_depth': hp.quniform('tvae_compress_depth', 2, 6, 1),
    'tvae_compress_width': hp.choice('tvae_compress_width', [64, 128, 256, 512, 1024]),
    'tvae_decompress_depth': hp.quniform('tvae_decompress_depth', 2, 6, 1),
    'tvae_decompress_width': hp.choice('tvae_decompress_width', [64, 128, 256, 512, 1024]),
    'tvae_embedding_dim': hp.choice('tvae_embedding_dim', [64, 128, 256, 512, 1024])
}

def train_and_predict(params):
    try:
        tvae_epochs = int(params['tvae_epochs'])
        tvae_compress_depth = int(params['tvae_compress_depth'])
        tvae_compress_width = params['tvae_compress_width']
        tvae_decompress_depth = int(params['tvae_decompress_depth'])
        tvae_decompress_width = params['tvae_decompress_width']
        tvae_embedding_dim = params['tvae_embedding_dim']

        tvae_compress_dims = [tvae_compress_width] * tvae_compress_depth
        tvae_decompress_dims = [tvae_decompress_width] * tvae_decompress_depth

        print(
            f"Running: tvae_epochs {tvae_epochs}, tvae_batch_size default, tvae_compress_dims {tvae_compress_dims}, tvae_decompress_dims {tvae_decompress_dims}, tvae_embedding_dim {tvae_embedding_dim}"
        )

        print('Original dataset shape %s' % Counter(carclaims_train['FraudFound']))
        rus = RandomUnderSampler(random_state=42)    
        X_rus, y_rus = rus.fit_resample(carclaims_train.drop('FraudFound', axis=1), carclaims_train['FraudFound'])

        print('Undersampled dataset shape %s' % Counter(y_rus))
        # Create synthesizer
        synthesizer = TVAESynthesizer(
            metadata, cuda=True,
            epochs=tvae_epochs,  # 300
            compress_dims=tvae_compress_dims,  # (128, 128)
            embedding_dim=tvae_embedding_dim,  # 128
            decompress_dims=tvae_decompress_dims,  # (128, 128),
        )
        synthesizer.fit(pd.concat([X_rus, y_rus], axis=1))

        # Create balanced synthetic data
        synthetic_data = synthesizer.sample(
            num_rows=100_000,
            batch_size=1_000
        )

        quality_report = evaluate_quality(
            real_data=df, synthetic_data=synthetic_data, metadata=metadata
        )

        return {'loss': -quality_report.get_score(), 'status': STATUS_OK, 'quality': quality_report.get_score(), "tvae_epochs": tvae_epochs, "tvae_batch_size": "default", "tvae_compress_dims": tvae_compress_dims, "tvae_decompress_dims": tvae_decompress_dims, "tvae_embedding_dim": tvae_embedding_dim}

    except Exception as e:
        print(f"Exception encountered: {e}")
        return {'status': STATUS_FAIL, 'exception': str(e)}

trials = MongoTrials('mongo://localhost:1234/hyperopt/jobs', exp_key='tvae_quality')

best = fmin(
    fn=train_and_predict,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials,
    rstate=np.random.default_rng(42),
    show_progressbar=True,
)

print(f"Best result: {best}")
