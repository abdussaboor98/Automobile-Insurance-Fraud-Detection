import pandas as pd
import numpy as np
# from imblearn.under_sampling import RandomUnderSampler
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality
from skopt import gp_minimize, dump, load
from skopt.space import Integer, Categorical, Real
# from collections import Counter
import os

# Load data
df = pd.read_csv(
    "../../data/Angoss Knowledge Seeker - carclaims.txt/carclaims_original.csv"
)
# Drop row with missing data
df.drop(df[df["DayOfWeekClaimed"] == "0"].index, inplace=True)
# Drop ID column
df.drop(columns="PolicyNumber", inplace=True)

# Load SDV metadata
metadata = Metadata.load_from_json(filepath="carclaims_metadata.json")

space = [
    Categorical([5000, 10000, 20000, 30000], name='tvae_epochs'),
    Integer(2, 6, name='tvae_compress_depth'),
    Categorical([128, 256, 512], name='tvae_compress_width'),
    Integer(2, 6, name='tvae_decompress_depth'),
    Categorical([128, 256, 512], name='tvae_decompress_width'),
    Categorical([64, 128, 256], name='tvae_embedding_dim'),
    Real(1e-5, 1e-1, name='l2scale')
]

def train_and_predict(params):
    try:
        tvae_epochs = params[0]
        tvae_compress_depth = params[1]
        tvae_compress_width = params[2]
        tvae_decompress_depth = params[3]
        tvae_decompress_width = params[4]
        tvae_embedding_dim = params[5]
        tvae_l2scale = params[6]

        tvae_compress_dims = [tvae_compress_width] * tvae_compress_depth
        tvae_decompress_dims = [tvae_decompress_width] * tvae_decompress_depth

        print(
            f"Running: tvae_epochs {tvae_epochs}, tvae_batch_size default, tvae_compress_dims {tvae_compress_dims}, tvae_decompress_dims {tvae_decompress_dims}, tvae_embedding_dim {tvae_embedding_dim}"
        )
        
        # print('Original dataset shape %s' % Counter(df['FraudFound']))
        # rus = RandomUnderSampler(random_state=42)    
        # X_rus, y_rus = rus.fit_resample(df.drop('FraudFound', axis=1), df['FraudFound'])

        # print('Undersampled dataset shape %s' % Counter(y_rus))
        # Create synthesizer
        synthesizer = TVAESynthesizer(
            metadata, 
            cuda=True,
            epochs=tvae_epochs,
            compress_dims=tvae_compress_dims,
            embedding_dim=tvae_embedding_dim,
            decompress_dims=tvae_decompress_dims,
            l2scale=tvae_l2scale,
            verbose=True
        )
        synthesizer.fit(df)

        # Create synthetic data
        synthetic_data = synthesizer.sample(
            num_rows=100_000,
            batch_size=1_000,
        )

        quality_report = evaluate_quality(
            real_data=df, synthetic_data=synthetic_data, metadata=metadata
        )
        quality_report_full = evaluate_quality(
            real_data=df, synthetic_data=synthetic_data, metadata=metadata
        )
        
        filename = f'full_data_synthesizer_{quality_report.get_score():.2f}.pkl'
        synthesizer.save(filepath='./models/' + filename)
        
        model_details = {
            'tvae_epochs': tvae_epochs,
            'tvae_compress_depth': tvae_compress_depth,
            'tvae_compress_width': tvae_compress_width,
            'tvae_decompress_depth': tvae_decompress_depth,
            'tvae_decompress_width': tvae_decompress_width,
            'tvae_embedding_dim': tvae_embedding_dim,
            'l2scale': tvae_l2scale,
            'filename': filename,
            'quality_score': quality_report.get_score(),
        }

        model_details_df = pd.DataFrame([model_details])
        csv_file_path = '~/fraud/models/model_details.csv'
        model_details_df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)

        return -quality_report.get_score()

    except Exception as e:
        print(f"Exception encountered: {e}")
        return 101

checkpoint_file = 'skopt_checkpoint_full_data.pkl'

# Define a callback to save the checkpoint after each iteration
def checkpoint_callback(res):
    dump(res, checkpoint_file, store_objective=False)
    print("Checkpoint saved.")
    
# Check if a checkpoint file exists and load it
try:
    result = load(checkpoint_file)
    print("Loaded previous skopt run from checkpoint.")
except FileNotFoundError:
    print("No checkpoint found. Starting a new skopt run.")
    result = None


# Run the optimization
result = gp_minimize(
    func=train_and_predict,
    dimensions=space,
    n_calls=100,
    random_state=42,
    callback=[checkpoint_callback],
    x0=result.x_iters if result else None,
    y0=result.func_vals if result else None
)

print(f"Best result: {result.x}")
