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
    'tvae_epochs': hp.quniform('tvae_epochs', 10000, 30000, 1),
    # 'tvae_batch_size': hp.quniform('tvae_batch_size', 100, 500, 1),
    'tvae_compress_depth': hp.quniform('tvae_compress_depth', 2, 6, 1),
    'tvae_compress_width': hp.choice('tvae_compress_width', [64, 128, 256, 512, 1024]),
    'tvae_decompress_depth': hp.quniform('tvae_decompress_depth', 2, 6, 1),
    'tvae_decompress_width': hp.choice('tvae_decompress_width', [64, 128, 256, 512, 1024]),
    'tvae_embedding_dim': hp.choice('tvae_embedding_dim', [64, 128, 256, 512, 1024]),
    'rf_n_estimators': hp.quniform('rf_n_estimators', 64, 512, 1),
    'rf_criterion': hp.choice('rf_criterion', ["gini", "entropy", "log_loss"]),
    'rf_max_depth': hp.quniform('rf_max_depth', 2, 50, 1),
    'rf_min_samples_split': hp.quniform('rf_min_samples_split', 2, 100, 1),
    'rf_min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 100, 1),
}
## rf_search
# space = {
#     'tvae_epochs': hp.quniform('tvae_epochs', 25000, 30000, 1),
#     # 'tvae_batch_size': hp.quniform('tvae_batch_size', 100, 500, 1),
#     'tvae_compress_depth': hp.quniform('tvae_compress_depth', 5, 6, 1),
#     'tvae_compress_width': hp.choice('tvae_compress_width', [512]),
#     'tvae_decompress_depth': hp.quniform('tvae_decompress_depth', 5, 6, 1),
#     'tvae_decompress_width': hp.choice('tvae_decompress_width', [512]),
#     'tvae_embedding_dim': hp.choice('tvae_embedding_dim', [512]),
#     'rf_n_estimators': hp.quniform('rf_n_estimators', 64, 512, 1),
#     'rf_criterion': hp.choice('rf_criterion', ["gini", "entropy", "log_loss"]),
#     'rf_max_depth': hp.quniform('rf_max_depth', 2, 50, 1),
#     'rf_min_samples_split': hp.quniform('rf_min_samples_split', 2, 100, 1),
#     'rf_min_samples_leaf': hp.quniform('rf_min_samples_leaf', 1, 100, 1),
# }

def train_and_predict(params):
    try:
        tvae_epochs = int(params['tvae_epochs'])
        tvae_compress_depth = int(params['tvae_compress_depth'])
        tvae_compress_width = params['tvae_compress_width']
        tvae_decompress_depth = int(params['tvae_decompress_depth'])
        tvae_decompress_width = params['tvae_decompress_width']
        tvae_embedding_dim = params['tvae_embedding_dim']
        rf_n_estimators = int(params['rf_n_estimators'])
        rf_criterion = params['rf_criterion']
        rf_max_depth = int(params['rf_max_depth'])
        rf_min_samples_split = int(params['rf_min_samples_split'])
        rf_min_samples_leaf = int(params['rf_min_samples_leaf'])
    
        tvae_compress_dims = [tvae_compress_width] * tvae_compress_depth
        tvae_decompress_dims = [tvae_decompress_width] * tvae_decompress_depth

        print(f'Running: tvae_epochs {tvae_epochs}, tvae_batch_size default, tvae_compress_dims {tvae_compress_dims}, tvae_decompress_dims {tvae_decompress_dims}, tvae_embedding_dim {tvae_embedding_dim}, rf_n_estimators {rf_n_estimators}, rf_criterion {rf_criterion}, rf_max_depth {rf_max_depth}, rf_min_samples_split {rf_min_samples_split}, rf_min_samples_leaf {rf_min_samples_leaf},')

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

        major_cnt = carclaims_train['FraudFound'].value_counts()['No']
        minor_cnt = carclaims_train['FraudFound'].value_counts()['Yes']

        balance_cnt = major_cnt - minor_cnt

        # Conditions for balancing the data
        fraud_samples = Condition(
            num_rows=20_000 + balance_cnt,
            column_values={'FraudFound': 'Yes'}
        )

        non_fraud_samples = Condition(
            num_rows=20_000,
            column_values={'FraudFound': 'No'}
        )

        # Create balanced synthetic data
        synthetic_data = synthesizer.sample_from_conditions(
            # conditions=[fraud_samples],
            conditions=[fraud_samples, non_fraud_samples], # Balance and oversample both classes
            batch_size=1_000
        )

        balanced_data = pd.concat([carclaims_train, synthetic_data], axis=0).reset_index(drop=True)
        carclaims_test.reset_index(drop=True)
        print('Balanced dataset shape %s' % Counter(balanced_data['FraudFound']))

        # X y split
        X_train = balanced_data.drop('FraudFound', axis=1)
        y_train = balanced_data['FraudFound']

        X_test = carclaims_test.drop('FraudFound', axis=1)
        y_test = carclaims_test['FraudFound']

        # Encode target variable
        y_train = y_train.map({'Yes': 1, 'No': 0})
        y_test = y_test.map({'Yes': 1, 'No': 0})

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Lebel Encode features
        column_labels = {
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'DayOfWeek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'DayOfWeekClaimed': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'MonthClaimed': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'AgeOfPolicyHolder': ['16 to 17', '18 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 65', 'over 65'],
            'NumberOfSuppliments': ['none', '1 to 2', '3 to 5', 'more than 5'],
            'AddressChange-Claim': ['no change', 'under 6 months', '1 year', '2 to 3 years', '4 to 8 years'],
            'NumberOfCars': ['1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8'],
            'VehiclePrice': ['less than 20,000', '20,000 to 29,000', '30,000 to 39,000', '40,000 to 59,000', '60,000 to 69,000', 'more than 69,000'],
            'Days:Policy-Accident': ['none', '1 to 7', '15 to 30', '8 to 15', 'more than 30'],
            'Days:Policy-Claim': ['15 to 30', '8 to 15', 'more than 30'],
            'PastNumberOfClaims': ['none', '1', '2 to 4', 'more than 4'],
            'AgeOfVehicle': ['new', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7'],
            'Make': ['Accura', 'BMW', 'Chevrolet', 'Dodge', 'Ferrari', 'Ford', 'Honda', 'Jaguar', 'Lexus', 'Mazda', 'Mecedes', 'Mercury', 'Nisson', 'Pontiac', 'Porche', 'Saab', 'Saturn', 'Toyota', 'VW']

            }

        for column, labels in column_labels.items():
            oe = OrdinalEncoder(categories=[labels], handle_unknown='error')
            X_train[column] = oe.fit_transform(X_train[[column]])
            X_test[column] = oe.transform(X_test[[column]])

        # one hot encode
        columns_one_hot = {
            'AccidentArea': ['Rural', 'Urban'],
            'Sex': ['Female', 'Male'],
            'MaritalStatus': ['Divorced', 'Married', 'Single', 'Widow'],
            'PoliceReportFiled': ['No', 'Yes'],
            'WitnessPresent': ['No', 'Yes'],
            'AgentType': ['External', 'Internal'],
            'BasePolicy': ['All Perils', 'Collision', 'Liability'],
            'Fault': ['Policy Holder', 'Third Party'],
            'PolicyType': ['Sedan - All Perils', 'Sedan - Collision', 'Sedan - Liability','Sport - All Perils', 'Sport - Collision', 'Sport - Liability', 'Utility - All Perils', 'Utility - Collision', 'Utility - Liability'],
            'VehicleCategory': ['Sedan', 'Sport', 'Utility'],

        }

        for column, labels in columns_one_hot.items():
            ohe = OneHotEncoder(sparse_output=False, categories=[labels], drop='first', handle_unknown='error')
            encoded_nominal = ohe.fit_transform(X_train[[column]])
            X_train = pd.concat([X_train, pd.DataFrame(encoded_nominal, columns=ohe.get_feature_names_out([column]), index=X_train.index)], axis=1)

            encoded_nominal = ohe.transform(X_test[[column]])
            X_test = pd.concat([X_test, pd.DataFrame(encoded_nominal, columns=ohe.get_feature_names_out([column]), index=X_test.index)], axis=1)

        X_test.drop(columns=columns_one_hot.keys(), axis=1, inplace=True)
        X_train.drop(columns=columns_one_hot.keys(), axis=1, inplace=True)

        rf_classifier = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            criterion=rf_criterion,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=141,
        )
        rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1 Score: {f1:.4f}")
        
        return {'loss': -f1, 'status': STATUS_OK, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1-score': f1}
    
    except Exception as e:
        print(f"Exception encountered: {e}")
        return {'status': STATUS_FAIL, 'exception': str(e)}
    
trials = MongoTrials('mongo://localhost:1234/hyperopt/jobs', exp_key='exp1')

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