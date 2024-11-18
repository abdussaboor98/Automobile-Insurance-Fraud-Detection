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
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

# Load data
df = pd.read_csv(
    "../../data/Angoss Knowledge Seeker - carclaims.txt/carclaims_original.csv"
)
# Drop row with missing data
df.drop(df[df["DayOfWeekClaimed"] == "0"].index, inplace=True)
# Drop ID column
df.drop(columns="PolicyNumber", inplace=True)

# Train-test split
carclaims_train, carclaims_test = train_test_split(df, test_size=0.2, random_state=42)

# Load SDV metadata
metadata = Metadata.load_from_json(filepath="carclaims_metadata.json")


space = [
    Integer(300, 1000, name="ctgan_epochs"),
    Integer(5, 100, name="ctgan_pac"),
    Integer(25, 50, name="ctgan_batch_multiple"),
    Real(1e-8, 1e-4, name="ctgan_discriminator_decay"),
    Real(2e-6, 2e-2, name="ctgan_discriminator_lr"),
    Real(1e-8, 1e-4, name="ctgan_generator_decay"),
    Real(2e-6, 2e-2, name="ctgan_generator_lr"),
    Integer(2, 5, name="ctgan_discriminator_layers"),
    Categorical([64, 128, 256, 512, 1024], name="ctgan_discriminator_width"),
    Integer(2, 5, name="ctgan_generator_layers"),
    Categorical([64, 128, 256, 512, 1024], name="ctgan_generator_width"),
    Categorical([64, 128, 256, 512, 1024], name="ctgan_embedding_dim"),
    Integer(64, 1024, name="rf_n_estimators"),
    Categorical(["gini", "entropy", "log_loss"], name="rf_criterion"),
    Integer(5, 50, name="rf_max_depth"),
    Integer(2, 50, name="rf_min_samples_split"),
    Integer(1, 50, name="rf_min_samples_leaf"),
]

@use_named_args(space)
def train_and_predict(
    ctgan_epochs, # 300
    ctgan_pac, # 10
    ctgan_batch_multiple, # 50
    ctgan_discriminator_decay, # 1e-6
    ctgan_discriminator_lr, #2e-4
    ctgan_generator_decay,
    ctgan_generator_lr,
    ctgan_discriminator_layers, # 2
    ctgan_discriminator_width, # 256
    ctgan_generator_layers, # 2
    ctgan_generator_width, # 256
    ctgan_embedding_dim, # 128
    rf_n_estimators,
    rf_criterion,
    rf_max_depth,
    rf_min_samples_split,
    rf_min_samples_leaf,
    ctgan_discriminator_steps = 1, # 1
):

    ctgan_batch_size = ctgan_pac * ctgan_batch_multiple * 2
    ctgan_discriminator_dim = [ctgan_discriminator_width] * ctgan_discriminator_layers
    ctgan_generator_dim = [ctgan_generator_width] * ctgan_generator_layers
    print(f'Testing for: ctgan_epochs:{ctgan_epochs}, ctgan_pac:{ctgan_pac}, ctgan_batch_size:{ctgan_batch_size},  ctgan_discriminator_decay:{ctgan_discriminator_decay}, ctgan_discriminator_lr:{ctgan_discriminator_lr}, ctgan_generator_decay:{ctgan_generator_decay}, ctgan_generator_lr:{ctgan_generator_lr}, ctgan_discriminator_dim:{ctgan_discriminator_dim}, ctgan_generator_dim:{ctgan_generator_dim}, ctgan_embedding_dim:{ctgan_embedding_dim}, rf_n_estimators:{rf_n_estimators}, rf_criterion:{rf_criterion}, rf_max_depth:{rf_max_depth}, rf_min_samples_split:{rf_min_samples_split}, rf_min_samples_leaf:{rf_min_samples_leaf}') 

    # Create synthesizer
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=ctgan_epochs,  # 300
        batch_size=ctgan_batch_size,  # Must be multiple of pac and / by 2 - 500
        pac=ctgan_pac,  # 10
        discriminator_decay=ctgan_discriminator_decay,  # 1e-6
        generator_decay=ctgan_generator_decay,  # 1e-6
        discriminator_lr=ctgan_discriminator_lr,  # 2e-4
        generator_lr=ctgan_generator_lr,  # 2e-4
        discriminator_steps=ctgan_discriminator_steps,  # 1 (As per original CTGAN)
        discriminator_dim=ctgan_discriminator_dim,  # (256, 256)
        embedding_dim=ctgan_embedding_dim,  # 128
        generator_dim=ctgan_generator_dim,  # (256, 256)
    )
    synthesizer.fit(carclaims_train)

    # Conditions for balancing the data
    fraud_samples = Condition(
        num_rows=50_000,
        column_values={'FraudFound': 'Yes'}
    )

    non_fraud_samples = Condition(
        num_rows=50_000,
        column_values={'FraudFound': 'No'}
    )

    # Create balanced synthetic data
    synthetic_data = synthesizer.sample_from_conditions(
        conditions=[fraud_samples, non_fraud_samples],
        batch_size=1_000
    )

    # X y split
    X_train = synthetic_data.drop('FraudFound', axis=1).reset_index(drop=True)
    y_train = synthetic_data['FraudFound'].reset_index(drop=True)

    X_test = carclaims_test.drop('FraudFound', axis=1).reset_index(drop=True)
    y_test = carclaims_test['FraudFound'].reset_index(drop=True)

    # Encode target variable
    y_train.loc[y_train[:] == 'No'] = 0
    y_train.loc[y_train[:] == 'Yes'] = 1

    y_test.loc[y_test[:] == 'No'] = 0
    y_test.loc[y_test[:] == 'Yes'] = 1

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
        'VehicleCategory': ['Sedan', 'Sport', 'Utility']
    }

    for column, labels in columns_one_hot.items():
        ohe = OneHotEncoder(sparse_output=False, categories=[labels], drop='first', handle_unknown='error')
        encoded_nominal = ohe.fit_transform(X_train[[column]])
        X_train = pd.concat([X_train, pd.DataFrame(encoded_nominal, columns=ohe.get_feature_names_out([column]))], axis=1)

        encoded_nominal = ohe.transform(X_test[[column]])
        X_test = pd.concat([X_test, pd.DataFrame(encoded_nominal, columns=ohe.get_feature_names_out([column]))], axis=1)

    X_test.drop(columns=columns_one_hot.keys(), axis=1, inplace=True)
    X_train.drop(columns=columns_one_hot.keys(), axis=1, inplace=True)

    rf_classifier = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        criterion=rf_criterion,
        max_depth=rf_max_depth,
        min_samples_split=rf_min_samples_split,
        min_samples_leaf=rf_min_samples_leaf,
        random_state=42,
    )
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1 * 100:.2f}%")
    return -f1

    # print("\n=== XGBoost on SMOTE + Autoencoder Processed Data ===")
    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    
res_gp = gp_minimize(train_and_predict, space, n_calls=50, random_state=42)

print(f'Best f1: {res_gp.fun}')
print(f'Best params: {res_gp.x}')