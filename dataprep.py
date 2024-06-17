import pandas as pd
from sklearn.model_selection import train_test_split


def mlp4_load_and_preprocess_data(file_path, test_size=0.2, safe_data_fraction=0.1, undersample_glycerol = False):
    data = pd.read_csv(file_path)
    data.fillna(0, inplace=True)
    data = data.iloc[:, :-13]  # drops the last 13 columns
    mask_all_zeros = (data.iloc[:, 300:304].sum(axis=1) == 0)  # mask rows with all zeros in columns 300 to 304
    data = data[~mask_all_zeros]

    # Undersampling glycerol rows (only if called to do so)
    if undersample_glycerol:
        glycerol_rows = data[data['1Mglycerol'] == 1]
        other_rows = data[data['1Mglycerol'] == 0]
        sampled_glycerol_rows = glycerol_rows.sample(frac=0.5, random_state=42)
        data = pd.concat([sampled_glycerol_rows, other_rows])
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data.iloc[:, :300]  # features
    y = data.iloc[:, 300:304]  # predictors

    # Training and safe data split
    X_train, X_safe, y_train, y_safe = train_test_split(X, y, test_size=safe_data_fraction, random_state=42)

    # Train test split on previous training
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

    return X_train, y_train, X_test, y_test, X_safe, y_safe


def mlp_load_and_preprocess_data(file_path, test_size=0.2, safe_data_fraction=0.1, undersample_glycerol = False):
    data = pd.read_csv(file_path)
    data.fillna(0, inplace=True)
    data = data.iloc[:, :-13]  # drops the last 13 columns
    mask_all_zeros = (data.iloc[:, 300:310].sum(axis=1) == 0)  # mask rows with all zeros in columns 300 to 310
    data = data[~mask_all_zeros]

    # Undersampling glycerol rows (only if called to do so)
    if undersample_glycerol:
        glycerol_rows = data[data['1Mglycerol'] == 1]
        other_rows = data[data['1Mglycerol'] == 0]
        sampled_glycerol_rows = glycerol_rows.sample(frac=0.5, random_state=42)
        data = pd.concat([sampled_glycerol_rows, other_rows])
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data.iloc[:, :300]  # features
    y = data.iloc[:, 300:310] # predictors

    # Training and safe data split
    X_train, X_safe, y_train, y_safe = train_test_split(X, y, test_size=safe_data_fraction, random_state=42)

    # Train test split on previous training
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

    return X_train, y_train, X_test, y_test, X_safe, y_safe


def rf_load_and_preprocess_data(file_path, undersample_glycerol=False, safe_data_fraction=0.1):
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)

    mask_last_13 = (df.iloc[:, -13:] == 1).any(axis=1) # mask rows with 1s in last 13 columns
    df = df[~mask_last_13]

    mask_all_zeros = (df.iloc[:, 300:311].sum(axis=1) == 0) # mask rows with all zeros in columns 300 to 310
    df = df[~mask_all_zeros]

    # Find the column names with maximum values for the remaining rows
    target = df.iloc[:, 300:311].idxmax(axis=1)
    df['target'] = target

    # Undersampling glycerol rows (only if called to do so)
    if undersample_glycerol:
        glycerol_rows = df[df['target'] == '1Mglycerol'] # filters the rows with '1Mglycerol'
        other_rows = df[df['target'] != '1Mglycerol']
        # Takes a fraction of the glycerol rows
        sampled_glycerol_rows = glycerol_rows.sample(frac=0.5, # change this to select desired undersampling rate
                                                     random_state=42)
        df = pd.concat([sampled_glycerol_rows, other_rows])

    # Setting the features and target
    X = df.iloc[:, :300]  # features
    y = df['target']  # target

    # Set aside safe data
    X_train, X_safe, y_train, y_safe = train_test_split(X, y, test_size=safe_data_fraction, random_state=42)

    return X_train, y_train, X_safe, y_safe


def autoencoder_dataprep(file_path):
    data = pd.read_csv(file_path)
    data.fillna(0, inplace=True)

    data = data.iloc[:,:-13] # drops the last 13 columns

    mask_all_zeros = (data.iloc[:, 300:310].sum(axis=1) == 0) # mask rows with all zeros in columns 300 to 310
    data = data[~mask_all_zeros]

    noisy_data = data.copy()
    noisy_data.iloc[:, 300:310] = 0.1 # makes the noise all values equal to 0.1

    return data, noisy_data
