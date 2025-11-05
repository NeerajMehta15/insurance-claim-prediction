import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from config import RAW_DATA_PATH, TARGET_COL


def load_data(path=RAW_DATA_PATH):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(path)
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def preprocess_data(data, target_column):
    """Preprocess the data by handling missing values, encoding categorical variables, and scaling features."""
    print("Starting preprocessing...")

    # Drop duplicates only
    data = data.drop_duplicates()
    print(f"After dropping duplicates: {data.shape}")

    # Fill missing values
    categorical_cols = data.select_dtypes(include=['object']).columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

    # Fill missing categorical with 'Unknown'
    data[categorical_cols] = data[categorical_cols].fillna('Unknown')
    # Fill missing numeric with median
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

    print(f"After filling missing values: {data.shape}")

    # Separate features and target
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Detected categorical columns: {categorical_cols}")

    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        encoded_data = encoder.fit_transform(X[categorical_cols])
        X_encoded = pd.DataFrame(
            encoded_data,
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        X = X.drop(columns=categorical_cols).reset_index(drop=True)
        X = pd.concat([X, X_encoded], axis=1)
        print(f"After encoding: {X.shape}")
    else:
        print("ℹNo categorical columns found. Skipping encoding.")

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print(f"After scaling: {X_scaled.shape}")

    # Encode target variable if it's categorical
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print("Target column encoded.")

    print("Preprocessing complete.")
    return X_scaled, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Data split: Train ({X_train.shape[0]}), Test ({X_test.shape[0]})")
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, prefix='preprocessed'):
    """Save the preprocessed data to CSV files."""
    X_train.to_csv(f'{prefix}_X_train.csv', index=False)
    X_test.to_csv(f'{prefix}_X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['target']).to_csv(f'{prefix}_y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['target']).to_csv(f'{prefix}_y_test.csv', index=False)
    print(f"Preprocessed files saved with prefix: {prefix}_")


if __name__ == "__main__":
    print("Running data preprocessing as standalone module...")
    data = load_data()
    if data is not None:
        try:
            X, y = preprocess_data(data, TARGET_COL)
            X_train, X_test, y_train, y_test = split_data(X, y)
            save_preprocessed_data(X_train, X_test, y_train, y_test)
            print("Preprocessing pipeline completed successfully.")
        except Exception as e:
            print(f"Preprocessing failed: {e}")
