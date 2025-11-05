import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data, target_column):
    """Preprocess the data by handling missing values, encoding categorical variables, and scaling features."""
    # Handle missing values
    data = data.dropna()

    # Drop duplicates or irrelevant columns
    data = data.drop_duplicates()

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
        X = X.drop(columns=categorical_cols).reset_index(drop=True)
        X = pd.concat([X, X_encoded], axis=1)

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Encode target variable if it's categorical
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    return X_scaled, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test, prefix='preprocessed'):
    """Save the preprocessed data to CSV files."""
    X_train.to_csv(f'{prefix}_X_train.csv', index=False)
    X_test.to_csv(f'{prefix}_X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['target']).to_csv(f'{prefix}_y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['target']).to_csv(f'{prefix}_y_test.csv', index=False)



if __name__ == "__main__":
    data = load_data()
    data = preprocess_data(data)
    save_processed_data(data)