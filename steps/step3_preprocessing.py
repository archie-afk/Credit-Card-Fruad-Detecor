from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess(df):

    # ── Scale Amount and Time
    scaler = StandardScaler()
    df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_Scaled']   = scaler.fit_transform(df[['Time']])
    df = df.drop(columns=['Amount', 'Time'])

    # ── Separate features and target
    X = df.drop(columns=['Class'])
    y =df ['Class']

    print("Features shape:", X.shape)
    print("Target shape:  ", y.shape)

    # ── Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining set:", X_train.shape)
    print("Test set:    ", X_test.shape)
    print("\nFraud cases in training:", y_train.sum())
    print("Fraud cases in test:    ", y_test.sum())

    return X_train, X_test, y_train, y_test