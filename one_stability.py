# module1_stability.py

import numpy as np
from sklearn.ensemble import IsolationForest # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore


def train_stability_model(df):
    """
    Trains Isolation Forest on athlete baseline gait.
    Returns trained model + scaler.
    """

    df['symmetry_ratio'] = df['prosthetic_load'] / df['intact_load']
    df['knee_angle_variance'] = df['knee_angle_variance'].rolling(5).var()
    df['socket_pressure_avg'] = df['socket_pressure'].rolling(5).mean()
    df['angular_velocity_avg'] = df['angular_velocity'].rolling(5).mean()
    df['ground_contact_avg'] = df['ground_contact_time'].rolling(5).mean()
    df['stride_length_avg'] = df['stride_length'].rolling(5).mean()

    df = df.dropna().reset_index(drop=True)

    features = [
        'symmetry_ratio',
        'socket_pressure_avg',
        'angular_velocity_avg',
        'ground_contact_avg',
        'stride_length_avg',
        'knee_angle_variance'
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42
    )

    model.fit(X_scaled)

    return model, scaler


def predict_stability(model, scaler, df):

    features = [
        'symmetry_ratio',
        'socket_pressure_avg',
        'angular_velocity_avg',
        'ground_contact_avg',
        'stride_length_avg',
        'knee_angle_variance'
    ]

    X = df[features]
    X_scaled = scaler.transform(X)

    anomaly_raw = -model.decision_function(X_scaled)

    anomaly_norm = 1 / (1 + np.exp(-5 * (anomaly_raw - np.mean(anomaly_raw))))
    stability_scores = 100 * (1 - anomaly_norm)

    return stability_scores
