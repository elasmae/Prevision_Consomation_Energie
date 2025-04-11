def main():
    import pandas as pd
    import numpy as np
    import joblib
    import os
    from datetime import timedelta
    import matplotlib.pyplot as plt

    data_path = "data/03_training_data"
    model_path = "data/05_models"
    output_path = "data/04_visualisation"
    os.makedirs(output_path, exist_ok=True)

    print("Chargement des modèles...")
    rfr = joblib.load(f"{model_path}/random_forest_model.pkl")
    xgb = joblib.load(f"{model_path}/xgboost_model.pkl")
    lgbm = joblib.load(f"{model_path}/lightgbm_model.pkl")

    df = pd.read_csv("data/02_processed/processed_weather_and_consumption_data.csv", index_col=0, parse_dates=True)

    
    def create_features(df, column_names, lags, window_sizes):
        created_features = []
        basic_features = ["dayofweek", "quarter", "month", "year", "dayofyear"]
        for feature in basic_features:
            df[feature] = getattr(df.index, feature)
            created_features.append(feature)

        for column in column_names:
            for lag in lags:
                df[f"{column}_lag_{lag}"] = df[column].shift(lag)
                created_features.append(f"{column}_lag_{lag}")
            for window in window_sizes:
                df[f"{column}_rolling_mean_{window}"] = df[column].shift(1).rolling(window=window).mean()
                created_features.append(f"{column}_rolling_mean_{window}")
        return df, created_features

    column_names = [
        "total_consumption", "Global_intensity", "Sub_metering_3", "Sub_metering_1",
        "temp", "day_length", "tempmax", "feelslike", "feelslikemax", "feelslikemin", "tempmin"
    ]
    lags = [1, 2, 3, 4, 5, 6, 7, 30, 90, 365]
    window_sizes = [2, 3, 4, 5, 6, 7, 30, 90, 365]
    df, created_features = create_features(df, column_names, lags, window_sizes)

    external_features = [
        "tempmax", "tempmin", "temp", "feelslikemax", "feelslikemin", "feelslike", "dew", "humidity",
        "precip", "precipprob", "precipcover", "snow", "snowdepth", "windgust", "windspeed", "winddir",
        "sealevelpressure", "cloudcover", "visibility", "moonphase", "conditions_clear",
        "conditions_overcast", "conditions_partiallycloudy", "conditions_rain",
        "conditions_rainovercast", "conditions_rainpartiallycloudy", "conditions_snowovercast",
        "conditions_snowpartiallycloudy", "conditions_snowrain", "conditions_snowrainovercast",
        "conditions_snowrainpartiallycloudy", "day_length", "is_holiday"
    ]
    FEATURES = created_features + external_features

    # === Prévision directe t+30
    latest = df.iloc[[-1]].copy()
    X_latest = latest[FEATURES]

    pred_direct = {
        "Date": [df.index[-1] + timedelta(days=30)],
        "Random Forest": [rfr.predict(X_latest)[0]],
        "XGBoost": [xgb.predict(X_latest)[0]],
        "LightGBM": [lgbm.predict(X_latest)[0]]
    }
    pred_direct_df = pd.DataFrame(pred_direct)
    pred_direct_df.to_csv(f"{output_path}/prevision_directe_t_plus_30.csv", index=False)
    print("Prévision directe t+30 enregistrée.")

    # === Prévision Rolling sur 30 jours
    rolling_df = df.copy()
    date_range = pd.date_range(df.index[-1] + timedelta(days=1), periods=30)
    results_rolling = []

    for d in date_range:
        last_row = rolling_df.iloc[[-1]].copy()
        last_row.name = d
        rolling_df = pd.concat([rolling_df, last_row])
        rolling_df, _ = create_features(rolling_df, column_names, lags, window_sizes)
        df_input = rolling_df.loc[rolling_df.index == d].dropna()

        if df_input.empty:
            continue

        row = {
            "Date": d,
            "Random Forest": rfr.predict(df_input[FEATURES])[0],
            "XGBoost": xgb.predict(df_input[FEATURES])[0],
            "LightGBM": lgbm.predict(df_input[FEATURES])[0]
        }
        rolling_df.loc[d, "total_consumption"] = row["Random Forest"]  
        results_rolling.append(row)

    rolling_df_results = pd.DataFrame(results_rolling)
    rolling_df_results.to_csv(f"{output_path}/prevision_rolling_30_jours.csv", index=False)
    print("Prévision rolling 30 jours enregistrée.")

    # === Graphique matplotlib ===
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in ["Random Forest", "XGBoost", "LightGBM"]:
        ax.plot(rolling_df_results["Date"], rolling_df_results[model], label=f"{model}")
    ax.set_title("Prévisions Rolling sur 30 jours")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consommation prévue")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_path}/plot_prevision_rolling_30_jours.png", dpi=300)
    print("Graphique des prévisions rolling enregistré.")


# === Point d’entrée ===
if __name__ == "__main__":
    main()

