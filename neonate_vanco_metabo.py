import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import GroupKFold, GridSearchCV, KFold
from sklearn.linear_model import LassoCV, Lasso, Ridge, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, 
    r2_score, make_scorer
)
from sklearn.utils import shuffle
import xgboost as xgb

# -------------------------------
# Utility Functions
# -------------------------------

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.replace('.', np.nan)

    # Convert and fill specific columns
    cols_to_fill = ['ALBUMIN_LEVEL', 'UREA_NITROGEN']
    df[cols_to_fill] = df[cols_to_fill].astype(float)

    for col in cols_to_fill:
        df[col] = df.groupby('MRN', group_keys=False)[col].apply(
            lambda group: group.fillna(group.median())
        )
        df[col].fillna(df[col].median(), inplace=True)

    cols_to_convert = ['Birth_Weightkg', 'GAw', 'PNAw', 'PMAw', 'DoseWT',
                       'CREATININE_LEVEL', 'ALBUMIN_LEVEL', 'UREA_NITROGEN']
    df[cols_to_convert] = df[cols_to_convert].astype(float)

    return df

def prepare_features(df):
    y = df['Cl']
    groups = df['MRN']

    gkf = GroupKFold(n_splits=5)
    X_train_list, X_test_list = [], []
    y_train_list, y_test_list = [], []

    df, y, groups = shuffle(df, y, groups, random_state=42)

    for train_idx, test_idx in gkf.split(df, y, groups=groups):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Metabolomics features assumed to start from column index 13
        X_train_metabolo = train_df.iloc[:, 13:]
        X_test_metabolo = test_df.iloc[:, 13:]

        # Standardize metabolomics features
        sc_metabolo = StandardScaler()
        X_train_metabolo_scaled = sc_metabolo.fit_transform(X_train_metabolo)
        X_test_metabolo_scaled = sc_metabolo.transform(X_test_metabolo)

        X_train_metabolo_scaled = pd.DataFrame(X_train_metabolo_scaled, columns=X_train_metabolo.columns)
        X_test_metabolo_scaled = pd.DataFrame(X_test_metabolo_scaled, columns=X_test_metabolo.columns)

        # LASSO for feature selection
        lasso_cv = LassoCV(
            cv=GroupKFold(n_splits=5).split(X_train_metabolo_scaled, y_train, groups=train_df['Birth_Weightkg']),
            random_state=42,
            max_iter=5000
        )
        lasso_cv.fit(X_train_metabolo_scaled, y_train)
        selected_features = lasso_cv.coef_ != 0
        selected_feature_names = X_train_metabolo_scaled.columns[selected_features]

        X_train_lasso = X_train_metabolo_scaled[selected_feature_names]
        X_test_lasso = X_test_metabolo_scaled[selected_feature_names]

        # Clinical features
        clinical_features = [
            'Gender', 'Race_Ethnicity', 'Birth_Weightkg', 'GAw', 'PNAw',
            'PMAw', 'DoseWT', 'CREATININE_LEVEL', 'ALBUMIN_LEVEL', 'UREA_NITROGEN'
        ]
        X_train_other = train_df[clinical_features]
        X_test_other = test_df[clinical_features]

        combined = pd.concat([X_train_other, X_test_other], axis=0)
        combined_encoded = pd.get_dummies(combined, drop_first=True)

        X_train_other_encoded = combined_encoded.iloc[:X_train_other.shape[0], :]
        X_test_other_encoded = combined_encoded.iloc[X_train_other.shape[0]:, :]
        X_test_other_encoded = X_test_other_encoded.reindex(columns=X_train_other_encoded.columns, fill_value=0)

        # Combine all features
        X_train_full = pd.concat([X_train_lasso.reset_index(drop=True), X_train_other_encoded.reset_index(drop=True)], axis=1)
        X_test_full = pd.concat([X_test_lasso.reset_index(drop=True), X_test_other_encoded.reset_index(drop=True)], axis=1)

        # Final scaling
        sc_full = StandardScaler()
        X_train = sc_full.fit_transform(X_train_full)
        X_test = sc_full.transform(X_test_full)

        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    return X_train_list, X_test_list, y_train_list, y_test_list

# -------------------------------
# Model Definitions and Grid
# -------------------------------

models = {
    "Nearest_Neighbors": KNeighborsRegressor(),
    "Linear_SVM": SVR(kernel='linear'),
    "Gradient_Boosting": GradientBoostingRegressor(random_state=42),
    "Decision_Tree": DecisionTreeRegressor(random_state=42),
    "Extra_Trees": ExtraTreesRegressor(random_state=42),
    "Random_Forest": RandomForestRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "XGB": xgb.XGBRegressor(random_state=42),
    "Linear_Regression": LinearRegression(),
    "Lasso": Lasso(random_state=42),
    "Ridge": Ridge()
}

param_grids = {
    "Nearest_Neighbors": {"n_neighbors": [5, 10, 20, 50, 100], "weights": ['uniform', 'distance']},
    "Linear_SVM": {"C": [0.01, 0.1, 1, 10], "kernel": ['linear']},
    "Decision_Tree": {"max_depth": [3, 5, 10, 15]},
    "Gradient_Boosting": {"n_estimators": [20, 50, 100, 150, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2]},
    "Extra_Trees": {"n_estimators": [20, 50, 100, 150, 200], "min_samples_split": [2, 5, 10]},
    "Random_Forest": {"n_estimators": [20, 50, 100, 150, 200], "max_depth": [3, 5, 10, 15]},
    "AdaBoost": {"n_estimators": [20, 50, 100, 150, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2]},
    "XGB": {"n_estimators": [20, 50, 100, 150, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "subsample": [0.5, 0.8, 1.0]},
    "Lasso": {"alpha": [0.01, 0.05, 0.1, 0.5, 1]},
    "Ridge": {"alpha": [0.01, 0.1, 1, 10]},
    "Linear_Regression": {}
}

# -------------------------------
# Training and Evaluation
# -------------------------------

def evaluate_models(X_train_list, X_test_list, y_train_list, y_test_list):
    results = []

    for name, model in tqdm(models.items(), desc="Training models"):
        all_preds, all_true = [], []

        for i in range(5):
            X_train = X_train_list[i]
            X_test = X_test_list[i]
            y_train = y_train_list[i]
            y_test = y_test_list[i]

            # Log-transform target if needed
            y_train_log = np.log(y_train)

            pipeline = make_pipeline(
                SimpleImputer(strategy='mean'),
                GridSearchCV(
                    model, 
                    param_grids.get(name, {}),
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='neg_mean_squared_error'
                )
            )

            pipeline.fit(X_train, y_train_log)
            y_pred_log = pipeline.predict(X_test)
            y_pred = np.exp(y_pred_log)

            all_preds.extend(y_pred)
            all_true.extend(y_test)

        mse = mean_squared_error(all_true, all_preds)
        rmse = root_mean_squared_error(all_true, all_preds)
        mae = mean_absolute_error(all_true, all_preds)
        r2 = r2_score(all_true, all_preds)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "MSE": mse,
            "R2": r2
        })

        # Plotting
        plt.figure(figsize=(6, 6))
        plt.scatter(all_true, all_preds, alpha=0.6)
        plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{name} - True vs Predicted")
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results)

# -------------------------------
# Main Execution
# -------------------------------

if __name__ == "__main__":
    os.chdir('D:/HuiYu metabo/alignmentresults/ML data')

    data_path = 'ml_dataset_RUV-random.csv'
    df = load_and_preprocess_data(data_path)
    X_train_list, X_test_list, y_train_list, y_test_list = prepare_features(df)
    results_df = evaluate_models(X_train_list, X_test_list, y_train_list, y_test_list)

    # Save results
    results_df.to_csv("model_comparison_results.csv", index=False)
