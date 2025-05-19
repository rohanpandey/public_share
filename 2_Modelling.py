# 2_Modelling.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, auc
from sklearn.utils import resample

base_dir = '/data_218/home1/rohan/0_homelessness'
raw_data_dir = f"{base_dir}/data/raw/"
save_data_dir = f"{base_dir}/data/processed/"
results_path = f"{base_dir}/data/results"

agg_level_list = ['yearly', 'quarterly', 'monthly']
agg_type_list = ['count']
target_list = ['HL_17_Q1','HL_17_Q1_Q2','HL_17_Q1_Q2_Q3', 'HL_17_Q1_Q2_Q3_Q4']
data_setting = 'balanced'


def get_column_groups(X, group_prefixes=None):
    group_prefixes = group_prefixes or ['demo', 'sdoh', 'diag', 'substances']
    group_cols = {prefix: [] for prefix in group_prefixes}
    for col in X.columns:
        for prefix in group_prefixes:
            if col.startswith(prefix + '_'):
                group_cols[prefix].append(col)
                break
    return group_cols


for agg_type in agg_type_list:
    for agg_level in agg_level_list:
        for target_year in target_list:
            file_name = f"df_{agg_level}_{agg_type}"
            df = pd.read_csv(save_data_dir+f"/{file_name}.csv")
            targets = target_list.copy()
            targets.remove(target_year)
            df = df.drop(columns=['patienticn'] + targets)
            df = df.rename(columns={target_year: 'target'})
            df['demo_maritalstatus'] = df['demo_maritalstatus'].astype(str)
            df['demo_gisurh'] = df['demo_gisurh'].astype(str)

            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numerical_cols.remove("target")

            label_encoder = LabelEncoder()
            df["target"] = label_encoder.fit_transform(df["target"])

            num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
            cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer(transformers=[('num', num_transformer, numerical_cols), ('cat', cat_transformer, categorical_cols)])

            X = df.drop(columns=['target'])
            y = df['target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

            if data_setting == 'balanced':
                train_df = pd.concat([X_train, y_train], axis=1)
                minority = train_df[train_df['target'] == 1]
                majority = train_df[train_df['target'] == 0]
                majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
                train_balanced = pd.concat([majority_downsampled, minority])
                train_balanced = train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
                X_train = train_balanced.drop(columns=['target'])
                y_train = train_balanced['target']

            group_cols = get_column_groups(X)
            ablation_sets = {'all': X.columns.tolist()}
            ablation_sets.update({f"only_{k}": v for k, v in group_cols.items()})
            ablation_sets.update({f"drop_{k}": [c for c in X.columns if c not in v] for k, v in group_cols.items()})

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100),
                "Gradient Boosting": GradientBoostingClassifier(),
            }

            param_grids = {
                "Logistic Regression": {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear', 'lbfgs']},
                "Decision Tree": {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]},
                "Random Forest": {'n_estimators': [50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2']},
                "Gradient Boosting": {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
            }

            for ablation_key, selected_cols in ablation_sets.items():
                print(f"\n🧪 Running ablation: {ablation_key} — using {len(selected_cols)} columns")
                X_train_sel = X_train[selected_cols].copy()
                X_test_sel = X_test[selected_cols].copy()

                X_train_proc = preprocessor.fit_transform(X_train_sel)
                X_test_proc = preprocessor.transform(X_test_sel)

                results, detailed_results, confusion_matrices, risk_group_results = [], [], [], []
                risk_group_sizes = [0.05, 0.10, 0.20, 0.60]
                os.makedirs(results_path+f"/{file_name}", exist_ok=True)
                os.makedirs(results_path+f"/{file_name}/confusion_matrices", exist_ok=True)

                for name, model in models.items():
                    print(f"Training {name} with hyperparameter tuning...")
                    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=3, scoring='f1', n_jobs=-1)
                    grid_search.fit(X_train_proc, y_train)
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test_proc)
                    y_proba = best_model.predict_proba(X_test_proc)[:, 1] if hasattr(best_model, "predict_proba") else None

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
                    pr_auc = auc(*precision_recall_curve(y_test, y_proba)[::-1]) if y_proba is not None else None

                    cm = confusion_matrix(y_test, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

                    results.append([f"{name}_{ablation_key}", accuracy, precision, recall, f1, roc_auc, pr_auc, sensitivity, specificity, ppv])

                    for p in risk_group_sizes:
                        top_n = int(p * len(y_test))
                        sorted_indices = np.argsort(y_proba)[::-1]
                        selected = sorted_indices[:top_n]
                        y_selected = y_test.iloc[selected]
                        tp = sum(y_selected == 1)
                        fp = sum(y_selected == 0)
                        fn = sum(y_test) - tp
                        tn = len(y_test) - top_n - fn

                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                        risk_group_results.append([f"{name}_{ablation_key}", p, sensitivity, specificity, ppv])

                    report = classification_report(y_test, y_pred, output_dict=True)
                    for class_label, metrics in report.items():
                        if isinstance(metrics, dict):
                            detailed_results.append({"Model": f"{name}_{ablation_key}", "Class": class_label, **metrics})

                df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC", "Sensitivity", "Specificity", "PPV"])
                df_results.to_csv(f"{results_path}/{file_name}/model_metrics_{file_name}_{target_year}_{ablation_key}.csv", index=False)
                df_group = pd.DataFrame(risk_group_results, columns=["Model", "Risk Group Sizes", "Sensitivity", "Specificity", "PPV"])
                df_group.to_csv(f"{results_path}/{file_name}/model_group_metrics_{file_name}_{target_year}_{ablation_key}.csv", index=False)
                df_detail = pd.DataFrame(detailed_results)
                df_detail.to_csv(f"{results_path}/{file_name}/detailed_model_metrics_{file_name}_{target_year}_{ablation_key}.csv", index=False)

                print(f"✔️ Results saved for ablation: {ablation_key}")
