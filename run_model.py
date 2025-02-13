# Imports
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import json
import re
import cloudpickle
from datetime import datetime
import time

import pandas as pd
import numpy as np
import random

from scipy.sparse import csr_matrix

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef, 
    average_precision_score, 
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.metrics.pairwise import (
    manhattan_distances,
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
     StandardScaler,
)

import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
import shap

# Constant
folds_size = 2
seed = 3
up_limit = 1200
sub_pop = 3
flag_words = True
how_df_words = 'inner'
k = 8
label = 'MST_ANAF_A'
n_features = 900
min_companies = 20
n_cpus = -1
version_model = 'v3'
version_model_output = version_model + '_6'
kpis_model_version =  version_model_output + f'_{n_features}_{sub_pop}_{up_limit}_{min_companies}_{folds_size}_{k}_{flag_words * 1}_{how_df_words}'
prediction_table_name = 'branch_prediction_2022_' + version_model_output
importance_table_name = 'branch_features_important_2022_' + version_model

param = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.3,
    "colsample_bytree": 0.3,
    "enable_categorical": True,
    'random_state': seed,
    'n_jobs': n_cpus,
}

np.random.seed(seed)
random.seed(seed)

def print_time(str_time, gap_time):
    print(f'{str_time}, time {gap_time:.2f}')


def split_it(llm_output):
    if llm_output:
        return re.findall(r"[\w']+", llm_output)
    return []


def masked_cosine_similarity_df(records_df, centroids_df):
    """
    Compute masked cosine similarities between two DataFrames efficiently using sparse matrices.

    Parameters:
    - records_df: pd.DataFrame
        DataFrame where each row is a record.
    - centroids_df: pd.DataFrame
        DataFrame where each row is a centroid.

    Returns:
    - similarities: pd.DataFrame
        A DataFrame where each entry (i, j) is the similarity from record i to centroid j.
    """
    # Ensure the two dataframes have the same columns
    assert len(records_df.columns) == len(centroids_df.columns), "Columns of records_df and centroids_df must match"

    # Replace NaN with 0 (sparse representation) and mask NaN separately
    records_array = records_df.fillna(0).to_numpy()
    centroids_array = centroids_df.fillna(0).to_numpy()

    records_mask = ~records_df.isna().to_numpy()
    centroids_mask = ~centroids_df.isna().to_numpy()

    # Apply masks
    # records_sparse = csr_matrix(records_array * records_mask)
    # centroids_sparse = csr_matrix(centroids_array * centroids_mask)
    records_sparse = records_array * records_mask
    centroids_sparse = centroids_array * centroids_mask

    # Compute cosine similarities using sparse matrix operations
    similarities_sparse = manhattan_distances(records_sparse, centroids_sparse)

    return similarities_sparse


def sampling_lmblearn(X, y, up_limit, random_state):
    """
    Balance class distribution by undersampling using imbalanced-learn's RandomUnderSampler.
    Args:
        X (DataFrame): Feature data.
        y (Series): Target labels.
        up_limit (int): Maximum samples per class after undersampling.
        clf__random_state (int): Random state for reproducibility.
    Returns:
        DataFrame, Series: Resampled feature set and target labels.
    """
    categories_count = y.value_counts()
    target_samples = {k: up_limit for k, _ in categories_count[categories_count > up_limit].to_dict().items() if k in y}
    rus = RandomUnderSampler(sampling_strategy=target_samples, random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res  # Return resampled features and labels


def metrics_kpis(real_values, pred_values):
    time       = datetime.now()
    Accuracy   = round(accuracy_score(real_values,pred_values), 4)
    Precision  = round(precision_score(real_values,pred_values, average='macro'), 4)
    Recall     = round(recall_score(real_values,pred_values, average='macro'), 4)
    F1         = round(f1_score(real_values,pred_values, average='macro'), 4)
    MCC        = round(matthews_corrcoef(real_values,pred_values) ,4)
    Prec_Avg   = round(average_precision_score(real_values,pred_values), 4)
    Accu_Avg   = round(balanced_accuracy_score(real_values,pred_values), 4)
    return [time, Accuracy, Precision, Recall, F1, MCC, Prec_Avg, Accu_Avg]


def create_connection():
    servername = 'bisqldwhd1'
    dbname = 'SRC_MCH'
    engine = create_engine(f'mssql+pyodbc://@{servername}/{dbname}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server')
    return engine


def load_data(engine, version_model):
    start = time.time()
    df = pd.read_sql_table(
        "branch_features_panel_2022_" + version_model,
        schema="dbo",
        con=engine,
    ).sample(5000)
    # ).sample(frac=1)
    end = time.time()
    print_time(' load data feature panel', end - start)
    return df


def load_words(engine, version_model, df):
    start = time.time()
    df_words = pd.read_sql_table(
        "branch_llm_branch_2022_" + version_model,
        schema="dbo",
        con=engine,
    )

    df_words['words'] = df_words.branch_llm.apply(split_it)
    df_words['words'] = df_words['words'].apply(lambda d: d if isinstance(d, list) else [])

    df_words = pd.DataFrame(df_words.words.values.tolist(), df_words.tik).add_prefix('word_')
    if len(list(df_words)) > 4:
        df_words.drop(columns=[f'word_{i}' for i in range(4, len(list(df_words)))], inplace=True)

    words_columns = list(df_words)
    df_words = df_words.replace('\n', '')
    df_words = df_words.astype('category')
    df_words.reset_index(inplace=True)

    df = df.merge(df_words, on='tik', how=how_df_words)
    end = time.time()
    print_time(f' words n records {len(df_words)}', end - start)
    return df, words_columns


def fillter_data(sub_pop, engine, version_model, df):
    start = time.time()
    if sub_pop == 1:
        final_df = pd.read_sql_table(
            "branch_prediction_2022_" + version_model,
            schema="dbo",
            con=engine,
        )

        final_df['hit'] = (final_df[label] == final_df.best_anaf) * 1

        subpopulation = (final_df[[label, 'hit']].groupby([label]).sum() / 
                                final_df[[label, 'hit']].groupby([label]).count()).reset_index()

        sub_pupolation = subpopulation.sort_values(by=['hit']).loc[subpopulation.hit > 0.1, label].values.tolist()
        df = df[(df.same_anaf == 1) & (df.MST_ANAF_A.isin(sub_pupolation))].reset_index(drop=True)

        del final_df
        print(f'sub pop {len(sub_pupolation)}')
    elif sub_pop == 2:
        df = df[(df.has_clients_features == 0) & (df.has_suppliers_features == 0)].reset_index(drop=True)
    elif sub_pop == 3:
        df = df[(df.same_anaf == 1)].reset_index(drop=True)
    else:
        df = df[(df.same_anaf == 1) & (df.MST_ANAF_A != 7020.0)].reset_index(drop=True)

    categories = df[label].value_counts()
    df = df[df[label].isin(categories[categories > min_companies].index)].reset_index(drop=True)
    end = time.time()
    print_time(f' n records {len(df)}, n classes {df[label].nunique()}', end - start)
    return df


def encoding_data(df, words_columns):
    start = time.time()
    le = LabelEncoder()
    df[label] = le.fit_transform(df[label])

    df = df.drop(['MST_D_HACHNASA_ESEK_R',
                    'MST_D_HACHNASA_SACAR_R', 'MST_D_HACHNASA_SACAR_BZ', 
                    'MST_MAAM_R', 'MST_KNAS_GERAON', 'MST_HACHNASOT_ESEK_R', 'MST_MAS_MEGIA_R',
                    'MST_NIKUI_MAKOR_SACHAR', 'MST_ITRAT_ZIK_R', 'doh_anaf', 'MST_HACHNASA_SACAR_R', 
                    'same_anaf'], axis=1)
                    # 'DOH_MIS_OSEK',

    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)

    # Categorical list
    categorical_list_tmp = ['KOD_DOAAR_CHOZER',
                        'YSHUV',
                        'YSHUV_ESEK',
                        'YSHUV_PRATI',
                        'has_suppliers_features',
                        'anaf_negdi_1_by_schum',
                        'anaf_negdi_2_by_schum',
                        'anaf_negdi_3_by_schum',
                        'anaf_negdi_4_by_schum',
                        'anaf_negdi_5_by_schum',
                        'anaf_negdi_1_by_tr',
                        'anaf_negdi_2_by_tr',
                        'anaf_negdi_3_by_tr',
                        'anaf_negdi_4_by_tr',
                        'anaf_negdi_5_by_tr',
                        'anaf_negdi_1_by_c_tik',
                        'anaf_negdi_2_by_c_tik',
                        'anaf_negdi_3_by_c_tik',
                        'anaf_negdi_4_by_c_tik',
                        'anaf_negdi_5_by_c_tik',
                        'has_clients_features',
                        'anaf_negdi_1_by_schum_clients',
                        'anaf_negdi_2_by_schum_clients',
                        'anaf_negdi_3_by_schum_clients',
                        'anaf_negdi_4_by_schum_clients',
                        'anaf_negdi_5_by_schum_clients',
                        'anaf_negdi_1_by_tr_clients',
                        'anaf_negdi_2_by_tr_clients',
                        'anaf_negdi_3_by_tr_clients',
                        'anaf_negdi_4_by_tr_clients',
                        'anaf_negdi_5_by_tr_clients',
                        'anaf_negdi_1_by_c_tik_clients',
                        'anaf_negdi_2_by_c_tik_clients',
                        'anaf_negdi_3_by_c_tik_clients',
                        'anaf_negdi_4_by_c_tik_clients',
                        'anaf_negdi_5_by_c_tik_clients',
                        'mst_anaf_lvl4_desc',
                        'mst_anaf_lvl3',
                        'mst_anaf_lvl3_desc',
                        'mst_anaf_lvl2',
                        'mst_anaf_lvl2_desc',
                        'mst_anaf_lvl1',
                        'mst_anaf_lvl1_desc',
                        'SHEM_MALE',
                        'SHEM_LOAZI',
                        'SHEM_MALE_KODEM',
                        'prt_shem_tik'
                        ] 
                        
    if flag_words:
        categorical_list_tmp += words_columns

    categorical_list = []
    for col in categorical_list_tmp:
        if col in df.columns:
            categorical_list.append(col)        
               
    remove_cols_tmp = ['tik',
                    label,
                    'mst_anaf_lvl4_desc',
                    'mst_anaf_lvl3',
                    'mst_anaf_lvl3_desc',
                    'mst_anaf_lvl2',
                    'mst_anaf_lvl2_desc',
                    'mst_anaf_lvl1',
                    'mst_anaf_lvl1_desc',
                    'SHEM_MALE',
                    'SHEM_LOAZI',
                    'SHEM_MALE_KODEM',
                    'prt_shem_tik',
                    'mam_melel_anaf',
                    'matara',]

    remove_cols = []
    for col in remove_cols_tmp:
        if col in df.columns:
            remove_cols.append(col)     


    remove_continuos_tmp = ['hevrot_y_rishum',
                        'PRT_SM_ST',
                        ]

    remove_continuos = []
    for col in remove_continuos_tmp:
        if col in df.columns:
            remove_continuos.append(col)   

    df[categorical_list] = df[categorical_list].astype('category')

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_list = df.select_dtypes(include=numerics).columns.tolist()

    for col in remove_cols:
        if col in numeric_list:
            numeric_list.remove(col)

    for col in remove_continuos:
        if col in numeric_list:
            numeric_list.remove(col)

    end = time.time()
    print_time(f' n numeric features {len(numeric_list)}', end - start)
    return df, numeric_list, remove_cols, le


def transforamtion(engine, df, numeric_list, words_columns, remove_cols):
    start = time.time()
    # Transformation
    series_list = []
    cols_iteration_list = []
    for i, a in enumerate(numeric_list[:-1]):
        for b in numeric_list[i + 1:]:
            if a != b:
                cols_iteration_list.append(a + '|' + b)
                series_list.append(df[a] / df[b])

    iteation_df = pd.concat(series_list, axis=1)
    iteation_df.columns = cols_iteration_list

    df = pd.concat([df, iteation_df], axis=1)
    df = df.rename(str, axis="columns")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_list += cols_iteration_list

    if n_features:
        xgb_fea_imp = pd.read_sql_table(importance_table_name,
                        engine, schema="dbo")    
        
        selected_features = xgb_fea_imp.feature.head(n_features).values.tolist()
        if flag_words:
            selected_features += words_columns

        columns_list_tmp = list(df)
        columns_list = []
        for col in columns_list_tmp:
            if col in selected_features:
                columns_list.append(col)
        df = df[remove_cols + columns_list]

        numeric_list_tmp = numeric_list.copy()
        numeric_list = []
        for col in columns_list:
            if col in numeric_list_tmp:
                numeric_list.append(col)

    end = time.time()
    print_time(f' n columns numeric {len(numeric_list)}', end - start)
    return df, numeric_list


def create_centers(df, numeric_list):
    start = time.time()
    # Add centers
    scaler = StandardScaler()
    df[numeric_list] = scaler.fit_transform(df[numeric_list])

    df_group_mean = df[numeric_list + [label]].groupby(label).mean()
    df_group_std = df[numeric_list + [label]].groupby(label).std()

    df_group_mean.columns = [col + '_mean' for col in numeric_list]
    df_group_std.columns = [col + '_std' for col in numeric_list]

    distance = masked_cosine_similarity_df(df[numeric_list], df_group_mean)
    ind = np.argpartition(distance, k, axis=1)[:, :k]

    series_list = []
    cols_iteration_list = []
    for a in range(k):
        cols_iteration_list.append(f'center_{a}')
        series_list.append(pd.Series(ind[:, a]))
        
    iteation_df = pd.concat(series_list, axis=1)
    iteation_df.columns = cols_iteration_list

    df = pd.concat([df, iteation_df], axis=1)
    df = df.rename(str, axis="columns")
    df[cols_iteration_list] = df[cols_iteration_list].astype('category')

    end = time.time()
    print_time(f' n columns {len(list(df))} in df', end - start)
    return df


def create_folds(df):
    # Create folds
    folds = np.random.randint(low=0, high=folds_size, size=len(df))

    folds_dict = {'folds': folds.tolist()}
    with open('folds.json', 'w') as f:
        json.dump(folds_dict, f)

    # with open('folds.json', 'r') as f:
    #     folds = np.array(json.load(f)['folds'])
    return folds


def run_model(df, remove_cols, level, folds_size, folds, previous_df_k_fold_results):
    start_first = time.time()
    #  'mst_anaf_lvl3',
    #  'mst_anaf_lvl2',
    #  'mst_anaf_lvl1',
    label_Secondery = 'mst_anaf_lvl1'
    if level == 'lv1':
        le_lv1 = LabelEncoder()
        df[label_Secondery + '_encoded'] = le_lv1.fit_transform(df[label_Secondery])

    results = []
    df_k_fold_results = {'train': [],
                        'test': [],
                        'x_test': [],
                        'y_test': [],
                        'model': [],
                        }
                        
    for fold in range(folds_size):
        start = time.time()
        print(f' {level} fold {fold}')
        train = df.loc[folds != fold, :].reset_index(drop=True)
        test = df.loc[folds == fold, :].reset_index(drop=True)
        if level == 'lv1':
            y_train, y_test = train[[label_Secondery + '_encoded']], test[[label_Secondery + '_encoded']]
            x_train, x_test = train.drop(remove_cols + 
                        [label_Secondery + '_encoded'], axis=1), test.drop(remove_cols + 
                        [label_Secondery + '_encoded'], axis=1)
        else:
            y_train, y_test = train[[label]], test[[label]]
            x_train, x_test = train.drop(remove_cols + 
                    [label_Secondery + '_encoded'], axis=1), test.drop(remove_cols + 
                    [label_Secondery + '_encoded'], axis=1)
            x_train[f'{level}_predicted'] = previous_df_k_fold_results['model'][fold].predict(x_train)
            x_test[f'{level}_predicted'] = previous_df_k_fold_results['model'][fold].predict(x_test)

        x_train, y_train = sampling_lmblearn(x_train, y_train, up_limit, seed)

        model = xgb.XGBClassifier(**param)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_predict)
        results.append(accuracy)

        df_k_fold_results['train'].append(train['tik'])
        df_k_fold_results['test'].append(test[remove_cols])
        df_k_fold_results['x_test'].append(x_test)
        df_k_fold_results['y_test'].append(y_test)
        df_k_fold_results['model'].append(model)
        end = time.time()
        print_time(f'  {level} accuracy {accuracy:.2f}', end - start)

    end = time.time()
    print_time(f' {level} mean accuracy {np.mean(results):.2f}', end - start_first)  
    return df_k_fold_results


def importance_features(df_k_fold_results, engine, version_model):
    start = time.time()
    model = df_k_fold_results['model'][0]
    xgb_fea_imp = pd.DataFrame(list(model.get_booster().get_fscore().items()),
                    columns=['feature','importance']).sort_values('importance', ascending=False).reset_index(drop=True)
    xgb_fea_imp.to_sql('branch_features_important_2022_' + version_model,
                    engine, schema="dbo", 
                        if_exists='replace',
                        index=False)  
    end = time.time()
    print_time(' importance', end - start)


def infernce_final_df(df_k_fold_results, le, remove_cols):
    start = time.time()
    list_dfs = []
    for fold in range(folds_size):
        print(f' fold {fold}')
        prob = df_k_fold_results['model'][fold].predict_proba(df_k_fold_results['x_test'][fold])
        current_prob = pd.Series(prob[np.arange(prob.shape[0]),
                            [a[0] for a in df_k_fold_results['y_test'][fold].values.tolist()]])
        best_branch = np.argmax(prob, axis=1)
        best_prob =  pd.Series(prob[np.arange(prob.shape[0]), best_branch])
        list_dfs.append(pd.concat([df_k_fold_results['test'][fold][remove_cols], 
                                    current_prob, pd.Series(le.inverse_transform(best_branch)),
                                    best_prob], axis=1))

        final_df = pd.concat(list_dfs, axis=0, ignore_index=True)
        final_df.columns = remove_cols + ['current_anaf_prob', 'best_anaf', 'best_anaf_prob']
        final_df[label] = le.inverse_transform(final_df[label])

    df_k_fold_results['le'] = le
    with open(f"metadata_{kpis_model_version}.pkl", "wb") as f:
            cloudpickle.dump(df_k_fold_results, f)
    end = time.time()
    print_time(' infernce final df', end - start)
    return final_df


def infernce(engine, version_model, prediction_table_name, final_df):
    start = time.time()
    anaf_df = pd.read_sql_table(
        "branch_features_panel_2022_" + version_model,
        schema="dbo",
        con=engine,
    )

    anaf_unique_df = anaf_df[[
                        'mst_anaf_lvl1',
                        'mst_anaf_lvl1_desc',
                        'mst_anaf_lvl2',
                        'mst_anaf_lvl2_desc',
                        'mst_anaf_lvl3',
                        'mst_anaf_lvl3_desc',
                        label,
                        'mst_anaf_lvl4_desc',
                        ]].drop_duplicates()

    anaf_unique_df_best = anaf_unique_df.rename(columns={label: 'best_anaf',
                                                'mst_anaf_lvl4_desc': 'mst_anaf_lvl4_desc_best',
                                                'mst_anaf_lvl3': 'mst_anaf_lvl3_best',
                                                'mst_anaf_lvl3_desc': 'mst_anaf_lvl3_desc_best',
                                                'mst_anaf_lvl2': 'mst_anaf_lvl2_best',
                                                'mst_anaf_lvl2_desc': 'mst_anaf_lvl2_desc_best',
                                                'mst_anaf_lvl1': 'mst_anaf_lvl1_best',
                                                'mst_anaf_lvl1_desc': 'mst_anaf_lvl1_desc_best'})

    final_df = final_df.merge(anaf_unique_df_best, on='best_anaf', how='inner')

    final_df[['mst_anaf_lvl2', 'mst_anaf_lvl3']] = final_df[['mst_anaf_lvl2', 
                                                    'mst_anaf_lvl3']].astype(float)

    final_df['distance_between_predictions'] = (10 - ((final_df.mst_anaf_lvl1 == final_df.mst_anaf_lvl1_best) * 1 +
                                                    (final_df.mst_anaf_lvl2 == final_df.mst_anaf_lvl2_best) * 2 +
                                                    (final_df.mst_anaf_lvl3 == final_df.mst_anaf_lvl3_best) * 3 +
                                                    (final_df[label] == final_df.best_anaf) * 4)) / 10

    final_df.to_sql(prediction_table_name, 
                    engine, schema="dbo", 
                    if_exists='replace',
                    index=False)
    end = time.time()
    print_time(' infernce predictions', end - start)
    return final_df, anaf_df


def infernce_kpis(engine, final_df, anaf_df):
    start = time.time()
    print(f'level 1 {accuracy_score(final_df.mst_anaf_lvl1.values, final_df.mst_anaf_lvl1_best.values):.2f}')
    print(f'level 2 {accuracy_score(final_df.mst_anaf_lvl2.values, final_df.mst_anaf_lvl2_best.values):.2f}')
    print(f'level 3 {accuracy_score(final_df.mst_anaf_lvl3.values, final_df.mst_anaf_lvl3_best.values):.2f}')
    print(f'level 4 {accuracy_score(final_df[label].values, final_df.best_anaf.values):.2f}')

    kpis_results = metrics_kpis(final_df[label].values.reshape(-1, 1), final_df.best_anaf.values.reshape(-1, 1))

    pd.DataFrame([kpis_results + [round(len(final_df) / len(anaf_df), 2)] + [kpis_model_version]], columns=['time', 
                                        'Accuracy', 
                                        'Precision', 
                                        'Recall', 
                                        'F1', 
                                        'MCC', 
                                        'Precision_Avg', 
                                        'Accuracy_Avg',
                                        'Coverage',
                                        'version']).to_sql('branch_kpis_2022_v1', 
                    engine, schema="dbo", 
                    if_exists='append',
                    index=False)
    end = time.time()
    print_time(' infernce kpis', end - start)


def main():
    engine = create_connection()
    words_columns = []
    print('load data')
    df = load_data(engine, version_model)
    if flag_words:
        df, words_columns = load_words(engine, version_model, df)
    df = fillter_data(sub_pop, engine, version_model, df)
    df, numeric_list, remove_cols, le = encoding_data(df, words_columns)
    df, numeric_list = transforamtion(engine, df, numeric_list, words_columns, remove_cols)
    df = create_centers(df, numeric_list)
    print('run model')
    folds = create_folds(df)
    lv1_df_k_fold_results = run_model(df, remove_cols, 'lv1', folds_size, folds, {})
    lv4_df_k_fold_results = run_model(df, remove_cols, 'lv4', folds_size, folds, lv1_df_k_fold_results)
    del lv1_df_k_fold_results, df
    if not n_features:
        importance_features(lv4_df_k_fold_results, engine, version_model)
    print('run inference')
    final_df = infernce_final_df(lv4_df_k_fold_results, le, remove_cols)
    final_df, anaf_df = infernce(engine, version_model, prediction_table_name, final_df)
    infernce_kpis(engine, final_df, anaf_df)


if __name__ == '__main__':
    main()
