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
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
import shap

# Model configuration parameters
folds_size = 2          # Number of cross-validation folds
seed = 3                # Random seed for reproducibility 
up_limit = 1200        # Maximum samples per class after balancing
sub_pop = 3            # Subpopulation selection criteria
flag_words = True      # Whether to use text features
how_df_words = 'inner' # Join type for text features
k = 8                  # Number of nearest centroids to compute
label = 'MST_ANAF_A'   # Target variable column name
n_features = 900       # Number of features to use
min_companies = 20     # Minimum companies per class
n_cpus = -1           # Number of CPU cores (-1 for all)

# Model version tracking
version_model = 'v3'
version_model_output = version_model + '_6'
kpis_model_version =  version_model_output + f'_{n_features}_{sub_pop}_{up_limit}_{min_companies}_{folds_size}_{k}_{flag_words * 1}_{how_df_words}'
prediction_table_name = 'branch_prediction_2022_' + version_model_output
importance_table_name = 'branch_features_important_2022_' + version_model

# XGBoost parameters
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

# Set random seeds for reproducibility
np.random.seed(seed)
random.seed(seed)

def print_time(str_time, gap_time):
    """Helper function to print elapsed time with message"""
    print(f'{str_time}, time {gap_time:.2f}')

def split_it(llm_output):
    """Split text into words, handling None values"""
    if llm_output:
        return re.findall(r"[\w']+", llm_output)
    return []

def masked_cosine_similarity_df(records_df, centroids_df):
    """
    Compute manhattan distances between records and centroids, handling missing values.
    Returns distance matrix between records and centroids.
    """
    assert len(records_df.columns) == len(centroids_df.columns), "Columns must match"
    
    records_array = records_df.fillna(0).to_numpy()
    centroids_array = centroids_df.fillna(0).to_numpy()
    records_mask = ~records_df.isna().to_numpy()
    centroids_mask = ~centroids_df.isna().to_numpy()
    
    records_sparse = records_array * records_mask
    centroids_sparse = centroids_array * centroids_mask
    
    return manhattan_distances(records_sparse, centroids_sparse)

def sampling_lmblearn(X, y, up_limit, random_state):
    """
    Balance class distribution by undersampling majority classes to up_limit samples.
    Returns balanced X and y datasets.
    """
    categories_count = y.value_counts()
    target_samples = {k: up_limit for k, _ in categories_count[categories_count > up_limit].to_dict().items() if k in y}
    rus = RandomUnderSampler(sampling_strategy=target_samples, random_state=random_state)
    return rus.fit_resample(X, y)

def metrics_kpis(real_values, pred_values):
    """Calculate and return multiple classification metrics"""
    time = datetime.now()
    return [
        time,
        round(accuracy_score(real_values, pred_values), 4),
        round(precision_score(real_values, pred_values, average='macro'), 4),
        round(recall_score(real_values, pred_values, average='macro'), 4),
        round(f1_score(real_values, pred_values, average='macro'), 4),
        round(matthews_corrcoef(real_values, pred_values), 4),
        round(average_precision_score(real_values, pred_values), 4),
        round(balanced_accuracy_score(real_values, pred_values), 4)
    ]

def create_connection():
    """Create and return SQL database connection"""
    servername = 'bisqldwhd1'
    dbname = 'SRC_MCH'
    return create_engine(f'mssql+pyodbc://@{servername}/{dbname}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server')

def load_data(engine, version_model):
    """Load base feature data from database"""
    start = time.time()
    df = pd.read_sql_table(
        "branch_features_panel_2022_" + version_model,
        schema="dbo",
        con=engine,
    ).sample(5000)
    end = time.time()
    print_time(' load data feature panel', end - start)
    return df

def load_words(engine, version_model, df):
    """Load and process text features if enabled"""
    start = time.time()
    df_words = pd.read_sql_table(
        "branch_llm_branch_2022_" + version_model,
        schema="dbo",
        con=engine,
    )
    
    # Process text into word columns
    df_words['words'] = df_words.branch_llm.apply(split_it)
    df_words['words'] = df_words['words'].apply(lambda d: d if isinstance(d, list) else [])
    df_words = pd.DataFrame(df_words.words.values.tolist(), df_words.tik).add_prefix('word_')
    
    # Limit word columns and merge with main dataframe
    if len(list(df_words)) > 4:
        df_words.drop(columns=[f'word_{i}' for i in range(4, len(list(df_words)))], inplace=True)
    words_columns = list(df_words)
    df_words = df_words.replace('\n', '').astype('category')
    df_words.reset_index(inplace=True)
    df = df.merge(df_words, on='tik', how=how_df_words)
    
    end = time.time()
    print_time(f' words n records {len(df_words)}', end - start)
    return df, words_columns

def fillter_data(sub_pop, engine, version_model, df):
    """Apply filtering based on subpopulation criteria"""
    start = time.time()
    
    # Different filtering logic based on sub_pop value
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

    # Filter out classes with too few samples
    categories = df[label].value_counts()
    df = df[df[label].isin(categories[categories > min_companies].index)].reset_index(drop=True)
    
    end = time.time()
    print_time(f' n records {len(df)}, n classes {df[label].nunique()}', end - start)
    return df

def encoding_data(df, words_columns):
    """Transform data types and encode categorical variables"""
    start = time.time()
    
    # Encode target variable
    le = LabelEncoder()
    df[label] = le.fit_transform(df[label])

    # Remove specific financial columns that aren't needed for modeling
    df = df.drop(['MST_D_HACHNASA_ESEK_R',
                 'MST_D_HACHNASA_SACAR_R', 'MST_D_HACHNASA_SACAR_BZ', 
                 'MST_MAAM_R', 'MST_KNAS_GERAON', 'MST_HACHNASOT_ESEK_R', 'MST_MAS_MEGIA_R',
                 'MST_NIKUI_MAKOR_SACHAR', 'MST_ITRAT_ZIK_R', 'doh_anaf', 'MST_HACHNASA_SACAR_R', 
                 'same_anaf'], axis=1)

    # Remove columns with only one unique value
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col, inplace=True, axis=1)

    # Define categorical columns
    categorical_list_tmp = ['KOD_DOAAR_CHOZER',
                        'YSHUV',
                        'YSHUV_ESEK',
                        'YSHUV_PRATI',
                        # ... rest of categorical columns
                        'prt_shem_tik'
                        ] 
    
    # Add word columns to categorical list if text features are enabled
    if flag_words:
        categorical_list_tmp += words_columns

    # Filter categorical list to only include columns present in dataframe
    categorical_list = [col for col in categorical_list_tmp if col in df.columns]
    
    # Define columns to remove
    remove_cols = [col for col in ['tik', label, 'mst_anaf_lvl4_desc', # ... other columns
                                 'matara'] if col in df.columns]

    # Define continuous columns to remove
    remove_continuos = [col for col in ['hevrot_y_rishum', 'PRT_SM_ST'] 
                       if col in df.columns]

    # Convert categorical columns
    df[categorical_list] = df[categorical_list].astype('category')

    # Get list of numeric columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_list = df.select_dtypes(include=numerics).columns.tolist()

    # Remove specified columns from numeric list
    for col in remove_cols + remove_continuos:
        if col in numeric_list:
            numeric_list.remove(col)

    end = time.time()
    print_time(f' n numeric features {len(numeric_list)}', end - start)
    return df, numeric_list, remove_cols, le

def transforamtion(engine, df, numeric_list, words_columns, remove_cols):
    """Create ratio features and apply feature selection"""
    start = time.time()
    
    # Create ratio features between all numeric columns
    series_list = []
    cols_iteration_list = []
    for i, a in enumerate(numeric_list[:-1]):
        for b in numeric_list[i + 1:]:
            if a != b:
                cols_iteration_list.append(a + '|' + b)
                series_list.append(df[a] / df[b])

    # Combine ratio features with original dataframe
    iteation_df = pd.concat(series_list, axis=1)
    iteation_df.columns = cols_iteration_list
    df = pd.concat([df, iteation_df], axis=1)
    df = df.rename(str, axis="columns")
    
    # Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_list += cols_iteration_list

    # Apply feature selection if n_features is specified
    if n_features:
        # Load feature importance rankings
        xgb_fea_imp = pd.read_sql_table(importance_table_name,
                        engine, schema="dbo")    
        
        # Select top n_features plus word features if enabled
        selected_features = xgb_fea_imp.feature.head(n_features).values.tolist()
        if flag_words:
            selected_features += words_columns

        # Filter columns based on selected features
        columns_list = [col for col in df.columns if col in selected_features]
        df = df[remove_cols + columns_list]

        # Update numeric list to only include selected features
        numeric_list = [col for col in columns_list if col in numeric_list]

    end = time.time()
    print_time(f' n columns numeric {len(numeric_list)}', end - start)
    return df, numeric_list

def create_centers(df, numeric_list):
    """Create centroid-based features using standardized numeric data"""
    start = time.time()
    
    # Standardize numeric features
    scaler = StandardScaler()
    df[numeric_list] = scaler.fit_transform(df[numeric_list])

    # Calculate class centroids and standard deviations
    df_group_mean = df[numeric_list + [label]].groupby(label).mean()
    df_group_std = df[numeric_list + [label]].groupby(label).std()
    df_group_mean.columns = [col + '_mean' for col in numeric_list]
    df_group_std.columns = [col + '_std' for col in numeric_list]

    # Calculate distances to centroids and find k nearest
    distance = masked_cosine_similarity_df(df[numeric_list], df_group_mean)
    ind = np.argpartition(distance, k, axis=1)[:, :k]

    # Create centroid features
    series_list = []
    cols_iteration_list = []
    for a in range(k):
        cols_iteration_list.append(f'center_{a}')
        series_list.append(pd.Series(ind[:, a]))
        
    iteation_df = pd.concat(series_list, axis=1)
    iteation_df.columns = cols_iteration_list

    # Add centroid features to dataframe
    df = pd.concat([df, iteation_df], axis=1)
    df = df.rename(str, axis="columns")
    df[cols_iteration_list] = df[cols_iteration_list].astype('category')

    end = time.time()
    print_time(f' n columns {len(list(df))} in df', end - start)
    return df

def create_folds(df):
    """Create random cross-validation folds"""
    # Generate random fold assignments
    folds = np.random.randint(low=0, high=folds_size, size=len(df))

    # Save fold assignments for reproducibility
    folds_dict = {'folds': folds.tolist()}
    with open('folds.json', 'w') as f:
        json.dump(folds_dict, f)

    return folds

def run_model(df, remove_cols, level, folds_size, folds, previous_df_k_fold_results):
    """
    Train and evaluate model for a specific hierarchical level.
    
    Args:
        df: Input dataframe
        remove_cols: Columns to exclude
        level: Hierarchical level ('lv1' or 'lv4')
        folds_size: Number of cross-validation folds
        folds: Fold assignments
        previous_df_k_fold_results: Results from previous level model
    """
    start_first = time.time()
    label_Secondery = 'mst_anaf_lvl1'
    
    # For level 1, encode secondary label
    if level == 'lv1':
        le_lv1 = LabelEncoder()
        df[label_Secondery + '_encoded'] = le_lv1.fit_transform(df[label_Secondery])

    # Initialize results tracking
    results = []
    df_k_fold_results = {
        'train': [],
        'test': [],
        'x_test': [],
        'y_test': [],
        'model': [],
    }
                        
    # Train and evaluate model for each fold
    for fold in range(folds_size):
        start = time.time()
        print(f' {level} fold {fold}')
        
        # Split data into train and test
        train = df.loc[folds != fold, :].reset_index(drop=True)
        test = df.loc[folds == fold, :].reset_index(drop=True)
        
        # Prepare features and target based on level
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
            # Add predictions from previous level as features
            x_train[f'{level}_predicted'] = previous_df_k_fold_results['model'][fold].predict(x_train)
            x_test[f'{level}_predicted'] = previous_df_k_fold_results['model'][fold].predict(x_test)

        # Balance classes and train model
        x_train, y_train = sampling_lmblearn(x_train, y_train, up_limit, seed)
        model = xgb.XGBClassifier(**param)
        model.fit(x_train, y_train)
        
        # Evaluate model
        y_predict = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_predict)
        results.append(accuracy)

        # Store results
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
    """Calculate and store feature importance scores"""
    start = time.time()
    
    # Get feature importance from first fold's model
    model = df_k_fold_results['model'][0]
    xgb_fea_imp = pd.DataFrame(
        list(model.get_booster().get_fscore().items()),
        columns=['feature','importance']
    ).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Store feature importance in database
    xgb_fea_imp.to_sql(
        'branch_features_important_2022_' + version_model,
        engine, schema="dbo", 
        if_exists='replace',
        index=False
    )  
    
    end = time.time()
    print_time(' importance', end - start)

def infernce_final_df(df_k_fold_results, le, remove_cols):
    """Generate final predictions and probabilities for all folds"""
    start = time.time()
    list_dfs = []
    
    # Process each fold
    for fold in range(folds_size):
        print(f' fold {fold}')
        # Get prediction probabilities
        prob = df_k_fold_results['model'][fold].predict_proba(df_k_fold_results['x_test'][fold])
        
        # Get probability for current class and best class
        current_prob = pd.Series(prob[np.arange(prob.shape[0]),
                            [a[0] for a in df_k_fold_results['y_test'][fold].values.tolist()]])
        best_branch = np.argmax(prob, axis=1)
        best_prob = pd.Series(prob[np.arange(prob.shape[0]), best_branch])
        
        # Combine results
        list_dfs.append(pd.concat([
            df_k_fold_results['test'][fold][remove_cols], 
            current_prob, 
            pd.Series(le.inverse_transform(best_branch)),
            best_prob
        ], axis=1))

    # Combine results from all folds
    final_df = pd.concat(list_dfs, axis=0, ignore_index=True)
    final_df.columns = remove_cols + ['current_anaf_prob', 'best_anaf', 'best_anaf_prob']
    final_df[label] = le.inverse_transform(final_df[label])

    # Save model metadata
    df_k_fold_results['le'] = le
    with open(f"metadata_{kpis_model_version}.pkl", "wb") as f:
        cloudpickle.dump(df_k_fold_results, f)
            
    end = time.time()
    print_time(' infernce final df', end - start)
    return final_df

def infernce(engine, version_model, prediction_table_name, final_df):
    """Add branch hierarchy information and calculate distances between predictions"""
    start = time.time()
    
    # Load branch hierarchy data
    anaf_df = pd.read_sql_table(
        "branch_features_panel_2022_" + version_model,
        schema="dbo",
        con=engine,
    )

    # Get unique branch hierarchies
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

    # Rename columns for predicted branch
    anaf_unique_df_best = anaf_unique_df.rename(columns={
        label: 'best_anaf',
        'mst_anaf_lvl4_desc': 'mst_anaf_lvl4_desc_best',
        'mst_anaf_lvl3': 'mst_anaf_lvl3_best',
        'mst_anaf_lvl3_desc': 'mst_anaf_lvl3_desc_best',
        'mst_anaf_lvl2': 'mst_anaf_lvl2_best',
        'mst_anaf_lvl2_desc': 'mst_anaf_lvl2_desc_best',
        'mst_anaf_lvl1': 'mst_anaf_lvl1_best',
        'mst_anaf_lvl1_desc': 'mst_anaf_lvl1_desc_best'
    })

    # Merge predictions with hierarchy information
    final_df = final_df.merge(anaf_unique_df_best, on='best_anaf', how='inner')
    final_df[['mst_anaf_lvl2', 'mst_anaf_lvl3']] = final_df[['mst_anaf_lvl2', 
                                                    'mst_anaf_lvl3']].astype(float)

    # Calculate hierarchical distance between actual and predicted branches
    final_df['distance_between_predictions'] = (10 - ((final_df.mst_anaf_lvl1 == final_df.mst_anaf_lvl1_best) * 1 +
                                                    (final_df.mst_anaf_lvl2 == final_df.mst_anaf_lvl2_best) * 2 +
                                                    (final_df.mst_anaf_lvl3 == final_df.mst_anaf_lvl3_best) * 3 +
                                                    (final_df[label] == final_df.best_anaf) * 4)) / 10

    # Store final results
    final_df.to_sql(prediction_table_name, 
                    engine, schema="dbo", 
                    if_exists='replace',
                    index=False)
                    
    end = time.time()
    print_time(' infernce predictions', end - start)
    return final_df, anaf_df

def infernce_kpis(engine, final_df, anaf_df):
    """Calculate and store model performance metrics"""
    start = time.time()
    
    # Print accuracy for each hierarchical level
    print(f' level 1 {accuracy_score(final_df.mst_anaf_lvl1.values, final_df.mst_anaf_lvl1_best.values):.2f}')
    print(f' level 2 {accuracy_score(final_df.mst_anaf_lvl2.values, final_df.mst_anaf_lvl2_best.values):.2f}')
    print(f' level 3 {accuracy_score(final_df.mst_anaf_lvl3.values, final_df.mst_anaf_lvl3_best.values):.2f}')
    print(f' level 4 {accuracy_score(final_df[label].values, final_df.best_anaf.values):.2f}')

    # Calculate detailed metrics
    kpis_results = metrics_kpis(final_df[label].values.reshape(-1, 1), final_df.best_anaf.values.reshape(-1, 1))

    # Store metrics in database
    pd.DataFrame([kpis_results + [round(len(final_df) / len(anaf_df), 2)] + [kpis_model_version]], 
                columns=['time', 
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
    """Main execution function for the model pipeline"""
    # Initialize database connection
    engine = create_connection()
    words_columns = []
    
    print('load data')
    # Load and prepare data
    df = load_data(engine, version_model)
    if flag_words:
        df, words_columns = load_words(engine, version_model, df)
    df = fillter_data(sub_pop, engine, version_model, df)
    df, numeric_list, remove_cols, le = encoding_data(df, words_columns)
    df, numeric_list = transforamtion(engine, df, numeric_list, words_columns, remove_cols)
    df = create_centers(df, numeric_list)
    
    print('run model')
    # Train and evaluate models
    folds = create_folds(df)
    lv1_df_k_fold_results = run_model(df, remove_cols, 'lv1', folds_size, folds, {})
    lv4_df_k_fold_results = run_model(df, remove_cols, 'lv4', folds_size, folds, lv1_df_k_fold_results)
    del lv1_df_k_fold_results, df
    
    # Calculate feature importance if needed
    if not n_features:
        importance_features(lv4_df_k_fold_results, engine, version_model)
    
    print('run inference')
    # Generate and store final predictions and metrics
    final_df = infernce_final_df(lv4_df_k_fold_results, le, remove_cols)
    final_df, anaf_df = infernce(engine, version_model, prediction_table_name, final_df)
    infernce_kpis(engine, final_df, anaf_df)

if __name__ == '__main__':
    main()
