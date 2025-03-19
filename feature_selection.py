from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Assuming `df` is your dataset with features and `target` is your label
X = df.drop("target", axis=1)
y = df["target"]

# Handle missing data, encode categorical variables, etc.

# 1. Filter Method (Chi-Square for categorical)
chi2_selector = SelectKBest(chi2, k=50)  # Select top 50 features
X_chi2 = chi2_selector.fit_transform(X, y)
selected_chi2_features = X.columns[chi2_selector.get_support()]

# 2. Embedded Method (Random Forest)
rf = RandomForestClassifier()
rf.fit(X, y)
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
top_rf_features = rf_importances.head(50).index.tolist()

# 3. Wrapper Method (RFE with Logistic Regression)
lr = LogisticRegression(max_iter=1000)
rfe_selector = RFE(lr, n_features_to_select=50)
rfe_selector = rfe_selector.fit(X, y)
selected_rfe_features = X.columns[rfe_selector.support_]

# Combine selected features
final_selected_features = (
    set(selected_chi2_features) | set(top_rf_features) | set(selected_rfe_features)
)

# Final dataset
X_selected = X[list(final_selected_features)]
