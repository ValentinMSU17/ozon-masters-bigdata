from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV


#
# Dataset fields
#
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)]

fields = ["id", "label"] + numeric_features + categorical_features + ["day_number"]
categorical_features = ["cf"+str(i) for i in range(1,27) if i not in [1,10,20,21,22]]
#
# Model pipeline
#

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Now we have a full prediction pipeline.
#class_weight='balanced', solver='saga', penalty='l1'
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('linearregression', LogisticRegression(class_weight='balanced', solver='lbfgs'))
])