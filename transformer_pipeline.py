from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer



# This code is the outline for a transformer pipeline to be used with
# Pandas Dataframes. The names of members/variables will need to be changed
# during implementation.

# See pg 72 for coding the CombinedAttributesAdder

# Imputer with 'median' strategy will replace missing values with the median value for that attrib
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # median can only be evaluated for numerical attribs
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])
# should drop cat vars from the dataset before running this next line
# dataframe_transformed = num_pipline.fit_transform(dataframe_numerical)
# above line is not needed if using the full_pipeline method

num_attrib = list(dataframe)
cat_attrib = ['''List_Cat_Attribs_Here''']

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attrib),
    ("cat", OneHotEncoder(), cat_attrib)
])

dataframe_prepared_final = full_pipeline.fit_transform(dataframe)