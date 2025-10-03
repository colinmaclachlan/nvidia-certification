import cudf
import cuml

import cupy as cp

gdf = cudf.read_csv('./data/clean_uk_pop_full.csv', usecols=['age', 'sex', 'infected'])
gdf.dtypes
gdf.shape
gdf.head()

logreg = cuml.LogisticRegression()
logreg.fit(gdf[['age', 'sex']], gdf['infected'])

type(logreg.coef_)
type(logreg.intercept_)

logreg_coef = logreg.coef_
logreg_int = logreg.intercept_

print("Coefficients: [age, sex]")
print([logreg_coef[0], logreg_coef[1]])

print("Intercept:")
print(logreg_int[0])

class_probs = logreg.predict_proba(gdf[['age', 'sex']])
class_probs

gdf['risk'] = class_probs[1]

gdf.take(cp.random.choice(gdf.shape[0], size=5, replace=False))

# %load solutions/risk_by_age
age_groups = gdf[['age', 'infected']].groupby(['age'])
print(age_groups.mean().head())
print(age_groups.mean().tail())

sex_groups = gdf[['sex', 'infected']].groupby(['sex'])
sex_groups.mean()

X_train, X_test, y_train, y_test  = cuml.train_test_split(gdf[['age', 'sex']], gdf['infected'], train_size=0.9)
logreg = cuml.LogisticRegression()
logreg.fit(X_train, y_train)

y_test_pred = logreg.predict_proba(X_test, convert_dtype=True)[1]
y_test_pred.index = X_test.index
y_test_pred

test_results = cudf.DataFrame()
test_results['age'] = X_test['age']
test_results['sex'] = X_test['sex']
test_results['infected'] = y_test
test_results['predicted_risk'] = y_test_pred

test_results['high_risk'] = test_results['predicted_risk'] > test_results['predicted_risk'].mean()

risk_groups = test_results.groupby('high_risk')
risk_groups.mean()

s_groups = test_results[['sex', 'age', 'infected', 'predicted_risk']].groupby(['sex', 'age'])
s_groups.mean()

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
