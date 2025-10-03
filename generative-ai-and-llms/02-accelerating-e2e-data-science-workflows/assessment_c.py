#step 3 correct

%load_ext cudf.pandas

import pandas as pd
import cuml

df = pd.read_csv('./data/week3.csv')
df['infected'] = df['infected'].astype('float32')
emp_groups = df[['employment','infected']].groupby('employment')
emp_rate_df = emp_groups.mean()
emp_codes = pd.read_csv('./data/code_guide.csv')

top_inf_emp = emp_rate_df.sort_values(by='infected', ascending=False).iloc[0:2].index
top_inf_emp_df = emp_codes.loc[emp_codes['Code'].isin(top_inf_emp), 'Field']
top_inf_emp_df.to_json('my_assessment/question_3.json', orient='records')