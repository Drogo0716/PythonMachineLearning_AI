import sys, logging, pandas as pd

logging.basicConfig(filename='DF_Splitting_Log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

pd.set_option('display.max_columns', None)  # set this number to >= your number of cols
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('Dataset2_Distinct_TrueTest.csv', dtype=object)
df.dropna(inplace=True)

pred_col = df['Risk_Score']
data_col = df.drop(['Risk_Score'], axis=1)
logging.info('Here is the input data before true testing:\n\n')
logging.info(data_col)
logging.info('Here is the correct/expected Risk_Scores of each patient:\n\n')
logging.info(pred_col)

pred_col.to_csv('TrueResults.csv', index=False)
