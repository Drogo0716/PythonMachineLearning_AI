import pandas as pd, csv

df = pd.read_csv("Dataset2_Distinct_TrueTest.csv")
df2 = df.dropna()

df2.to_csv("Dataset2_Distinct_TrueTest_PreRaceEncoding_PostDropNA.csv")