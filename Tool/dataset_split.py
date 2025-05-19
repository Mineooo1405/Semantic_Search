import pandas as pd
from sklearn.model_selection import train_test_split

triplets_path = "D:/SemanticSearch/TrainingData_MatchZoo_BEIR/msmarco_semantic-grouping/train_2/msmarco_semantic-grouping_train_triplets.tsv"

df = pd.read_csv(triplets_path, sep='\t', header=None, names=["query", "pos", "neg"])
train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42)

train_df.to_csv("train.tsv", sep='\t', index=False, header=False)
dev_df.to_csv("dev.tsv", sep='\t', index=False, header=False)
