import pandas as pd

df = pd.read_csv("./data/KRD-WBC dataset/class labels.csv")
print(df["Categoriy"].value_counts())