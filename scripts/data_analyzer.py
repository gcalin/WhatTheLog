import pandas as pd
from whatthelog.definitions import PROJECT_ROOT

metrics_0 = pd.read_csv(PROJECT_ROOT.joinpath('resources/k_fold_metrics/600/metrics_0.csv'))
metrics_1 = pd.read_csv(PROJECT_ROOT.joinpath('resources/k_fold_metrics/600/metrics_1.csv'))
metrics_2 = pd.read_csv(PROJECT_ROOT.joinpath('resources/k_fold_metrics/600/metrics_2.csv'))
metrics_3 = pd.read_csv(PROJECT_ROOT.joinpath('resources/k_fold_metrics/600/metrics_3.csv'))
metrics_4 = pd.read_csv(PROJECT_ROOT.joinpath('resources/k_fold_metrics/600/metrics_4.csv'))

metric_0 = metrics_0.iloc[-1]
metric_1 = metrics_1.iloc[-1]
metric_2 = metrics_2.iloc[-1]
metric_3 = metrics_3.iloc[-1]
metric_4 = metrics_4.iloc[-1]

df = pd.DataFrame(columns=("Data set size", "Seed", "Episodes", "Recall", "Specificity",
                           "Precision", "Compression", "Nodes", "Transitions", "Duration (s)"))

durations_df = pd.DataFrame(columns=("Data set size", "Duration"))

for size in [100, 200, 400, 600, 800, 1000]:
    for i in range(5):
        metrics = pd.read_csv(PROJECT_ROOT.joinpath(f'resources/k_fold_metrics/{size}/metrics_{i}.csv'))
        new_row = {}

        metric = metrics.iloc[-1]
        new_row['Data set size'] = size
        new_row['Seed'] = i
        new_row['Episodes'] = metric["episode"] + 1
        new_row['Recall'] = metric[" recall"]
        new_row['Specificity'] = metric[" specificity"]
        new_row['Precision'] = metric[" precision"]
        new_row['Compression'] = metric[" size"]
        new_row['Nodes'] = metric[" nodes"]
        new_row['Transitions'] = metric[" transitions"]
        new_row['Duration (s)'] = metrics[" duration"].sum()
        row_to_be_added = pd.Series(new_row, name=df.size)

        df = df.append(row_to_be_added)
        durations_df = durations_df.append(pd.Series({"Data set size": new_row["Data set size"], "Duration": new_row["Duration (s)"]}, name=durations_df.size))

df = df.round(2)
df[["Data set size", "Seed", "Episodes", "Nodes", "Transitions", "Duration (s)"]] = \
    df[["Data set size", "Seed", "Episodes", "Nodes", "Transitions", "Duration (s)"]].astype(int)
df.to_latex(PROJECT_ROOT.joinpath("resources/k_fold_metrics/all_results_table.tex"), index=False)

durations_df: pd.DataFrame = durations_df.groupby("Data set size").mean()
durations_df.columns = ["duration"]
durations_df.index.name = "data_set_size"
durations_df.to_csv(PROJECT_ROOT.joinpath("resources/k_fold_metrics/scalability.csv"))

