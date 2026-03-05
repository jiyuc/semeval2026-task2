import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import argparse

def split_by_user_id_kfold(
    df: pd.DataFrame,
    user_col: str = "user_id",
    n_splits: int = 5,
    random_state: int = 42
):
    """
    Split a DataFrame into K-folds based on unique users.
    Each fold ensures no user overlap between train and test sets.
    """
    users = df[user_col].dropna().unique()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(kf.split(users), start=1):
        users_train = users[train_idx]
        users_test = users[test_idx]

        train_df = df[df[user_col].isin(users_train)].reset_index(drop=True)
        test_df = df[df[user_col].isin(users_test)].reset_index(drop=True)

        yield fold, train_df, test_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data_dir", default="data/")
    args = parser.parse_args()

    data_files = {
        "subtask1": f"{args.source_data_dir}train_subtask1.csv",
        "subtask2a": f"{args.source_data_dir}train_subtask2a.csv",
        "subtask2b": f"{args.source_data_dir}train_subtask2b.csv",
        # "subtask2b_detailed": f"{args.source_data_dir}train_subtask2b_detailed.csv",
        # "subtask2b_user_disposition_change": f"{args.source_data_dir}train_subtask2b_user_disposition_change.csv",
    }

    for k, file in data_files.items():
        df = pd.read_csv(file, encoding="utf-8")
        print(f"- {k}")

        for fold, train, test in split_by_user_id_kfold(df, user_col="user_id", n_splits=5, random_state=42):
            print(f"  - Fold {fold}")
            print(f"    - Train users: {len(train['user_id'].unique())}; Test users: {len(test['user_id'].unique())}")
            print(f"    - Train: {len(train)}; Test: {len(test)}")

            train.to_csv(f"{args.source_data_dir}split/{k}_train_cv{fold}.csv", index=False)
            test.to_csv(f"{args.source_data_dir}split/{k}_test_cv{fold}.csv", index=False)