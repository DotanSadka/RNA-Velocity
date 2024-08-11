import pandas as pd


def filter_TCell_closeness(is_close_file, file):
    """
    Filter from binary closeness indication in "is_close_file" if a cell was in interaction with tumor
    Save it in a new column in "file"
    """
    is_close = pd.read_csv(is_close_file)
    TCell = pd.read_csv(file)

    filtered_values = is_close[is_close["cell_name"].isin(TCell["Unnamed: 0"])]["count_tumor_total"]

    TCell["isClose"] = list(filtered_values)

    TCell.to_csv(f"closeness_of_{file}", index=False)


if __name__ == "__main__":
    for identity in ["3D", "3E", "4", "8A"]:
        filter_TCell_closeness("adata_genes_1311.csv", f"T-cell_CD{identity}.csv")
