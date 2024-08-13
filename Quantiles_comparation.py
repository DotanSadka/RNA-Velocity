import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_max_k_100(df, df_5_percent, df_7point5_percent, df_10_percent):
    # Extract row with index 5
    row = df.loc[5]

    # Sort the row values in descending order
    copy_row = row.copy()
    indices = [(i, value) for i, value in enumerate(copy_row)]
    sorted_row = sorted(indices, key=lambda x: x[1])[::-1]

    # Get the top 10 values from the sorted row
    top_values = [value[1] for value in sorted_row][1:21]
    top_indices = [index[0] for index in sorted_row][1:21]

    # Print the top 5 values and their indices
    x_value = [0.05, 0.075, 0.1]
    genes = list(df.columns)

    y_value = []
    for value, index in zip(top_values, top_indices):
        y_value.append(df_5_percent.iloc[5, index])
        y_value.append(df_7point5_percent.iloc[5, index])
        y_value.append(df_10_percent.iloc[5, index])

        gene = genes[index]

        plt.plot(x_value, y_value, 'o')
        plt.plot(x_value, y_value, '-')
        plt.xlabel('Quantiles value')
        plt.ylabel('Gammas value')
        title_fontsize = 10
        plt.title(f"K: 100 Gene: {gene}\npercentage change: {value}\n", fontsize=title_fontsize)
        plt.show()
        y_value = []


def create_percentage_change_df():
    # df = pd.read_csv("cell_pc_df.csv")
    # genes_name = list(df.columns)[1:]

    df_5_percent = pd.read_csv('quantiles_0.05.csv')
    df_7point5_percent = pd.read_csv('quantiles_0.075.csv')
    df_10_percent = pd.read_csv('quantiles_0.1.csv')

    # Initialize empty data frame for the divided values
    # divided_df = pd.DataFrame()
    #
    #
    # counter = 0
    # # Iterate over the data frames using zip()
    # for (col1, col2, col3) in zip(df_5_percent.columns, df_7point5_percent.columns, df_10_percent.columns):
    #     if counter < 1:
    #         change_col = pd.DataFrame({col1: [max(a, b, c) for a, b, c in
    #                                           zip(df_5_percent[col1], df_7point5_percent[col2], df_10_percent[col3])]})
    #         counter += 1
    #     else:
    #         # Calculate the division of maximum and minimum values for the corresponding blocks
    #         change_col = pd.DataFrame({col1: [100 * (max(abs(a - b)/b, abs(b - c)/b)) for a, b, c in
    #                                           zip(df_5_percent[col1], df_7point5_percent[col2], df_10_percent[col3])]})
    #
    #     # Append the calculated values to the divided_df data frame
    #     divided_df = pd.concat([divided_df, change_col], axis=1)
    #
    # print(divided_df)
    # divided_df.to_csv("percentage_change.csv")

    # get_max(pd.read_csv("percentage_change.csv"), df_5_percent, df_7point5_percent, df_10_percent)
    get_max_k_100(pd.read_csv("percentage_change.csv"), df_5_percent, df_7point5_percent, df_10_percent)


def get_max(df, df_5_percent, df_7point5_percent, df_10_percent):
    array = np.array(df)

    # Flatten the array to a 1D array - ignore first column
    flattened_array = array.flatten()

    # Sort the flattened array in descending order and get the indices
    sorted_indices = np.argsort(flattened_array)[::-1]

    # Get the top 5 values and their indices
    top_values = flattened_array[sorted_indices[:20]]
    top_indices = np.unravel_index(sorted_indices[:20], array.shape)

    # Print the top 5 values and their indices
    x_value = [0.05, 0.075, 0.1]
    genes = list(df.columns)

    y_value = []
    for value, index in zip(top_values, zip(*top_indices)):
        y_value.append(df_5_percent.iloc[index])
        y_value.append(df_7point5_percent.iloc[index])
        y_value.append(df_10_percent.iloc[index])

        k = df.iloc[index[0], 0]
        gene = genes[index[1]]

        plt.plot(x_value, y_value, 'o')
        plt.plot(x_value, y_value, '-')
        plt.xlabel('Quantiles value')
        plt.ylabel('Gammas value')
        title_fontsize = 10
        plt.title(f"K: {k} Gene: {gene}\npercentage change: {value}\n", fontsize=title_fontsize)
        plt.show()
        y_value = []

def plot_top():
    df = pd.read_csv('quantiles_0.1.csv')
    df_p = pd.read_csv('percentage_change_quantile_10.csv')
    row = df_p.loc[0]

    # Sort the row values in descending order
    copy_row = row.copy()
    indices = [(i, value) for i, value in enumerate(copy_row)]
    sorted_row = sorted(indices[1:], key=lambda x: x[1])[::-1]

    # Get the top 10 values from the sorted row
    top_values = [value[1] for value in sorted_row][1:21]
    top_indices = [index[0] for index in sorted_row][1:21]

    # Print the top values and their indices
    x_value = [100, 200]
    genes = list(df.columns)
    y_value = []
    for value, index in zip(top_values, top_indices):
        y_value.append(df.iloc[5, index])
        y_value.append(df.iloc[6, index])

        gene = genes[index]

        plt.plot(x_value, y_value, 'o')
        plt.plot(x_value, y_value, '-')
        plt.xlabel('Quantiles value')
        plt.ylabel('Gammas value')
        title_fontsize = 10
        plt.title(f"K: 100 Gene: {gene}\npercentage change: {value}\n", fontsize=title_fontsize)
        # plt.show()
        plt.savefig(f'TopPercentageChange/{gene}.png')
        plt.clf()
        y_value = []


def create_pc_quantile10():
    df_10_percent = pd.read_csv('quantiles_0.1.csv')

    divided_df = pd.DataFrame()

    counter = 0
    # Iterate over the data frames using zip()
    for col in df_10_percent.columns:
        if counter < 1:
            change_col = pd.DataFrame({col: ['Percent change']})
            counter += 1
        else:
            x = df_10_percent[col].iloc[5]
            y = df_10_percent[col].iloc[6]
            # Calculate the division of maximum and minimum values for the corresponding blocks
            change_col = pd.DataFrame({col: [100 * abs(x - y) / x]})

        # Append the calculated values to the divided_df data frame
        divided_df = pd.concat([divided_df, change_col], axis=1)

    print(divided_df)
    divided_df.to_csv("percentage_change_quantile_10.csv")

def filter_top_percentage():
    df = pd.read_csv("percentage_change_quantile_10.csv")
    genes = list(df.columns)


if __name__ == '__main__':
    create_percentage_change_df()
    create_pc_quantile10()
    plot_top()





