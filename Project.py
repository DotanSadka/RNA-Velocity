# for each cell we have an initial state of its spliced RNA for a gene (s(t) = s0 + vt s.t. t = 0 => s = s0)
# plot embedding - PCA
# calculate all gammas for each gene using knn pooling (gamma = u/s) (seems like gammas are calculated from s0 and u0)
# calculate v for each gene according ds/dt = v = u(t) - gamma*s(t)
# for each cell we calculate the new s(t) and then get the trajectory

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv

METRIC = None
K = 0
PERCENTAGE = 0.1


def save_to_csv(file_name, header, row_names, values):
    df = pd.DataFrame(values, columns=header, index=row_names)

    # Save the data frame to CSV
    df.to_csv(file_name, index=True)


class CellsDict(dict):
    """
    Dict of cell type as keys and list of cells of that cell type as values
    """

    def __init__(self, df):
        super(CellsDict, self).__init__()
        self.df = df
        self.number_of_cells = self.df.shape[0]
        spliced_df = pd.read_csv("counts_0_N.csv")
        unspliced_df = pd.read_csv("counts_1.csv")
        self.neighbors_df = pd.read_csv('neighbors.csv')

        self.genes = []

        self.S = self.initiate_mRNA_expression(spliced_df, True)
        self.U = self.initiate_mRNA_expression(unspliced_df)

    def initiate_mRNA_expression(self, df, is_first=False):
        """
        Initiate all expression values of all genes in all cells.
        :param df: dataframe which contains expression values
        :return: dict that holds all cells' names as keys and each cell initiate expression level of all its genes
                 as values, such that values are a dict of a gene name - key and it's expression val - value.
        """
        cells_name = list(df["cell_name"])
        genes_name = list(df.columns)[1:]
        values = df.values

        expression_dict = {}
        for i in range(self.number_of_cells):
            expression_dict.update({cells_name[i]: {gene: value for gene, value in zip(genes_name, values[i, 1:])}})

        if is_first:
            self.init_genes(genes_name)

        return expression_dict

    def init_genes(self, genes_name):
        for name in genes_name:
            self.genes.append(Gene(name))

    def set_dict(self):
        """
        Set the cells to be clustered according to their cell type
        """
        global METRIC

        cells_name = list(self.df['Row.names'])
        cells_type = list(self.df["labels_c"])

        self.cell_types = list(set(cells_type))

        if METRIC == "celltype":
            for i in range(self.number_of_cells):
                cell_name = cells_name[i]
                # self.update(
                #     {cells_type[i]: Cell(cell_name, self.S[cell_name], self.U[cell_name], cells_type[i], METRIC)})
                self.setdefault(cells_type[i], []).append(
                    Cell(cell_name, self.S[cell_name], self.U[cell_name], cells_type[i], METRIC))
        else:
            pc1 = self.df["pc1"]
            pc2 = self.df["pc2"]
            pc3 = self.df["pc3"]
            for i in range(self.number_of_cells):
                cell_name = cells_name[i]
                self.update({cells_name[i]: Cell(cells_name[i], self.S[cell_name], self.U[cell_name],
                                                 cells_type[i], METRIC, pca=np.array([pc1[i], pc2[i], pc3[i]]))})

    def save_distances_to_csv(self):
        cells_neighbors = {}
        for current_cell in self.values():
            distances = []
            for i, cell in enumerate(self.values()):
                # Calculate Euclidean distance to all points in PCA space
                distance = np.linalg.norm(current_cell.pca - cell.pca, axis=0)
                distances.append((distance, cell))

            # sort the list based on the first value in each tuple
            sorted_list = sorted(distances, key=lambda x: x[0])

            # get all names in the order that is correlated to their distance
            sorted_cells_names = [x[1].name for x in sorted_list]
            cells_neighbors.update({current_cell.name: sorted_cells_names})

        df = pd.DataFrame(cells_neighbors)
        df.to_csv('neighbors.csv', index=False)


class Gene:
    """
    Gene class.
    Each gene has 2 dict of all the spliced and unspliced RNA expression (as value) in a current cell (as key)
    """

    def __init__(self, name):
        self.name = name
        self.gamma = None
        self.cells_coordinates = {}

    def set_gamma(self, slope):
        self.gamma = slope

    def get_gamma(self):
        return self.gamma

    def set_coordinates(self, cell_name, coordinates):
        """
        Set the cell's coordinates for that specific gene - spliced is x-axis and unspliced is y-axis
        :param cell_name: the cell that gene coordinates were calculated for
        :param coordinates: the coordinates from the s_knn and u_knn of the cell
        """
        self.cells_coordinates.update({cell_name: coordinates})


class Cell:
    """
    Cell class.
    Each cell has a dict of all its genes such that gene name is the key and the value is a tuple of gene expression (s, u)
    """

    def __init__(self, name, S0, U0, cell_type, metric, pca=None):
        """
        :param name: The cell name.
        :param S0: A dict contains cell's initial amount of spliced mRNA (values) for each gene (gene objects - keys)
        :param s0_norm: normalized s0 dict - divide s0 by all spliced count of current cell - use for velocity calculation
        :param total_S_counts: The amount of Spliced counts of current cell
        :param s_knn: A dict like S0, but normalized according to K nearest neighbors (divide by sum of neighbors spliced counts) - use for gamma calculation
        :param U0: A dict contains cell's initial amount of spliced mRNA (values) for each gene (gene objects - keys)
        :param u0_norm: normalized u0 dict - divide u0 by all unspliced count of current cell - use for velocity calculation
        :param total_U_counts: The amount of Unspliced counts of current cell
        :param u_knn: A dict like U0, but normalized according to K nearest neighbors (divide by sum of neighbors unspliced counts) - use for gamma calculation
        :param cell_type: What cluster the cell belongs to.
        :param neighbors: List of all cell's neighbors (according to the definition of neighbors that is used such as
                                                        celltype, PCA embedding, etc)
        :param pca: vector (pc1, pc2, pc3) of current cell.
        """
        self.name = name
        self.S0 = S0
        self.total_S_counts = self.get_total_counts(self.S0)
        self.s0_norm = self.normalization(self.S0, self.total_S_counts)
        self.s_knn = {}
        self.U0 = U0
        self.total_U_counts = self.get_total_counts(self.U0)
        self.u0_norm = self.normalization(self.U0, self.total_U_counts)
        self.u_knn = {}
        self.cell_type = cell_type
        self.neighbors = None
        self.metric_for_neighbors = metric
        self.pca = pca

    def get_st(self, t):
        """
        :param t: The time we want to predict the S in the cell.
        :return: A vector with all the spliced amount of each gene of the cell in time t.
        """
        return self.s0_norm + self.get_v_calculation() * t

    def get_v_calculation(self):
        """
        Calculate V according to V=ds/dt which is the constant u - gamma * s - o.
        :return: vector size 297 which contains all the velocities for each gene in cell.
        """
        return self.u0_norm - self.gamma * self.s0_norm

    def initialize_neighbors(self, cells):
        """
        According to method that is used to calculate the neighbors - initialize them for the cell.
        """
        if self.metric_for_neighbors == "PCA":
            self.neighbors = self.get_PCA_neighbors(cells)

        elif self.metric_for_neighbors == "celltype":
            self.neighbors = self.get_celltype_neighbors(cells)

    def get_total_counts(self, expression_dict):
        """
        Calculate the total count of mRNA in current cell
        :param expression_dict: S0 or U0
        :return: the total count
        """
        total = 0
        for x in expression_dict.values():
            total += x

        return total

    def normalization(self, expression_dict, total_counts):
        """
        normalize S0 and U0 for knn pooling
        :param expression_dict: S0 or U0
        :param total_counts: counts of spliced or unspliced.
        :return: s0_norm or u0_norm
        """
        dict = {gene: expression / total_counts for gene, expression in expression_dict.items()}
        return dict

    def substitute_S_by_knn(self):
        """
        Calculate new S0 according to cell's neighbors
        :return: the new S0 according to the knn pooling
        """
        sum_vec = 0  # holds the sum of all neighbors spliced counts for each gene
        total_neighbor_S_counts = 0  # holds the total splice count of all neighbors
        for neighbor in self.neighbors:
            sum_vec += np.array(list(neighbor.S0.values()))
            total_neighbor_S_counts += neighbor.total_S_counts

        # sum of vector is 1 - normalization for the vector
        sum_vec = sum_vec / total_neighbor_S_counts

        dict = {gene: sum_vec[i] for i, gene in enumerate(self.S0.keys())}
        self.s_knn = dict

    def substitute_U_by_knn(self):
        """
        Calculate new U0 according to cell's neighbors
        :return: the new U0 according to the knn pooling
        """
        sum_vec = 0  # holds the sum of all neighbors unspliced counts for each gene
        total_neighbor_U_counts = 0  # holds the total ubsplice count of all neighbors
        for neighbor in self.neighbors:
            sum_vec += np.array(list(neighbor.U0.values()))
            total_neighbor_U_counts += neighbor.total_U_counts

        # sum of vector is 1 - normalization for the vector
        sum_vec = sum_vec / total_neighbor_U_counts

        dict = {gene: sum_vec[i] for i, gene in enumerate(self.U0.keys())}
        self.u_knn = dict

    def get_celltype_neighbors(self, cells):
        """
        According to celltype - find k nearest neighbors.
        :param cells: data structure which holds all cells
        :return: k neighbors of current cell
        """
        return cells[self.cell_type]

    def get_PCA_neighbors(self, cells):
        """
        According to PCA calculation - find k nearest neighbors using euclidean distance metric.
        :param cells: data structure which holds all cells
        :return: k neighbors of current cell
        """
        global K
        names = list(cells.neighbors_df[str(self.name)].head(K))
        neighbors = []
        for i in range(K):
            neighbors.append(cells[names[i]])

        return neighbors




def check_if_df_work():
    head = ["NOGA", "TAL"]
    value = [[10, 20], [30, 40]]
    row = ["A", "B"]

    save_to_csv("CHECK.csv", head, row, value)
    return 0


def main():
    global METRIC, K, PERCENTAGE
    METRIC = "PCA"
    k = [10, 20, 30, 50, 70]
    gammas = []
    gammas_lists = []
    # for each gene find its gamma and plot each cell in 2D regarding that gene expression
    colors = {'T-cell_CD8A': "mediumblue",
              'T-cell_CD3D': "blue",
              'T-cell_CD3E': "lightblue",
              'T-cell_CD4': "grey",
              'Tumor_PGR': "hotpink",
              'Tumor_EGFR': "pink",
              'Tumor_ALDH1A3': "deeppink",
              'Tumor_CD44': "maroon",
              'Tumor_EPCAM': "lightpink",
              'Tumor_CD24': "darksalmon",
              'Un': "lightseagreen",
              'B-cell_IGHG4': "orange",
              'Fibroblast_SULF1': "yellow"
              }
    for i in k:
        global K
        K = i
        cells = CellsDict(pd.read_csv("cell_pc_df.csv"))
        cells.set_dict()

        # todo: if first time running code - run this to calculate neighbors of each cell
        # cells.save_distances_to_csv()
        # return

        # celltype
        if METRIC == "celltype":
            for cell_list in cells.values():
                for cell in cell_list:
                    cell.initialize_neighbors(cells)
                    cell.substitute_S_by_knn()
                    cell.substitute_U_by_knn()
                    for j, (x, y) in enumerate(zip(cell.s_knn.values(), cell.u_knn.values())):
                        cells.genes[j].set_coordinates(cell.name, (x, y))
        # pca
        else:
            for cell in cells.values():

                # initialize all neighbors and calculate new s_knn and u_knn accordinglly
                cell.initialize_neighbors(cells)
                cell.substitute_S_by_knn()
                cell.substitute_U_by_knn()

                # set the knn_normalization (the coordinates) of each gene for each cell
                for j, (x, y) in enumerate(zip(cell.s_knn.values(), cell.u_knn.values())):
                    cells.genes[j].set_coordinates(cell, (x, y))

        # the creation of the plot

        c = [colors[cell.cell_type] for cell in cells.values()]

        gene = [x for x in cells.genes if x.name == 'KRT19'][0]
        # Separate the x and y values from the points list
        for w in cells.genes:
            points = gene.cells_coordinates.values()  # get all (x=s_knn, y=u_knn) of all cells
            x_values = [point[0] for point in points]  # get all x values from points
            y_values = [point[1] for point in points]  # get all y values from points

            # """
            # FILTRATION RULE
            # """
            # if max(x_values) < 0.001 or max(y_values) < 0.005:
            #     plt.clf()
            #     continue

            # Create the scatter plot
            plt.scatter(x_values, y_values, c=c)
            plt.xlabel('Spliced')
            plt.ylabel('Unspliced')
            plt.title("{}\nK = {}".format(gene.name, K))

            # Create the legend
            legend_handles = []
            for cell_type, color in colors.items():
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w', label=cell_type, markerfacecolor=color, markersize=10))

            # Add legend to the plot
            plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize="small")

            # Adjust plot to make space for the legend
            plt.subplots_adjust(right=0.7)

            ## Find gamma and plot it ##

            # Find the extreme points on the top right and bottom left
            max_x, max_y = max(x_values), max(y_values)
            min_x, min_y = 0, 0

            # Get the number of points for specific percentage of the total number of points
            num_points = len(points)  # supposed to be 1311
            num_points_percent = int(num_points * PERCENTAGE)

            # Sort the points by distance from the top right point
            sorted_points = sorted(points, key=lambda point: ((point[0] - max_x) ** 2 + (point[1] - max_y) ** 2) ** 0.5)

            # Get the top percentage of the sorted points
            top_points = sorted_points[:num_points_percent]

            # Sort the points by distance from the bottom left point
            sorted_points = sorted(points, key=lambda point: ((point[0] - min_x) ** 2 + (point[1] - min_y) ** 2) ** 0.5)

            # Get the bottom 5% of the sorted points
            bottom_points = sorted_points[:num_points_percent]

            points = top_points + bottom_points
            x_values = np.array([point[0] for point in points]).reshape(-1, 1)  # get all x values from points
            y_values = np.array([point[1] for point in points])  # get all y values from points

            # calculate the slope for the gamma
            reg = LinearRegression(fit_intercept=False)
            reg.fit(x_values, y_values)

            slope = reg.coef_

            # todo: checked and coef_ equals the mathematical equation
            # sum(x_values.T.dot(y_values)) / sum(x_values.T.dot(x_values)[0])

            # Create an array of x-values for the regression line
            x_regression = np.linspace(0, np.max(x_values), len(points))

            # Calculate the corresponding y-values for the regression line starting from the origin
            y_regression = slope * x_regression

            # Plot the regression line
            plt.plot(x_regression, y_regression, color='red', linewidth=3)

            # Show the plot
            # plt.show()

            plt.savefig(f'gamma_plots/k={K}/{gene.name}.png')
            plt.clf()

            # calculate gamma
            gene.set_gamma(slope[0])
            gammas.append(slope[0])

        gammas_lists.append(gammas)
        gammas = []

    # genes = [g.name for g in cells.genes]
    # # x-axis: k value y-axis: gamma value
    # k = [10, 20, 30, 50, 70, 100, 200]
    # gamma_y_value = []
    # for i in range(297):
    #     for list in gammas_lists:
    #         gamma_y_value.append(list[i])
    #
    #     plt.subplots_adjust()
    #     plt.plot(k, gamma_y_value, 'o')
    #     plt.plot(k, gamma_y_value, '-')
    #     plt.xlabel('K values')
    #     plt.ylabel('Gamma values')
    #     plt.title("K influence gamma values in gene: {}\n".format(genes[i]))
    #     plt.savefig(f'{genes[i]}.png')
    #     plt.clf()
    #     gamma_y_value = []

    # check if gammas slightly change
    # save_to_csv(f'quantiles_{PERCENTAGE}.csv', header=genes, row_names=k, values=gammas_lists)


if __name__ == "__main__":
    main()
