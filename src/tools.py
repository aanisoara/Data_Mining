import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.impute import KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score

sns.set_theme(style="darkgrid")

################################################## OUTLIERS 1 ################################################
def plot_boxplots_outliers(
    dataframe: pd.DataFrame, colonnes: list = [], title: str = "Analyse des outliers"
) -> None:
    """
    Trace une boîte à moustache des variables d'un dataset donné.

    Args:
        dataframe (pd.DataFrame): Le dataset à analyser.
        colonnes (list): La liste des colonnes sur lesquels filtrer le dataset.
        title (str, optional): Le titre du graphique. Defaults to "Analyse des outliers".
    """
    plt.figure(figsize=(15, 5))
    plt.title(title, fontsize=13, fontweight="bold")
    if colonnes:
        sns.boxplot(data=dataframe.loc[:, colonnes])
    else:
        sns.boxplot(data=dataframe)
    plt.xticks(rotation=90)
    plt.show()



def remove_outliers(
    dataframe: pd.DataFrame, colonnes: list, seuil: float = 0.9
) -> pd.DataFrame:
    """
    Enlève les outliers d'un dataset donné. Les outliers sont définis comme l'ensemble des valeurs supérieurs à un quantile seuil.

    Args:
        dataframe (pd.DataFrame): Le dataframe à traiter.
        colonnes (list): La liste des colonnes à traiter.
        seuil (float, optional): Le quantile maximum accepté, tout se qui se trouve en dessus est passé en NaN. Defaults to 0.9.

    Returns:
        pd.DataFrame: _description_
    """
    copied_data = dataframe.copy()
    plot_boxplots_outliers(
        dataframe=copied_data,
        colonnes=colonnes,
        title="Analyse des outliers - avant traitement",
    )

    for colonne in colonnes:
        indexes = copied_data[
            copied_data[colonne] > copied_data[colonne].quantile(seuil)
        ].index
        copied_data.loc[indexes, colonne] = np.nan

    plot_boxplots_outliers(
        copied_data,
        colonnes=colonnes,
        title="Analyse des outliers - après traitement",
    )

    return copied_data

####
#second methode using ecart-interquartile (not used here)

def iqr(
    dataframe: pd.DataFrame, colonnes: list, seuil: float = 0.9
) -> pd.DataFrame:
    copied_data = dataframe.copy()
    for col in copied_data.columns:
        if copied_data[col].dtype != object:
            Q1 = copied_data[col].quantile(0.25)
            Q3 = copied_data[col].quantile(0.75)
            IQR = Q3 - Q1 # ecart inter-quartile
            S = 1.5*IQR 
            LB = Q1 - S
            UB = Q3 + S
            copied_data.loc[copied_data[col] > UB,col] = UB
            copied_data.loc[copied_data[col] < LB,col] = LB
        else:
            break
    
    for colonne in colonnes:
        indexes = copied_data[
            copied_data[colonne] > copied_data[colonne].quantile(seuil)
        ].index
        copied_data.loc[indexes, colonne] = np.nan

    plot_boxplots_outliers(
        copied_data,
        colonnes=colonnes,
        title="Analyse des outliers - après traitement",
    )
    return copied_data
################################################## TRAITEMENT DES NAN ################################################


def visualize_nan(
    dataframe: pd.DataFrame,
    title: str = "Heatmap représentative des valeurs manquantes.",
) -> None:
    """
    Affiche une heatmap montrant les valeurs manquantes sur un dataset donné.

    Args:
        dataframe (pd.DataFrame): _description_
        title (str): _description_
    """
    sns.set(font_scale=1)
    cmap = sns.cubehelix_palette(
        start=1.0, rot=-0.3, light=1, dark=0.05, gamma=0.4, hue=0.8, n_colors=2
    )
    grid_kws = {"width_ratios": (0.9, 0.03), "wspace": 0.1}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(16, 8))
    print("Shape :", dataframe.shape)

    sns.heatmap(
        dataframe.isna(),
        ax=ax,
        yticklabels=False,
        cbar_ax=cbar_ax,
        cmap=ListedColormap(cmap),
        cbar_kws={"orientation": "vertical"},
    )

    cbar_ax.set_yticklabels(["Présent", "Manquant"], fontsize=12)
    cbar_ax.yaxis.set_ticks([0.25, 0.75])

    # set title, x and y labels
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("Lignes", fontsize=14)
    ax.set_xlabel("Colonnes", fontsize=14)
    _ = plt.show()


################################################## CLUSTERING ################################################


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Affiche une ellipse illustrative de la covariance des données d'entrée.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Données d'entrée.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


################################################## ALGORITHMES ITERATIFS POUR L'OPTIMISATION DES MODELES ################################################


def find_best_knn(
    dataset: pd.DataFrame, columns: list, nb_iter: int = 100, plot_results: bool = True
):
    """
    /!\ Algorithme glouton.
    Calcule le nombre optimal de voisins pour les colonnes d'un dataset donnée.
    Il remplace des index aléatoires et essayes de les estimer par le KNNImputer.
    Le choix considéré comme optimal est le choix qui minimise l'erreur entre la réalité et l'estimation.

    Args:
        dataset (pd.DataFrame): Le dataset à tester.
        columns (list): Les colonnes spécifiques à tester.
        plot_results (bool, optional): Option pour afficher ou non les résultats. Defaults to True.

    Returns:
        int: Le nombre de voisins optimal
    """
    indexes = np.random.choice(
        dataset.loc[:, columns].dropna(axis=0).index, size=nb_iter
    )
    copied_df = dataset.copy()
    errors_dict = dict()
    for neighbors in np.arange(1, 6):
        print(f" {neighbors} voisins ".center(100, "="))
        absolute_error = list()
        for col in columns:
            copied_df = dataset.copy()
            copied_df.loc[indexes, col] = np.nan
            imputer = KNNImputer(n_neighbors=neighbors)
            imputed_df = pd.DataFrame(
                imputer.fit_transform(copied_df.select_dtypes(include=np.number)),
                columns=copied_df.select_dtypes(include=np.number).columns,
            )
            absolute_error.append(
                np.abs(imputed_df.loc[indexes, col] - dataset.loc[indexes, col])
            )
        errors_dict[neighbors] = np.mean(absolute_error)

    if plot_results:
        plt.figure(figsize=(15, 5))
        plt.title(
            "Erreur moyenne pour différents voisins proches considérés pour l'imputation",
            fontsize=13,
            fontweight="bold",
        )
        plt.plot(
            list(errors_dict.values()),
            marker="o",
            color="black",
            markeredgewidth=2,
            markersize=10,
            markeredgecolor="white",
        )
        plt.xticks(ticks=np.arange(len(errors_dict)), labels=errors_dict.keys())
        plt.ylabel("MAE des variables imputées")
        plt.xlabel("Nombre de voisins proches considérés")
        plt.show()

    return {min(errors_dict, key=errors_dict.get)}


def find_best_k_gmm(
    dataset: pd.DataFrame,
    columns: list,
    neighbors_min: int = 2,
    neighbors_max: int = 16,
    plot_results: bool = True,
):
    """
    /!\ Algorithme glouton.
    Trouve le meilleur nombre de clusters pour un dataset filtré sur des colonnes donnés.
    Effectue le choix par le critère du "Bayesian Information Criterion" (BIC).

    Args:
        dataset (pd.DataFrame): Le dataset à analyser.
        columns (list): Les colonnes du dataset à analyser.
        neighbors_min (int, optional): Le nombre de voisins minimum (> 1). Defaults to 2.
        neighbors_max (int, optional): Le nombre de voisins maximum. Defaults to 16.
        plot_results (bool, optional): Option pour afficher ou non les résultats. Defaults to True.

    Returns:
        int: Le nombre optimal de clusters.
    """
    bic_array = np.array(
        [
            GaussianMixture(n_components=i)
            .fit(dataset.loc[:columns])
            .bic(dataset.loc[:columns])
            for i in range(neighbors_min, neighbors_max)
        ]
    )

    if plot_results:
        plt.figure(figsize=(14, 8))
        plt.title(
            "Variation du BIC par rapport au nombre de clusters considérés",
            fontsize=13,
            fontweight="bold",
        )
        plt.plot(
            bic_array,
            color="black",
            marker="o",
            markeredgewidth=2,
            markersize=10,
            markeredgecolor="white",
        )
        plt.ylabel("Bayesian Information Criterion")
        plt.xlabel("Nombre de clusters")
        plt.xticks(
            ticks=np.arange(
                neighbors_min - neighbors_min, neighbors_max - neighbors_min
            ),
            labels=np.arange(neighbors_min, neighbors_max),
        )
        plt.show()
        print(f"Nombre de clusters optimal : {np.argmin(bic_array)+2}")

    return np.argmin(bic_array) + 2


################################################## EVALUATION DU MODELE ################################################


def compute_results(y_test: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    """Calcule les résultats obtenus pour une prédiction donné

    Args:
        y_test (np.ndarray): Le jeu de test.
        pred (np.ndarray): La prédiction pour le jeu de test associé.

    Returns:
        pd.DataFrame: Le dataset des résultats.
    """
    if y_test.shape != pred.shape:
        raise ValueError(
            "Dimensions inexactes (les deux dimensions doivent être identiques)"
        )
    res = pd.DataFrame()
    res["RMSE"] = np.sqrt(np.mean(np.square(y_test - pred), axis=0))
    res["MAE"] = np.mean(np.abs(y_test - pred), axis=0)
    res["MAE/std en %"] = (
        np.mean(np.abs(y_test - pred), axis=0) / y_test.std(axis=0) * 100
    )
    res["R²"] = r2_score(y_test, pred)

    res = res.T
    res.columns = ["Industrie", "Agriculture", "Résidence"]
    return res


################################################## imputation par regression ################################################


def impute_par_regression(df,target_col,features_col):
    # creation de la variable qui sera imputé 
    df[target_col+"_reg"] = df[target_col]
    
    # Recuperation de toutes les observations ne contenant aucune valeur manquante
    mask = df[target_col].notna()
    for c in features_col:
        mask = (mask) & (df[c].notna()) 
    
    # Données d'apprentissage
    X_train = df.loc[mask,features_col].values
    y_train = df.loc[mask,target_col].values
    
    # Entrainement du modele
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    
    # Prediction sur les valeurs manquantes
    y_pred = clf.predict(df.loc[df[target_col+"_reg"].isna(),features_col].values)
    df.loc[df[target_col+"_reg"].isna(),target_col+"_reg"] = y_pred
    return df