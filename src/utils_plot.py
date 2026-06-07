import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def class_colors(labels):
    labels_series = pd.Series(labels).dropna().astype(str)
    classes = sorted(labels_series.unique())
    base_palette = sns.color_palette("Set2", n_colors=len(classes))
    return {class_name: base_palette[index] for index, class_name in enumerate(classes)}


def boxplot(X, x_col, y_col, title=""):
    plot_df = X.copy()
    plot_df[x_col] = plot_df[x_col].astype(str)
    palette = class_colors(plot_df[x_col])

    plt.figure(figsize=(7, 5))
    sns.boxplot(x=x_col, y=y_col, data=plot_df, palette=palette)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()


def PCA_plot(X, y, title=""):
    """Draw a scatter plot using the first two principal components."""
    y_plot = pd.Series(y).astype(str)
    palette = class_colors(y_plot)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_plot, palette=palette)
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()


def tsne_plot(X, y, title=""):
    """Draw a scatter plot of the data transformed with t-SNE."""
    y_plot = pd.Series(y).astype(str)
    palette = class_colors(y_plot)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_plot, palette=palette)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()
