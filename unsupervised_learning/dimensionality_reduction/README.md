What is PCA?
    PCA stands for Principal Component Analysis.
    It's a way to reduce the number of features (dimensions) in your data
    while keeping as much important information as possible.

    Why reduce dimensions?
    Imagine you have data with many features (like height, weight, age, etc).
    Some of these features may be related or redundant.
    PCA helps you find a smaller set of features that still captures
    most of the patterns in your data.

    How does PCA work?
    Input: A dataset where each row is a data point.
    Centering: Subtract the mean from each feature (already done here).
    Covariance: Check how features change together.
    Eigenvectors: Find new directions (called components).
    Sort: Choose the directions that explain the most variance.
    Project: Keep only the directions that explain enough variance.