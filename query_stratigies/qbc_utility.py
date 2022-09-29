import numpy as np
from sklearn.utils.validation import check_array, _is_arraylike

def average_kl_divergence(probas):
    """Calculates the average Kullback-Leibler (KL) divergence for measuring
    the level of disagreement in QueryByCommittee.
    Parameters
    ----------
    probas : array-like, shape (n_estimators, n_samples, n_classes)
        The probability estimates of all estimators, samples, and classes.
    Returns
    -------
    scores: np.ndarray, shape (n_samples)
        The Kullback-Leibler (KL) divergences.
    References
    ----------
    [1] A. McCallum and K. Nigam. Employing EM in pool-based active learning
        for text classification. In Proceedings of the International Conference
        on Machine Learning (ICML), pages 359-367. Morgan Kaufmann, 1998.
    """
    # Check probabilities.
    probas = check_array(probas, allow_nd=True)
    if probas.ndim != 3:
        raise ValueError(
            f"Expected 3D array, got {probas.ndim}D array instead."
        )
    n_estimators = probas.shape[0]

    # Calculate the average KL divergence.
    probas_mean = np.mean(probas, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.nansum(
            np.nansum(probas * np.log(probas / probas_mean), axis=2), axis=0
        )
    scores = scores / n_estimators

    return scores


def vote_entropy(votes, classes):
    """Calculates the vote entropy for measuring the level of disagreement in
    QueryByCommittee.
    Parameters
    ----------
    votes : array-like, shape (n_samples, n_estimators)
        The class predicted by the estimators for each sample.
    classes : array-like, shape (n_classes)
        A list of all possible classes.
    Returns
    -------
    vote_entropy : np.ndarray, shape (n_samples)
        The vote entropy of each row in `votes`.
    References
    ----------
    [1] Engelson, Sean P., and Ido Dagan.
        Minimizing manual annotation cost in supervised training from corpora.
        arXiv preprint cmp-lg/9606030 (1996).
    """
    # Check `votes` array.
    votes = check_array(votes)
    n_estimators = votes.shape[1]

    # Count the votes.
    vote_count = compute_vote_vectors(
        y=votes, classes=classes
    )

    # Compute vote entropy.
    v = vote_count / n_estimators

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.nansum(-v * np.log(v), axis=1)
    return scores

def compute_vote_vectors(y, w=None, classes=None, missing_label=np.nan):
    """Counts number of votes per class label for each sample.
    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_annotators)
        Class labels.
    w : array-like, shape (n_samples) or (n_samples, n_annotators),
    default=np.ones_like(y)
        Class label weights.
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class.
    missing_label : scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.
    Returns
    -------
    v : array-like, shape (n_samples, n_classes)
        V[i,j] counts number of votes per class j for sample i.
    """
    # check input parameters
    n_classes = len(np.unique(classes))
    y = y if y.ndim == 2 else y.reshape((-1, 1))
    y = y.astype(int)
    if n_classes == 0:
        raise ValueError(
            "Number of classes can not be inferred. "
            "There must be at least one assigned label or classes must not be"
            "None. "
        )

    w = (
        np.ones_like(y)
        if w is None
        else check_array(
            w, ensure_2d=False, force_all_finite=False, dtype=float, copy=True
        )
    )
    w = w if w.ndim == 2 else w.reshape((-1, 1))

    y_off = y + np.arange(y.shape[0])[:, None] * n_classes
    v = np.bincount(
        y_off.ravel(), minlength=y.shape[0] * n_classes, weights=w.ravel()
    )
    v = v.reshape(-1, n_classes)

    return v
'''
def compute_vote_vectors(y, classes=None):
    """Counts number of votes per class label for each sample.
    Parameters
    ----------
    y : array-like, shape (n_samples) or (n_samples, n_annotators)
        Class labels.
    w : array-like, shape (n_samples) or (n_samples, n_annotators),
    default=np.ones_like(y)
        Class label weights.
    classes : array-like, shape (n_classes), default=None
        Holds the label for each class.
    missing_label : scalar|string|np.nan|None, default=np.nan
        Value to represent a missing label.
    Returns
    -------
    v : array-like, shape (n_samples, n_classes)
        V[i,j] counts number of votes per class j for sample i.
    """
    # check input parameters
    n_classes = len(np.unique(classes))
    y = y if y.ndim == 2 else y.reshape((-1, 1))
    y = y.astype(int)
    if n_classes == 0:
        raise ValueError(
            "Number of classes can not be inferred. "
            "There must be at least one assigned label or classes must not be"
            "None. "
        )

    w = np.ones_like(y)
    w = w if w.ndim == 2 else w.reshape((-1, 1))

    # count class labels per class and weight by confidence scores
    y_off = y + np.arange(y.shape[0])[:, None] * n_classes
    v = np.bincount(
        y_off.ravel(), minlength=y.shape[0] * n_classes, weights=w.ravel()
    )
    v = v.reshape(-1, n_classes)

    return v
'''