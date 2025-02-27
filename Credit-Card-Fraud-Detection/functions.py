from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier


class PermutationImportanceWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, *, loss='log_loss', learning_rate=0.1, max_iter=100, 
                 max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, 
                 l2_regularization=0.0, max_features=1.0, max_bins=255, 
                 categorical_features='from_dtype', 
                 monotonic_cst=None, interaction_cst=None, 
                 warm_start=False, 
                 early_stopping='auto', scoring='loss', 
                 validation_fraction=0.1, n_iter_no_change=10, 
                 tol=1e-07, verbose=0, random_state=None, class_weight=None):
        # Initialize the parameters for HistGradientBoostingClassifier
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.max_bins = max_bins
        self.categorical_features = categorical_features
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.class_weight = class_weight

        # Initialize the estimator with the parameters passed to __init__
        self.estimator = HistGradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_features=self.max_features,
            max_bins=self.max_bins,
            categorical_features=self.categorical_features,
            monotonic_cst=self.monotonic_cst,
            interaction_cst=self.interaction_cst,
            warm_start=self.warm_start,
            early_stopping=self.early_stopping,
            scoring=self.scoring,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            class_weight=self.class_weight
        )

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

    @property
    def feature_importances_(self):
        result = permutation_importance(self.estimator, self.X_train, self.y_train, n_repeats=5, random_state=self.random_state)
        return result.importances_mean

