from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import numpy as np
from typing import Literal, Mapping
from numpy.typing import ArrayLike
import time
from datetime import datetime
from stepsel.binning import get_tree_cut_points, bin_values
from pandas import DataFrame

class OptimalBinningUsingDecisionTreeRegressor():
    """
    Class for optimal binning of one variable.
    
    Steps:
    ------
    - All cp values are found using DecisionTreeRegressor.
    - If the number of cp values is higher than max_grid_length, KMeans is used to reduce the number of cp values.
        - CV is performed for n_clusters of cp values.
        - The best n_best_clusters_for_extensive_search are selected for further CV of cp values in that cluster.
        - If the number of cp values is still higher than max_grid_length, the process is repeated.
    - CV is performed for all cp values.
    - The best cp value is selected.

    Methods
    -------
    fit(X, y, w = None)
        Perform optimal binning.

    set_feature_names(feature_names)
        Set feature names of the input data X. It is used for outputs like plot of the tree and cut points.
        Feature names must be set after each fit, because the tree is reinitialized.

    plot_tree(figsize: tuple=(25,20), feature_names: ArrayLike | None = None, filled: bool=True, rounded: bool=False,
              precision: int=3, fontsize: int=14)
        Plot the final tree. 

    predict(X)
        Predict regression target for X.
    
    bin_values(X)
        Bin X into intervals using fitted optimal binning.

    TODO:
    -----
    - cut_points are collected before setting feature_names, therefore the dictionary does not contain the feature names as keys.
    """

    class Logging():
        """
        SubClass responsible for logging of temporary results.
        """
        def __init__(self) -> None:
            """
            Logs templates.
            """
            self.fit_log_template = {"cp_values": None,
                                     "cp_values_reduction": None,
                                     "cp_values_reduction_cycles": {},
                                     "cv_fit": None}
            self.fit_cycle_log_template = {"cp_values_all": None,
                                           "cp_values_all_cluster": None,
                                           "cp_values_reduced": None,
                                           "cv_results": None}
        
        def log_init(self):
            """
            Logs initialization.
            """
            self.fit_log = self.fit_log_template.copy()
        
        def log_cycle(self, cycle, cp_values_cycle, cv_cp_paths_kmeans_clusters, cp_values_cycle_reduced, cv_results):
            """
            Log one cycle of cp values reduction with KMeans.
            
            Params:
            -------
            cycle
                cycle number
            cp_values_cycle
                all the cp values entering the cycle
            cv_cp_paths_kmeans_clusters
                assigned cluster for cp values
            cp_values_cycle_reduced
                reduced cp values based on KMeans for which CV is performed
            cv_results
                CV results (output of GridSearchCV.cv_results_)
            """
            self.fit_cycle_log = self.fit_cycle_log_template.copy()
            self.fit_cycle_log["cp_values_all"] = cp_values_cycle
            self.fit_cycle_log["cp_values_all_cluster"] = cv_cp_paths_kmeans_clusters
            self.fit_cycle_log["cp_values_reduced"] = cp_values_cycle_reduced
            self.fit_cycle_log["cv_results"] = cv_results
            self.fit_log["cp_values_reduction_cycles"].update({cycle: self.fit_cycle_log})

        def log_final(self, cp_values, cv_fit, in_cycle):
            """
            Log final fit state.

            Params:
            -------
            cp_values
                all cp values of the experiment
            cv_fit
                CV fit object (GridSearchCV)
            in_cycle [True/False]
                indicator whether KMeans reduction was performed
            """
            self.fit_log["cp_values"] = cp_values
            self.fit_log["cp_values_reduction"] = in_cycle
            self.fit_log["cv_fit"] = cv_fit

        def log_cycle_final(self, cycle, cp_values_cycle, cv_cp_paths_kmeans_clusters, cp_values_cycle_reduced, cv_results, cv_fit, cp_values, in_cycle = True):
            """
            Function combining log_cycle() and log_final() for saving final log inside of KMeans loop.
            
            Params:
            -------
            cycle
                cycle number
            cp_values_cycle
                all the cp values entering the cycle
            cv_cp_paths_kmeans_clusters
                assigned cluster for cp values
            cp_values_cycle_reduced
                reduced cp values based on KMeans for which CV is performed
            cv_results
                CV results (output of GridSearchCV.cv_results_)
            cv_fit
                CV fit object (GridSearchCV)
            cp_values
                all cp values of the experiment
            in_cycle [True/False]
                indicator whether KMeans reduction was performed
            """
            self.log_cycle(cycle, cp_values_cycle, cv_cp_paths_kmeans_clusters, cp_values_cycle_reduced, cv_results)
            self.log_final(cp_values, cv_fit, in_cycle)
        

    def __init__(self,
                # tree attributes
                criterion: Literal['squared_error', 'friedman_mse', 'absolute_error', 'poisson'] = "squared_error",
                splitter: Literal['best', 'random'] = "best",
                max_depth: int | None = None,
                min_samples_split: float | int = 2,
                min_samples_leaf: float | int = 1,
                min_weight_fraction_leaf: float = 0,
                max_features: float | int | Literal['auto', 'sqrt', 'log2'] | None = None,
                max_leaf_nodes: int | None = None,
                min_impurity_decrease: float = 0,
                # CV attributes
                n_splits: int = 10,
                scoring: ArrayLike | tuple | Mapping | None = "neg_mean_squared_error",
                n_jobs: int | None = None,
                refit: str | bool = "neg_mean_squared_error",
                verbose_cv: int = 0,
                return_train_score: bool = False,
                # KMeans attributes
                n_clusters: int = 100,
                # Flow attributes
                max_grid_length: int = 100,
                n_best_clusters_for_extensive_search: int = 5,
                max_cycles_of_extensive_cluster_search: int = 5,
                verbose_flow: bool = True) -> None:
        """
        Class for optimal binning of one variable.

        DecisionTreeRegressor Params:
        -------
        criterion : {"squared_error", "friedman_mse", "absolute_error", \
                "poisson"}, default="squared_error"
            The function to measure the quality of a split. Supported criteria
            are "squared_error" for the mean squared error, which is equal to
            variance reduction as feature selection criterion and minimizes the L2
            loss using the mean of each terminal node, "friedman_mse", which uses
            mean squared error with Friedman's improvement score for potential
            splits, "absolute_error" for the mean absolute error, which minimizes
            the L1 loss using the median of each terminal node, and "poisson" which
            uses reduction in Poisson deviance to find splits.

            Mean Absolute Error (MAE) criterion.

                Poisson deviance criterion.

        splitter : {"best", "random"}, default="best"
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to choose
            the best random split.

        max_depth : int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until
            all leaves are pure or until all leaves contain less than
            min_samples_split samples.

        min_samples_split : int or float, default=2
            The minimum number of samples required to split an internal node:

            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
            `ceil(min_samples_split * n_samples)` are the minimum
            number of samples for each split.

            Added float values for fractions.

        min_samples_leaf : int or float, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at
            least ``min_samples_leaf`` training samples in each of the left and
            right branches.  This may have the effect of smoothing the model,
            especially in regression.

            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and
            `ceil(min_samples_leaf * n_samples)` are the minimum
            number of samples for each node.

            Added float values for fractions.

        min_weight_fraction_leaf : float, default=0.0
            The minimum weighted fraction of the sum total of weights (of all
            the input samples) required to be at a leaf node. Samples have
            equal weight when sample_weight is not provided.

        max_features : int, float or {"auto", "sqrt", "log2"}, default=None
            The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
            `max(1, int(max_features * n_features_in_))` features are considered at each
            split.
            - If "auto", then `max_features=n_features`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.

            Note: the search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

        max_leaf_nodes : int, default=None
            Grow a tree with ``max_leaf_nodes`` in best-first fashion.
            Best nodes are defined as relative reduction in impurity.
            If None then unlimited number of leaf nodes.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.

            The weighted impurity decrease equation is the following::

                N_t / N * (impurity - N_t_R / N_t * right_impurity
                                    - N_t_L / N_t * left_impurity)

            where ``N`` is the total number of samples, ``N_t`` is the number of
            samples at the current node, ``N_t_L`` is the number of samples in the
            left child, and ``N_t_R`` is the number of samples in the right child.

            ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
            if ``sample_weight`` is passed.

        KFold Params:
        ----------
        n_splits : int, default=10
            Number of folds. Must be at least 2.

        GridSearchCV Params:
        --------------------
        scoring : str, callable, list, tuple or dict, default="neg_mean_squared_error"
            Strategy to evaluate the performance of the cross-validated model on
            the test set.

            If `scoring` represents a single score, one can use:

            - a single string (see :ref:`scoring_parameter`);
            - a callable (see :ref:`scoring`) that returns a single value.

            If `scoring` represents multiple scores, one can use:

            - a list or tuple of unique strings;
            - a callable returning a dictionary where the keys are the metric
            names and the values are the metric scores;
            - a dictionary with metric names as keys and callables a values.

            See :ref:`multimetric_grid_search` for an example.

        n_jobs : int, default=None
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

            .. versionchanged:: v0.20
            `n_jobs` default changed from 1 to None

        refit : bool, str, or callable, default="neg_mean_squared_error"
            Refit an estimator using the best found parameters on the whole
            dataset.

            For multiple metric evaluation, this needs to be a `str` denoting the
            scorer that would be used to find the best parameters for refitting
            the estimator at the end.

            Where there are considerations other than maximum score in
            choosing a best estimator, ``refit`` can be set to a function which
            returns the selected ``best_index_`` given ``cv_results_``. In that
            case, the ``best_estimator_`` and ``best_params_`` will be set
            according to the returned ``best_index_`` while the ``best_score_``
            attribute will not be available.

            The refitted estimator is made available at the ``best_estimator_``
            attribute and permits using ``predict`` directly on this
            ``GridSearchCV`` instance.

            Also for multiple metric evaluation, the attributes ``best_index_``,
            ``best_score_`` and ``best_params_`` will only be available if
            ``refit`` is set and all of them will be determined w.r.t this specific
            scorer.

            See ``scoring`` parameter to know more about multiple metric
            evaluation.

            See :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py`
            to see how to design a custom selection strategy using a callable
            via `refit`.

        verbose_cv : int
            Controls the verbosity: the higher, the more messages.

            - >1 : the computation time for each fold and parameter candidate is
            displayed;
            - >2 : the score is also displayed;
            - >3 : the fold and candidate parameter indexes are also displayed
            together with the starting time of the computation.

        return_train_score : bool, default=False
            If ``False``, the ``cv_results_`` attribute will not include training
            scores.
            Computing training scores is used to get insights on how different
            parameter settings impact the overfitting/underfitting trade-off.
            However computing the scores on the training set can be computationally
            expensive and is not strictly required to select the parameters that
            yield the best generalization performance.

        KMeans Params:
        --------------
        n_clusters : int, default=100
            The number of clusters to form as well as the number of
            centroids to generate.

        Flow Params:
        ------------
        max_grid_length: Int, default=100
            The maximum number of grid points to search over. If the number of grid points
            is greater than this value, then KMeans is used to cluster the grid points
            to reduce the number of grid points to n_clusters.

        n_best_clusters_for_extensive_search: Int, default=5
            The number of clusters to use for the extensive search. The extensive search
            of cp values in n_best_clusters_for_extensive_search of kmeans clusters
            is performed

        max_cycles_of_extensive_cluster_search: Int, default=5
            The maximum number of times to perform an extensive search over the grid points
            if the number of grid points is greater than max_grid_length. If the number of
            grid points is greater than max_grid_length, then KMeans is used to cluster the
        
        verbose_flow: bool, default=True
            If True, then print the progress of the flow search and estimated time to completion.

        Examples
        --------
        #### Poisson (frequency model)
        >>> binning = OptimalBinningUsingDecisionTreeRegressor(criterion = "poisson",
                                                               scoring = "neg_mean_poisson_deviance",
                                                               refit = "neg_mean_poisson_deviance",
                                                               max_depth = 3)
        >>> binning.fit(X, y, w)
        >>> binning.cut_points
        >>> binning.plot_tree()

        #### Gamma (severity model)
        >>> binning = OptimalBinningUsingDecisionTreeRegressor(criterion = "squared_error",
                                                               scoring = "neg_mean_gamma_deviance",
                                                               refit = "neg_mean_gamma_deviance",
                                                               max_depth = 3)
        >>> binning.fit(X, y, w)
        >>> binning.cut_points
        >>> binning.plot_tree()
        """
        # tree attributes
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features= max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        # cv attributes
        self.n_splits = n_splits
        self.scoring = scoring
        self.refit = refit
        self.n_jobs = n_jobs
        self.return_train_score = return_train_score
        self.verbose_cv = verbose_cv
        # kmeans attributes
        self.n_clusters = n_clusters
        # flow attributes
        self.max_grid_length = max_grid_length
        self.n_best_clusters_for_extensive_search = n_best_clusters_for_extensive_search
        self.max_cycles_of_extensive_cluster_search = max_cycles_of_extensive_cluster_search
        self.verbose_flow = verbose_flow
        # other attributes (classess, objects, etc.)
        self.cv_folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=666)
        self.tree_model = DecisionTreeRegressor(criterion = self.criterion,
                                                splitter = self.splitter,
                                                max_depth = self.max_depth,
                                                min_samples_split = self.min_samples_split,
                                                min_samples_leaf = self.min_samples_leaf,
                                                min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                                                max_features= self.max_features,
                                                max_leaf_nodes = self.max_leaf_nodes,
                                                min_impurity_decrease = self.min_impurity_decrease,
                                                random_state = 666)
        self.feature_names = None
        # logs and results
        self.Log = self.Logging()
        self.cut_points = None
        self.final_model = None

    def _check_feature_names(self, X):
        """Check if the feature names are present in the input data.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data.
        
        Returns
        -------
        None
            Saves the feature names in the ``feature_names`` attribute.
        """
        # If the input data is a pandas DataFrame, then the feature names are
        # taken from the columns names.
        if isinstance(X, DataFrame):
            self.feature_names = X.columns.tolist()

        # If the input data is not a pandas DataFrame, then the feature names
        # are set to default None.
        else:
            self.feature_names = None

    def set_feature_names(self, feature_names):
        """Set the feature names. This method is useful when the input data X is a numpy array.

        Parameters
        ----------
        feature_names : list
            The feature names.
        
        Returns
        -------
        None
            Saves the feature names in the ``feature_names`` attribute.
        """
        if not isinstance(feature_names, list):
            raise TypeError("feature_names must be a list")
        self.feature_names = feature_names


    @staticmethod
    def _calculate_available_cp_params(tree_model: DecisionTreeRegressor, cv_folds: KFold, X, y, w):
        """Collects the available cp parameters from the cross-validation folds.

        Parameters
        ----------
        tree_model : DecisionTreeRegressor
            The decision tree regressor model.
        
        cv_folds : KFold
            The cross-validation folds.
        
        X : numpy.ndarray
            The input data.
        
        y : numpy.ndarray
            The target data.
        
        w : numpy.ndarray
            The sample weights.
        
        Returns
        -------
        cv_cp_paths : numpy.ndarray
            The available cp parameters.       
        """

        cv_cp_paths = np.empty(shape=(0,), dtype=np.float64)
        for train_index, test_index in cv_folds.split(X):
            cp_path = tree_model.cost_complexity_pruning_path(X[train_index], y[train_index], w[train_index] if w is not None else None)
            cv_cp_paths = np.append(cv_cp_paths, cp_path["ccp_alphas"])
        # Drop duplicates
        cv_cp_paths = np.unique(cv_cp_paths)
        return cv_cp_paths

    @staticmethod
    def _kmeans_cv_cp_paths_reduction(cv_cp_paths: ArrayLike, n_clusters: int):
        """Reduces the number of cp parameters for grid search using KMeans.

        Parameters
        ----------
        cv_cp_paths : numpy.ndarray
            The available cp parameters to reduce.
        
        n_clusters : int
            The number of clusters to use for the KMeans algorithm (maximum number of cp parameters to search at once).
        
        Returns
        -------
        cv_cp_paths_reduced : numpy.ndarray
            The reduced cp parameters.
        
        cv_cp_paths_kmeans_clusters : numpy.ndarray
            The cluster labels for each cp parameter.
        """
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=666)
        kmeans.fit(cv_cp_paths.reshape(-1, 1))
        labels = kmeans.labels_
        cluster_centroids = kmeans.cluster_centers_.astype(float)[:,0]
        #   # plotting the results:
        #   u_labels = np.unique(labels)
        #   for i in u_labels:
        #       plt.scatter(cv_cp_paths[labels == i][1:] , np.ones_like(cv_cp_paths)[labels == i][1:], label = i)
        #   plt.scatter(cluster_centroids , np.ones_like(cluster_centroids) , s = 10, color = 'k')
        #   plt.legend()
        #   plt.show()
        cv_cp_paths_reduced = cluster_centroids
        cv_cp_paths_kmeans_clusters = labels
        return cv_cp_paths_reduced, cv_cp_paths_kmeans_clusters

    def _x_validation(self, cv_cp_paths: ArrayLike, X, y, sample_weight=None):
        """Performs cross-validation for the given cp parameters.

        Parameters
        ----------
        cv_cp_paths : numpy.ndarray
            The cp parameters to use for cross-validation.
        
        X : numpy.ndarray
            The input data.
        
        y : numpy.ndarray
            The target data.
        
        sample_weight : numpy.ndarray, optional
            The sample weights.
        
        Returns
        -------
        gs : GridSearchCV
            The grid search object.
        """
        gs = GridSearchCV(
            estimator = self.tree_model,
            param_grid = {"ccp_alpha": cv_cp_paths},
            cv = self.cv_folds.split(X),
            scoring = self.scoring,
            refit = self.refit,
            n_jobs = self.n_jobs,
            return_train_score = self.return_train_score,
            verbose = self.verbose_cv
        )
        gs.fit(X, y, sample_weight = sample_weight)
        return gs

    @staticmethod
    def _get_cp_params_for_best_xval_clusters(gs, cv_cp_paths, cv_cp_paths_kmeans_clusters, n_best_clusters_for_extensive_search):
        """Get cp parameters for the best clusters from cross-validation.

        Parameters
        ----------
        gs : GridSearchCV object
            Fitted GridSearchCV object.
        
        cv_cp_paths : numpy.ndarray
            The available cp parameters.
        
        cv_cp_paths_kmeans_clusters : numpy.ndarray
            The cluster labels for each cp parameter.
        
        n_best_clusters_for_extensive_search : int
            The number of best clusters to use for extensive search.
        
        Returns
        -------
        best_cp_params : numpy.ndarray
            The cp parameters for the best clusters.
        """
        results = gs.cv_results_
        test_scores = results["mean_test_score"]
        cluster_for_extensive_search = np.argsort(-test_scores)[:n_best_clusters_for_extensive_search] # *(-1) to get descending order
        best_cp_params = cv_cp_paths[np.isin(cv_cp_paths_kmeans_clusters, cluster_for_extensive_search)]
        return best_cp_params

    def _collect_results(self, gs):
        """Collects the results from the grid search.

        Parameters
        ----------
        gs : GridSearchCV object
            Fitted GridSearchCV object.
        
        Returns
        -------
        None
            Saves results as properties of the class.
        """

        # Cut points for features
        self.cut_points = get_tree_cut_points(gs.best_estimator_, self.feature_names)
        self.final_model = gs.best_estimator_

    def plot_tree(self, figsize: tuple=(25,20), filled: bool=True, rounded: bool=False,
                  precision: int=3, fontsize: int=14):
        """Plot the decision tree.

        Parameters
        ----------
        figsize : tuple of int, default=(25,20)
            The size of the figure to create in matplotlib.
        feature_names : list of str, default=None
            The names of the features.
        filled : bool, default=True
            When set to True, paint nodes to indicate majority class for classification, extremity of values for regression, or purity of node for multi-output.
        rounded : bool, default=False
            When set to True, draw node boxes with rounded corners and use Helvetica fonts instead of Times-Roman.
        precision : int, default=3
            The precision for displaying split thresholds and other float values.
        fontsize : int, default=14
            The fontsize for node labels.
        
        Returns
        -------
        matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        _ = tree.plot_tree(self.final_model, feature_names=self.feature_names, filled=filled, rounded=rounded, precision=precision, fontsize=fontsize)
        plt.show()

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict regression value for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        return self.final_model.predict(np.array(X))


    def bin_values(self, data: ArrayLike) -> np.ndarray:
        """Bin values of X into intervals.

        Parameters
        ----------
        data : array-like
            The input values to be binned.
        
        Returns
        -------
        binned_values : array-like
            The binned values.
        """
        if len(self.cut_points) == 0:
            raise ValueError("The model has not been fitted yet or the cut points are not available, e.g. no optimal cut points found.")
        # Keys can be numeric ids or feature names if feature_names were provided
        dict_key = list(self.cut_points.keys())[0]
        result = bin_values(data, self.cut_points[dict_key])
        return result


    def fit(self, X, y, w = None):
        """Fit the model and get the optimal binning.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (real numbers). Use ``dtype=np.float64`` and
            ``order='C'`` for maximum efficiency.

        w : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.
        
        Returns
        -------
        self : OptimalBinningUsingDecisionTreeRegressor
            Fitted estimator.
        """
        # Save feature names if X is a dataframe
        self._check_feature_names(X)

        # Convert to numpy arrays
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        self.Log.log_init()

        # All potential cp values
        str_time = time.time()
        cp_values = self._calculate_available_cp_params(self.tree_model, self.cv_folds, X, y, w) # one tree calculation time proxy value (for total runtime estimate)
        end_time = time.time()
        one_cp_value_calculation_time = end_time - str_time

        # If more than 100 values, reduce to 100 using kmeans clustering
        if len(cp_values) > self.max_grid_length:
            cycles_of_extensive_cluster_search = 0
            cp_values_cycle = cp_values.copy()

            while (len(cp_values_cycle) > self.max_grid_length) & (cycles_of_extensive_cluster_search < self.max_cycles_of_extensive_cluster_search):
                cp_values_cycle_reduced, cv_cp_paths_kmeans_clusters = self._kmeans_cv_cp_paths_reduction(cp_values_cycle, self.n_clusters)

                if self.verbose_flow:
                    print(f"\nToo many cp values ({len(cp_values_cycle)}), reducing to {len(cp_values_cycle_reduced)} using kmeans clustering.")
                    print(f"Cycle {cycles_of_extensive_cluster_search + 1} of extensive cluster search started at {datetime.now().strftime('%H:%M:%S')}.")
                    print(f"Estimated runtime: {one_cp_value_calculation_time * len(cp_values_cycle_reduced) / 60} minutes.")

                # Perform cross-validation for reduced cp values
                gs = self._x_validation(cp_values_cycle_reduced, X, y, w)

                # Log cycle information
                self.Log.log_cycle(cycles_of_extensive_cluster_search, cp_values_cycle, cv_cp_paths_kmeans_clusters, cp_values_cycle_reduced, gs.cv_results_)
                
                # Update cp values for the next cycle
                cp_values_cycle = self._get_cp_params_for_best_xval_clusters(gs, cp_values_cycle, cv_cp_paths_kmeans_clusters, self.n_best_clusters_for_extensive_search)
                cycles_of_extensive_cluster_search += 1

            if self.verbose_flow:
                print(f"\nStarting final cross-validation for {len(cp_values_cycle)} cp values at {datetime.now().strftime('%H:%M:%S')}.")
                print(f"Estimated runtime: {one_cp_value_calculation_time * len(cp_values_cycle) / 60} minutes.")

            # Perform final cross-validation for remaining cp values
            gs = self._x_validation(cp_values_cycle, X, y, w)

            # Log final cycle information
            self.Log.log_cycle_final(cycles_of_extensive_cluster_search, cp_values_cycle, cv_cp_paths_kmeans_clusters, cp_values_cycle_reduced, gs.cv_results_, gs, cp_values)
        else:
            if self.verbose_flow:
                print(f"\nStarting cross-validation for {len(cp_values)} cp values at {datetime.now().strftime('%H:%M:%S')}.")
                print(f"Estimated runtime: {one_cp_value_calculation_time * len(cp_values) / 60} minutes.")

            # Perform cross-validation for all cp values
            gs = self._x_validation(cp_values, X, y, w)

            # Log final cycle information
            self.Log.log_final(cp_values, gs, False)
        
        # Collect results
        self._collect_results(gs)

        return self
