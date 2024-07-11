import numpy as np
import pandas as pd
from typing import Any
import statsmodels.api as sm

from stepsel.modeling.prep import prepare_model_matrix
from stepsel.stattests import likelihood_ratio_test_models, likelihood_ratio_test

class StepwiseGLM():
    """
    Class for performing stepwise GLM.

    Methods
    -------
    fit()
        Fit stepwise GLM.    
    """

    def __init__(self,
                 formula: str, data: pd.DataFrame, include: list[str] | None = None, slentry: float = 0.001, slstay: float = 0.001,
                 model_fit_log: pd.DataFrame = None,
                 # sm.GLM arguments
                 family: Any | None = None, offset: Any | None = None, exposure: Any | None = None,
                 freq_weights: Any | None = None, var_weights: Any | None = None, missing: str = 'none',
                 **kwargs: Any) -> None:
        """ Initialize StepwiseGLM object.

        Parameters
        ----------
        formula : str
            Formula for GLM.
        data : pd.DataFrame
            Dataframe containing variables in formula.
        include : list[str] | None, optional
            List of variables to always include in the model, by default None.
            Variables in include cannot be in the formula!
        slentry : float, optional
            Significance level for entering the model, by default 0.001.
        slstay : float, optional
            Significance level for staying in the model, by default 0.001.
        model_fit_log : pd.DataFrame, optional
            Log of model fits, by default None.
            This can be used to continue a previous stepwise GLM and to avoid refitting models.
        family : Any | None, optional
            sm.GLM argument, by default None.
        offset : Any | None, optional
            sm.GLM argument, by default None.
        exposure : Any | None, optional
            sm.GLM argument, by default None.
        freq_weights : Any | None, optional
            sm.GLM argument, by default None.
        var_weights : Any | None, optional
            sm.GLM argument, by default None.
        missing : str, optional
            sm.GLM argument, by default 'none'.
        **kwargs : Any, optional
            sm.GLM argument, by default None.

        Notes
        -----
        The formula is extended with the variables in include.
        The full formula is used to create the model matrix.
        The model matrix is used to fit the model.
        The model fit is logged in model_fit_log.

        TODO
        ----
        - Check if include variables are in formula and raise error if so.
        - Check if model_fit_log follows the correct format and raise error if not.
        - Wrap _model_fit in try-except and raise warning if model fit fails.
        It will ensure that one bad fit during stepwise won't disrupt the whole process.
        - Allow calling add1 and drop1 separately. Or even better, prepare functions of it.
        """
        self.formula = formula
        self.data = data
        self.include = include
        self.slentry = slentry
        self.slstay = slstay
        self.family = family
        self.offset = offset
        self.exposure = exposure
        self.freq_weights = freq_weights
        self.var_weights = var_weights
        self.missing = missing
        self.kwargs = kwargs

        # Full formula
        self.full_formula = formula + " + " + " + ".join(include) if include is not None else formula

        # Prepare model matrix
        self.y, self.X, self.feature_ids = prepare_model_matrix(self.full_formula, data)
        self.feature_ids = pd.Series(self.feature_ids)

        # Model fits
        self.model_fit_log = pd.DataFrame({}, columns=["llf", "df"]) if model_fit_log is None else model_fit_log
        self.current_model = None
        self.current_model_features = None

    @staticmethod
    def _get_model_string(features: pd.Series) -> str:
        """ Get model string from features.

        Parameters
        ----------
        features : pd.Series
            Features to include in model.

        Returns
        -------
        str
            Model string.
        
        Notes
        -----
        The model string is used as unique identifier for the model.
        Only the exogenous variables are included in the model string.
        The variables are sorted alphabetically.
        """
        return features.sort_values().astype(str).str.cat(sep=" + ")

    def _log_model_fit(self, features: pd.Series, fit: sm.GLM) -> None:
        """ Log model fit.

        Parameters
        ----------
        features : pd.Series
            Features to include in model.
        fit : sm.GLM
            Fitted model.

        Returns
        -------
        None

        Modifies
        --------
        self.model_fit_log : pd.DataFrame
            Log of model fits - adds row for model.

        Notes
        -----
        The model string is used as unique identifier for the model.
        """
        # Get log-likelihood, and degrees of freedom
        llf = fit.llf
        df = fit.df_model

        # Log model fit
        model_string = self._get_model_string(features)
        self.model_fit_log.loc[model_string, :] = [llf, df]


    def _model_fit(self, features: list[str]) -> None:
        """ Fit model.

        Parameters
        ----------
        features : list[str]
            Features to include in model.

        Returns
        -------
        fit : sm.GLM
            Fitted model.
        """
        columns_of_X_to_include = np.array(self.feature_ids.isin(features))
        fit = sm.GLM(endog=self.y,
                    exog=self.X.loc[:,columns_of_X_to_include],
                    family=self.family, offset=self.offset, exposure=self.exposure, freq_weights=self.freq_weights,
                    var_weights=self.var_weights, missing=self.missing, **self.kwargs)
        fit = fit.fit() # try start_params for faster convergence (?)

        self._log_model_fit(pd.Series(features), fit)

        return fit
        

    def _add1(self, scope: list[str]) -> str | None:
        """ Add one feature to the model if it improves the model fit.

        Parameters
        ----------
        scope : list[str]
            List of features to consider for adding to the model.

        Returns
        -------
        best_new_eligible_feature : str | None
            Best feature to add to the model.
            None if no feature improves the model fit.

        Modifies
        --------
        self.current_model : sm.GLM
            Fitted model if a feature is added.
        self.current_model_features : list[str]
            Features in the model if a feature is added.
        self.model_fit_log : pd.DataFrame
            Log of model fits - adds row for model.

        Notes
        -----
        The model fit is improved if the p-value of the likelihood ratio test is below slentry.
        """
        # Log model fits in scope
        model_fits = pd.DataFrame({"feature": [], "llf": [], "df": [], "lr": [], "p_value": []})

        # Keep best eligible model
        lowest_p_value = 1
        best_eligible_model = None
        best_eligible_model_features = None
        best_new_eligible_feature = None

        # Loop over all features in scope
        for feature in scope:
            features = self.current_model_features + [feature]
            model_string = self._get_model_string(pd.Series(features))
            
            # Check if model fit is already available
            if model_string in self.model_fit_log.index:
                llf = self.model_fit_log.loc[model_string,"llf"]
                df = self.model_fit_log.loc[model_string,"df"]
                fit = "not available"
                lr, p = likelihood_ratio_test(llf, self.current_model.llf, df, self.current_model.df_model)
                model_fits.loc[model_string] = [feature, llf, df, lr, p]
            
            # Fit model
            else:
                fit = self._model_fit(features)
                lr, p = likelihood_ratio_test_models(fit, self.current_model)
                model_fits.loc[model_string] = [feature, fit.llf, fit.df_model, lr, p]

            # Check if model is eligible
            if (p < self.slentry) & (p < lowest_p_value):
                best_eligible_model = fit
                lowest_p_value = p
                best_eligible_model_features = features
                best_new_eligible_feature = feature

        # Update current model
        if (best_eligible_model is not None) & (best_eligible_model != "not available"):
            self.current_model = best_eligible_model
            self.current_model_features = best_eligible_model_features
        elif (best_eligible_model is not None):
            self.current_model = self._model_fit(best_eligible_model_features)
            self.current_model_features = best_eligible_model_features
        else:
            pass

        # Print output - best candidates
        print("Best entry candidates:")
        print(model_fits.sort_values("p_value")[["feature","p_value"]].head(5).reset_index(drop=True))
        print(f">> {best_new_eligible_feature if best_new_eligible_feature is not None else 'No feature'} entered the model.")
        print("")

        return best_new_eligible_feature


    def _drop1(self, scope: list[str]) -> str | None:
        """ Drop one feature from the model if it does not worsen the model fit.

        Parameters
        ----------
        scope : list[str]
            List of features to consider for dropping from the model.

        Returns
        -------
        best_dropped_eligible_feature : str | None
            Best feature to drop from the model.
            None if no feature can be dropped without worsening the model fit.

        Modifies
        --------
        self.current_model : sm.GLM
            Fitted model if a feature is dropped.
        self.current_model_features : list[str]
            Features in the model if a feature is dropped.
        self.model_fit_log : pd.DataFrame
            Log of model fits - adds row for model.

        Notes
        -----
        The model fit is worsened if the p-value of the likelihood ratio test is above slstay.
        """
        # Log model fits in scope
        model_fits = pd.DataFrame({"feature": [], "llf": [], "df": [], "lr": [], "p_value": []})

        # Keep best eligible model
        highest_p_value = 0
        best_eligible_model = None
        best_eligible_model_features = None
        best_dropped_eligible_feature = None

        # Loop over all features in scope
        for feature in scope:
            features = self.current_model_features.copy()
            features.remove(feature)
            model_string = self._get_model_string(pd.Series(features))

            # Check if model fit is already available
            if model_string in self.model_fit_log.index:
                llf = self.model_fit_log.loc[model_string,"llf"]
                df = self.model_fit_log.loc[model_string,"df"]
                fit = "not available"
                lr, p = likelihood_ratio_test(self.current_model.llf, llf, self.current_model.df_model, df)
                model_fits.loc[model_string] = [feature, llf, df, lr, p]
            
            # Fit model
            else:
                fit = self._model_fit(features)
                lr, p = likelihood_ratio_test_models(self.current_model, fit)
                model_fits.loc[model_string] = [feature, fit.llf, fit.df_model, lr, p]

            # Check if model is eligible
            if (p > self.slstay) & (p > highest_p_value):
                best_eligible_model = fit
                highest_p_value = p
                best_eligible_model_features = features
                best_dropped_eligible_feature = feature

        # Update current model
        if (best_eligible_model is not None) & (best_eligible_model != "not available"):
            self.current_model = best_eligible_model
            self.current_model_features = best_eligible_model_features
        elif (best_eligible_model is not None):
            self.current_model = self._model_fit(best_eligible_model_features)
            self.current_model_features = best_eligible_model_features
        else:
            pass

        # Print output - best candidates
        print("Best removal candidates:")
        print(model_fits.sort_values("p_value", ascending=False)[["feature","p_value"]].head(5).reset_index(drop=True))
        print(f">> {best_dropped_eligible_feature if best_dropped_eligible_feature is not None else 'No feature'} removed from the model.")
        print("")

        return best_dropped_eligible_feature

        
    def fit(self):
        """ Fit the model using stepwise selection.

        Modifies
        --------
        self.current_model : sm.GLM
            Fitted model.
        self.current_model_features : list[str]
            Features in the model.
        self.model_fit_log : pd.DataFrame
            Log of model fits.

        Notes
        -----
        Selection is done in a forward-backward fashion.
        It starts with the intercept and features in self.include.
        Then, it adds a feature if it improves the model fit.
        Then, it drops a feature if it does not worsen the model fit.
        Adding and dropping is repeated until no feature can be added or dropped.
        """
        # Initial model features
        self.current_model_features = ["Intercept"] + self.include if self.include is not None else ["Intercept"]

        # Initial model fit
        # TODO: skip if in self.model_fit_log
        fit = self._model_fit(self.current_model_features)

        # Initial model becomes current model
        self.current_model = fit

        stepwise_running = True
        while stepwise_running:

            # Add1 available features
            scope = self.feature_ids[~self.feature_ids.isin(self.current_model_features)].drop_duplicates().tolist()
            if len(scope) > 0:
                added = self._add1(scope)

                # Drop1 available feature
                filtr = self.feature_ids.isin(self.current_model_features) & (~self.feature_ids.isin(["Intercept"] + (self.include if self.include is not None else [])))
                scope = self.feature_ids[filtr].drop_duplicates().tolist()
                dropped = self._drop1(scope)
            else:
                added = None
                dropped = None

            # Check if finished
            if added is None and dropped is None:
                stepwise_running = False
                print(f"\nSelection stopped because all candidates for removal are significant at the {self.slstay} level and no candidate for entry is significant at the {self.slentry} level.\n")
                print(self.current_model.summary())
            elif added == dropped:
                stepwise_running = False
                print(f"\nSelection stopped because adding and removing candidates is cycling.\n")
                print(self.current_model.summary())
            else:
                pass

        return self
