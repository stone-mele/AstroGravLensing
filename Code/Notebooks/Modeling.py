import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn import warnings
from astropy.io import fits


class EbossWaveSpectraDataset():
    
    max_size = 4639
    
    def __init__(self, csv_locator):
        self.spectra_infos = pd.read_csv(csv_locator)
        
    def __len__(self):
        return len(self.spectra_infos)
    
    def _pad_flux(self, flux, max_size):
        needed_pad = max_size - len(flux)
        if needed_pad % 2 == 0:
            return np.pad(flux, (needed_pad//2, needed_pad//2), 'constant')
        else:
            return np.pad(flux, ((needed_pad//2) +1, needed_pad//2), 'constant')
        
    
    def get_ext_data(self, idx, ext_num):
        cur_spec = self.spectra_infos.iloc[idx]
        flux = fits.open(cur_spec.location)[ext_num].data
        flux = self._pad_flux(flux, max_size=4639)
        return flux, cur_spec
    
    def __getitem__(self, idx):
        cur_spec = self.spectra_infos.iloc[idx]
        flux = fits.open(cur_spec.location)[3].data
        flux = self._pad_flux(flux, max_size=4639)
        return flux, cur_spec
    
class MangaWaveSpectraDataset():
    
    max_size = 4563
    
    def __init__(self, csv_locator):
        self.spectra_infos = pd.read_csv(csv_locator)
        
    def __len__(self):
        return len(self.spectra_infos)
    
    def _pad_flux(self, flux, max_size):
        needed_pad = max_size - len(flux)
        if needed_pad % 2 == 0:
            return np.pad(flux, (needed_pad//2, needed_pad//2), 'constant')
        else:
            return np.pad(flux, ((needed_pad//2) +1, needed_pad//2), 'constant')
        
        
    def get_ext_data(self, idx, ext_num):
        cur_spec = self.spectra_infos.iloc[idx]
        flux = fits.open(cur_spec.location)[ext_num].data
        flux = self._pad_flux(flux, max_size=self.max_size)
        return flux, cur_spec
    
    def __getitem__(self, idx):
        cur_spec = self.spectra_infos.iloc[idx]
        flux = fits.open(cur_spec.location)[3].data
        flux = self._pad_flux(flux, max_size=self.max_size)
        return flux, cur_spec


def read_manga_dataset(filename, scale=True, remove_emission_locations=True):
    """
    Helper method to read in the manga data set and clean it up.
    :param filename: file location
    :param scale: If true data will be min max scaled
    :param remove_emission_locations: If true the emission lines location will be removed.
    :return: dictionary containing the full dataset, and the data separated into and X set and Y set.
    """
    full_set = pd.read_csv(filename)

    # Read in the desired columns for the x-set.
    x_cols = np.r_[3:14, 26:len(full_set.columns)-1] if remove_emission_locations else np.r_[3:len(full_set.columns)-1]
    x = full_set.iloc[:, x_cols].copy(deep=True)
    x = x.fillna('0')

    # Read in the label column and transform it to a numeric value.
    y = full_set.iloc[:, [-1]].copy(deep=True)
    y = y.Hits.map({'bad': 0, 'good': 1})

    # If scaling was enabled scaled all the x-values to be within 0 to 1.
    if scale:
        mms = MinMaxScaler()
        x = pd.DataFrame(mms.fit_transform(x), columns=x.columns)

    return {'full': full_set, 'X': x, 'Y': y}


def read_eboss_dataset(filename, scale=True):
    """
    Helper method to read in the eboss data set and clean it up.
    :param filename: File location of the data set.
    :param scale: If we want the data to be scaled.
    :return: dictionary containing the full dataset, and the data separated into an X set and Y set.
    """

    # Saving the fullset.
    full_set = pd.read_csv(filename)

    # Getting rid of the id column, and the label column for the X variables.
    x = full_set.iloc[:, 1:-1].copy(deep=True)

    # Making sure we only have numeric values.
    x = x.apply(pd.to_numeric, args={'errors': 'coerce'})

    # Any NA values will be replaced with null
    x = x.fillna('0')

    # Extracting the label and transforming the 'good', 'bad' labels into numeric values.
    y = full_set.iloc[:, -1:].copy(deep=True)
    y = y.Hits.map({'bad': 0, 'good': 1})

    # If scaling was enabled scaled all the x-values to be within 0 to 1.
    if scale:
        mms = MinMaxScaler()
        x = pd.DataFrame(mms.fit_transform(x), columns=x.columns)

    return {'full': full_set, 'X': x, 'Y': y}


class CrossValidationModeler:
    """
    This class is a simple wrapper on some sklearn functionality to make the running
    of the different models easier and simplier.
    """

    __version__ = '0.01'

    def __init__(self, x, y, model):
        """
        Sets up the variables to use when running cross validation.
        :param x:
        :param y:
        :param model:
        """
        self.x = x
        self.y = y
        self.model = model

        # initial scoring dictionary.
        self.scoring_dict = {
            'precision (no)': make_scorer(precision_score, pos_label=0),
            'precision (yes)': make_scorer(precision_score, pos_label=1),
            'recall (no)': make_scorer(recall_score, pos_label=0),
            'recall (yes)': make_scorer(recall_score, pos_label=1),
            'accuracy': make_scorer(accuracy_score),
            #'auc': make_scorer(auc),
            'roc_auc': make_scorer(roc_auc_score),
            'f1': make_scorer(f1_score)
        }

        self.last_results = None

    def run_cross_val(self, params_grid={}, n_splits=10, rando_state=None, only_relevant_metrics=True, block_warnings=False):
        """
        Runs cross-validation using the given parameter grid.
        :param params_grid:
        :param n_splits:
        :param only_relevant_metrics:
        :param block_warnings:
        :param rando_state:
        :return:
        """

        # Catch warnings if the variable is set.
        with warnings.catch_warnings():
            if block_warnings:
                warnings.simplefilter("ignore")
            # Running the cross-validation on the k-NN with 10 folds.
            grid_clf = GridSearchCV(self.model,
                                    cv=ShuffleSplit(n_splits=n_splits, test_size=.2, random_state=rando_state),
                                    param_grid=params_grid,
                                    scoring=self.scoring_dict,
                                    refit='accuracy',
                                    return_train_score=False, verbose=3).fit(self.x, self.y)

        # Get the results
        self.last_results = grid_clf.cv_results_

        # If we only want relevant metrics just return that.
        if only_relevant_metrics:
            return self.get_last_run_relevant_metrics()

        return self.last_results

    def _extract_relevant_metrics(self):
        """
        Helper method to return relevant metrics.
        :return: Dictionary containing relevant metrics.
        """
        relevant_metric = {'mean_test_{}'.format(k): self.last_results['mean_test_{}'.format(k)] for k in
                           self.scoring_dict.keys()}
        return relevant_metric

    def get_last_run_relevant_metrics(self):
        """
        Generates pandas dataframe to present the metrics in a nice manner.
        :return: pd.DataFrame
        """
        rel_df = pd.DataFrame(self._extract_relevant_metrics(), index=self.last_results['params'])
        rel_df.index.name = 'Parameters'
        return rel_df
