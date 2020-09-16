from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from imblearn.under_sampling import NearMiss
from imblearn.ensemble import EasyEnsembleClassifier as EEC
from imblearn.ensemble import BalancedRandomForestClassifier as BRC

SAMPLER_RNG = 114
CLASSIFIER_RNG = 514
SELECT_FEATURES = True # whether to perform feature selection based on wilcoxon rank-sum
NORMALISE = False

## For different categories to investigate. modify to ["Pre-Cls"] for 1st-stage characterization.
## modify as ["Anti-Virus", "non-AVP", "non-AMP", "All-Neg", "All-AMP"] for single-prediciton.
## modify as ["All-AMP"] for the 2nd-stage identification
cls_categories = ["Anti-Virus", "non-AVP", "non-AMP", "All-Neg", "All-AMP"]
## For models. You need to remove all imb_ensemble models while step-1 prediction training.
model_dicts = {

    "RF": {"model": RFC(random_state=CLASSIFIER_RNG), "param_grid": {"n_estimators": [50, 100, 120, 180, 200, 240]}},
    "BalancedRF": {"model": BRC(random_state=CLASSIFIER_RNG), "param_grid": {"n_estimators": [50, 100, 180]}},
}
# For different imb samplers. remove all imb_samplers while step-1 prediction training.
imb_strategies = {
    # Under-sampling
    'NearMiss_3': NearMiss(version=3),
    'Default': None
}
