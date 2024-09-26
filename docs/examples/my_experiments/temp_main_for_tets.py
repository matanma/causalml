import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBRegressor
import warnings

from causalml.inference.meta import LRSRegressor
from causalml.inference.meta import XGBTRegressor, MLPTRegressor
from causalml.inference.meta import BaseXRegressor, BaseRRegressor, BaseSRegressor, BaseTRegressor
from causalml.match import NearestNeighborMatch, MatchOptimizer, create_table_one
from causalml.propensity import ElasticNetPropensityModel
from causalml.dataset import *
from causalml.metrics import *

warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


def main():
    y, X, treatment, tau, b, e = synthetic_data(mode=1, n=10000, p=8, sigma=1.0)

    # S-learner
    learner_s = LRSRegressor()
    ate_s = learner_s.estimate_ate(X=X, treatment=treatment, y=y)
    print(ate_s)
    print('ATE estimate: {:.03f}'.format(ate_s[0][0]))
    print('ATE lower bound: {:.03f}'.format(ate_s[1][0]))
    print('ATE upper bound: {:.03f}'.format(ate_s[2][0]))

    # After calling estimate_ate, add pretrain=True flag to skip training
    # This flag is applicable for other meta learner
    ate_s = learner_s.estimate_ate(X=X, treatment=treatment, y=y, pretrain=True)
    print(ate_s)
    print('ATE estimate: {:.03f}'.format(ate_s[0][0]))
    print('ATE lower bound: {:.03f}'.format(ate_s[1][0]))
    print('ATE upper bound: {:.03f}'.format(ate_s[2][0]))

    # T-learner
    # Ready-to-use T-Learner using XGB
    learner_t = XGBTRegressor()
    ate_t = learner_t.estimate_ate(X=X, treatment=treatment, y=y)
    print('Using the ready-to-use XGBTRegressor class')
    print(ate_t)


    #X-learner
    # X Learner with propensity score input
    # Calling the Base Learner class and feeding in XGB
    learner_x = BaseXRegressor(learner=XGBRegressor())
    ate_x = learner_x.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    print('Using the BaseXRegressor class and using XGB:')
    print(ate_x)

    # Calling the Base Learner class and feeding in LinearRegression
    learner_x = BaseXRegressor(learner=LinearRegression())
    ate_x = learner_x.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    print('\nUsing the BaseXRegressor class and using Linear Regression:')
    print(ate_x)

    ## Calculate ITE
    learner_s = LRSRegressor()
    cate_s = learner_s.fit_predict(X=X, treatment=treatment, y=y)
    print(cate_s)

    #
    # train_summary, validation_summary = get_synthetic_summary_holdout(simulate_nuisance_and_easy_treatment,
    #                                                                   n=10000,
    #                                                                   valid_size=0.2,
    #                                                                   k=10)


    train_preds, valid_preds = get_synthetic_preds_holdout(simulate_nuisance_and_easy_treatment,
                                                           n=50000,
                                                           valid_size=0.2)
    get_synthetic_auuc(train_preds, drop_learners=['S Learner (LR)'])
    x=2
if __name__ == '__main__':
    main()