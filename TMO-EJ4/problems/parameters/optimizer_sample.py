from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
from optimization.gp import bayesian_optimization, plot_iteration

data, target = make_classification(n_samples=2500,
                                   n_features=45,
                                   n_informative=15,
                                   n_redundant=5)

def sample_loss(params):
    return cross_val_score(SVC(C=10 ** params[0], gamma=10 ** params[1], random_state=12345),
                           X=data, y=target, scoring='roc_auc', cv=3).mean()


bounds = np.array([[-4, 1], [-4, 1]])

xp, yp = bayesian_optimization(n_iters=30,
                               sample_loss=sample_loss,
                               bounds=bounds,
                               n_pre_samples=3,
                               random_search=100000)


lambdas = np.linspace(1, -4, 25)
gammas = np.linspace(1, -4, 20)
plot_iteration(lambdas, xp, yp, first_iter=3, second_param_grid=gammas, optimum=[0.58333333, -2.15789474], filepath='data/gp')