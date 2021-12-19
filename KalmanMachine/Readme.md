## The Kalman machine library
# Python files
A library to assess Bayesian sequential algorithms based on the Kalman filtering framework.

The arborescence is the following:

- KDataGenerator: generate synthetic dataset for linear and logistic regression
- BayesianLogisticReg: the Bayesian framework for logistic regression: include Laplace approximation
- Kalman4LogisticReg: the online Bayesian algorithms for logistic regression: include three versions for logistic regression: EKF, QKF and RVGA
- KEvalPosterior: the evaluation metrics to assess posterior estimation for logistic regression
