# The Kalman machine library
## Python files
A library to assess Bayesian sequential algorithms based on the Kalman filtering framework.

The arborescence is the following:

- KDataGenerator: generate synthetic dataset for linear and logistic regression
- KFactorAnalysis: the class to compute factor analysis with several methods: Batch EM, Online EM, Recursive EM or SVD 
- KBayesianReg: abstract class describing the Bayesian framework for large scale online regression with or without factor analysis
- Kalman4LinearReg: the online Bayesian algorithms for linear regression, including Linear Kalman (or RLS) and large scale Linear Kalman with       factor analysis.
- Kalman4LogisticReg: the online Bayesian algorithms for logistic regression (without factor analysis),
  including several methods: EKF, QKF, RVGA implicit, RVGA explicit, Mirror prox
- Kalman4LogisticRegLS: the online Bayesian algorithms for large scale (LS) logistic regression with factor analysis,
  including several methods: large scale version of EKF, RVGA implicit, RVGA explicit, Mirror prox and Mirror prox with Sampling
- KEvalPosterior: a class to compute the KL with respect to the true posterior in linear and logistic regression
- KUtils: usefull static functions such as logistic losses, importance sampling, fast log det, etc. 
- KVizualizationsHD: functions to draw the KL results for different methods in the same plot.
