# The Kalman machine library (v2)
## Python files
A library to assess Bayesian sequential algorithms based on the Kalman filtering framework.

The content of the files are the following:

- [KDataGenerator][1]: generate synthetic dataset for linear and logistic regression
- [KFactorAnalysis][2]: the class to compute factor analysis with several methods: Batch EM, Online EM, Recursive EM or SVD 
- [KBayesianReg][3]: abstract class describing the Bayesian framework for large scale online regression with or without factor analysis
- [Kalman4LinearReg][4]: the online Bayesian algorithms for linear regression, including Linear Kalman (or RLS) and large scale Linear Kalman with       factor analysis.
- [Kalman4LogisticReg][5]: the online Bayesian algorithms for logistic regression (without factor analysis),
  including several methods: EKF, QKF, RVGA implicit, RVGA explicit, Mirror prox
- [Kalman4LogisticRegLS][6]: the online Bayesian algorithms for large scale (LS) logistic regression with factor analysis,
  including several methods: large scale version of EKF, RVGA implicit, RVGA explicit, Mirror prox and Mirror prox with Sampling
- [KEvalPosterior][7]: a class to compute the KL with respect to the true posterior in linear and logistic regression
- [KUtils][8]: usefull static functions such as logistic losses, importance sampling, fast log det, etc. 
- [KVizualizationsHD][9]: functions to draw the KL results for different methods in the same plot.

[1]: ./KDataGenerator.py
[2]: ./KFactorAnalysis.py
[3]: ./KBayesianReg.py
[4]: ./Kalman4LinearReg.py
[5]: ./Kalman4LogisticReg.py
[6]: ./Kalman4LogisticRegLS.py
[7]: ./KEvalPosterior.py
[8]: ./KUtils.py
[9]: ./KVizualizationsHD.py
