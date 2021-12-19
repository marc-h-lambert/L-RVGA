# L-RVGA

## Object

This is the companion code for the article "The limited recursive variational Gaussian approximation (L-RVGA)" \[[1][A]\]. Please cite this reference if you use this code. This code is the last update and may be not compatible with the code published for the previous article in \[[3][X]\].  

## Installation
The code is available in python using the standard library. We depends on the "scipy" library for optimization, we use the scipy powell method to compute the implicit scheme and the scipy l-bfgs method to compute the maximum posterior for Laplace approximation. If not available on your distribution, you may install scipy following https://www.scipy.org/install.html.

## python files
The source of the library "Kalman Machine" which implement the assesed algorithms is available in python [here][0]. To reproduce the results of the paper we can run one of the following files:
- [XP_LRVGA_LargeScaleCovariance][1]: Run the Recursive EM method to approximate large scale covariance matrix with factor analysis (Section 7.1 of the article \[[1][A]\]). If we want to reproduce the results on libsvm datset we may download the BreastCancer, Madelon and Heart dataset from \[[2][B]\] and put the text files in the repository DataSet.
- [XP_LRVGA_LinearRegression][2]: Apply the recsurive EM to a linear regression problem (Section 7.2 of the article \[[1][A]\])
- [XP_LRVGA_LargeScaleCovariance][3]: Apply the recsurive EM to a logistic regression problem (Section 7.3 of the article \[[1][A]\]) and assess sampling to extend results to more general problems (Section 7.4 of the article \[[1][A]\]) 

[0]: ./KalmanMachine
[1]: ./XP_LRVGA_LargeScaleCovariance.py
[2]: ./XP_LRVGA_LinearRegression.py
[3]: ./XP_LRVGA_LogisticRegression.py

[X]: https://github.com/marc-h-lambert/Kalman4Classification
[A]: https://hal.inria.fr/hal-0308662X
[B]: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

\[1\]: ["The limited recursive variational Gaussian approximation (L-RVGA), Marc Lambert, Silvere Bonnabel and Francis Bach".][A] 

\[2\]: ["LIBSVM Data: Classification, Regression, and Multi-label".][B] 

