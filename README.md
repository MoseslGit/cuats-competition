## Mid With Bad Ideas Code

# Execution Instructions

1. Download all files
2. Open all files in the QuantConnect terminal
3. Run main.py 

For detailed breakdowns of the code, refer to comments around functions.

# Model Training
A Bayesian Gaussian mixture model from sklearn is used. For documentation, refer to https://scikit-learn.org/stable/modules/mixture.html#bgmm. The BIC information criterion is used by this package naturally to decide the ideal number of clusters below a maximum set limit. 
In practice it is found that setting of the priors has little effect on cluster assignments for this 4-cluster case. In general, larger values lead to more evenly weighted clusters, but too large may result in unnecessary splitting. The default weight concentration prior (for mixing coefficients) for the Dirichlet distribution is 1.0. We tested up to 100 with no discernible difference. Smaller values should not be used in this instance as it may result in setting one or more clusters to zero.

Additional variables: any number of additional data variables can be added simply by extending the number of columns in the data array. In practice it is found that if some particularly nonpredictive variables are used, the clustering becomes less effective. These can be removed by decomposing individual cluster Gaussian covariances into an eigenvalue-eigenvector decomposition, and projecting each of the eigenvectors onto their n closest axes. Tallying up the number of times an axis corresponds to a large eigenvalue for all pairs and all clusters indicates how relevant an axis is. 

Data normalisation: to analyse the covariance eigenvector-eigenvalue decomposition properly to clean out unimportant variables, all data fed in should be normalised. This can be done using sklearn's preprocessing package to fit a normal distribution N(0,1) as

```
data = np.array(data) # data is some 2D array
scaler = preprocessing.StandardScaler().fit(data)
data_scaled = scaler.transform(data)
```
