# In[] Preprocessing
import HappyML.preprocessor as pp

# Load Dataset
dataset = pp.dataset(file="CreditCards.csv")

# Features overview
discribe=dataset.describe()

# Missing value count
dataset.isna().sum()

# Decomposition
X = pp.decomposition(dataset, x_columns=[i for i in range(1, 18)])

# Processing outliers 
import numpy as np
def outlier_percent(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    minimum = Q1 - (1.5 * IQR)
    maximum = Q3 + (1.5 * IQR)
    num_outliers =  np.sum((data < minimum) |(data > maximum))
    num_total = data.count()
    return (num_outliers/num_total)*100

for column in X.columns:
    data = X[column]
    percent = str(round(outlier_percent(data), 2))
    print(f'Outliers in "{column}": {percent}%')
    
for column in X.columns:
    data = X[column]
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    minimum = Q1 - (1.5 * IQR)
    maximum = Q3 + (1.5 * IQR)
 
    outliers = ((data < minimum) |(data > maximum))
    X[column].loc[outliers] = np.nan
    
X.isna().sum()

# Imputing missing values
from sklearn.impute import KNNImputer
import pandas as pd
imputer = KNNImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X.isna().sum()

# Feature Scaling (for PCA Feature Selection)
X_std = pp.feature_scaling(fit_ary=X, transform_arys=X)

# Feature Selection (PCA)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=0.9, random_state=0)
pca.fit(X_std)
PC_names = ['PC'+str(x) for x in range(1,len(pca.components_)+1)]
pca_data = pd.DataFrame(pca.transform(X_std), columns=PC_names)

fig, ax = plt.subplots(figsize=(24, 16))
plt.imshow(pca.components_.T,
           cmap=plt.cm.gist_ncar,
           vmin=-1,
           vmax=1,
          )
plt.yticks(range(len(X_std.columns)), X_std.columns)
plt.xticks(range(len(pca_data.columns)), pca_data.columns)
plt.xlabel("Principal Component")
plt.ylabel("Contribution")
plt.title("Contribution of Features to Components")
plt.colorbar()

# Feature Selection (PCA)
from HappyML.preprocessor import PCASelector

selector = PCASelector()

X_select = selector.fit(x_ary=X_std,verbose=True, plot=True).transform(x_ary=X_std)

# In[] K-Means Clustering (With HappyML's Class)
from HappyML.clustering import KMeansCluster

cluster = KMeansCluster()
Y_pred = cluster.fit(x_ary=X_select, verbose=True, plot=True).predict(x_ary=X_select, y_column="Customer Type")

# Attach the Y_pred to Dataset & Save as .CSV file
dataset = pp.combine(X, Y_pred)
dataset.to_csv("Creditcard_answers.csv", index=False)


# In[] Visualization 
import HappyML.model_drawer as md
selector = PCASelector(best_k=2)

X_std = selector.fit(x_ary=X_std,verbose=True, plot=True).transform(x_ary=X_std)

md.cluster_drawer(x=X_std, y=Y_pred, centroids=cluster.centroids, title="Customers Segmentation")
