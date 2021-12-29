# In[] Preprocessing
import numpy as np
import pandas as pd

def dataset(file=""):
    if file != "":
        dataset = pd.read_csv(file)
    else:
        dataset = None

    return dataset

# Load Dataset
dataset = dataset(file="CreditCards.csv")

# Features overview
discribe=dataset.describe()

# Missing value count
dataset.isna().sum()

# In[] Decomposite Dataset into Independent Variables & Dependent Variables
def decomposition(dataset, x_columns, y_columns=[]):
    X = dataset.iloc[:, x_columns]
    Y = dataset.iloc[:, y_columns]

    if len(y_columns) > 0:
        return X, Y
    else:
        return X

X = decomposition(dataset, x_columns=[i for i in range(1, 18)])

# In[] Processing outliers 
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

# In[]Imputing missing values

from sklearn.impute import KNNImputer
import pandas as pd
imputer = KNNImputer()
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X.isna().sum()

# In[] Feature Scaling (for PCA Feature Selection)

from sklearn.preprocessing import StandardScaler

def feature_scaling(fit_ary, transform_arys=None):
    scaler = StandardScaler()
    scaler.fit(fit_ary.astype("float64"))

    if type(transform_arys) is tuple:
        return (pd.DataFrame(scaler.transform(ary.astype("float64")), index=ary.index, columns=ary.columns) for ary in transform_arys)
    else:
        return pd.DataFrame(scaler.transform(transform_arys.astype("float64")), index=transform_arys.index, columns=transform_arys.columns)

X_std = feature_scaling(fit_ary=X, transform_arys=X)

# In[] PCA

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

# In[] Feature Selection (PCA)

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class PCASelector:
    __selector = None
    __best_k = None
    __info_covered = None
    __strategy = None

    def __init__(self, best_k="auto"):
        if type(best_k) is int:
            self.__strategy = "fixed"
            self.__best_k = best_k
        elif type(best_k) is float:
            self.__strategy = "percentage"
            self.__info_covered = best_k
            self.__best_k = None
        else:
            self.__strategy = "auto"
            self.__best_k = None

        self.__selector = PCA(n_components=self.__best_k)

    @property
    def selector(self):
        return self.__selector

    @property
    def best_k(self):
        return self.__best_k

    def fit(self, x_ary, verbose=False, plot=False):
        # Get information covered by each component
        pca = PCA(n_components=None)
        pca.fit(x_ary)
        info_covered = pca.explained_variance_ratio_
        if verbose:
            print("Information Covered by Each Component:")
            print(info_covered)
            print()
        
        # Cumulated Coverage of Information
        cumulated_sum = np.cumsum(info_covered)
        info_covered_dict = dict(zip([i+1 for i in range(info_covered.shape[0])], cumulated_sum))
        if verbose:
            print("Cumulated Coverage of Information:")
            for n, c in info_covered_dict.items():
                print("{:3d}: {}".format(n, c))
            print()

        if self.__strategy == "auto":
            scaler = MinMaxScaler(feature_range=(0, info_covered.shape[0]-1))
            scaled_info_covered = scaler.fit_transform(info_covered.reshape(-1, 1)).ravel()
            for i in range(1, scaled_info_covered.shape[0]):
                if (scaled_info_covered[i-1]-scaled_info_covered[i]) < 1:
                    break
            self.__best_k = i
        elif self.__strategy =="percentage":
            current_best = 1
            cummulated_info = 0.0
            for i in info_covered:
                cummulated_info += i
                if cummulated_info < self.__info_covered:
                    current_best += 1
                else:
                    break
            self.__best_k = current_best

        self.__selector = PCA(n_components=self.best_k)
        self.selector.fit(x_ary)

        if verbose:
            print("Strategy: {}".format(self.__strategy))
            print("Select {} components, covered information {:.2%}".format(self.best_k, info_covered_dict[self.best_k]))
            print()

        if plot:
            np.insert(cumulated_sum, 0, 0.0)
            plt.plot(cumulated_sum, color="blue")
            plt.scatter(x=self.best_k, y=cumulated_sum[self.best_k], color="red")
            plt.title("Components vs. Information")
            plt.xlabel("# of Components")
            plt.ylabel("Covered Information (%)")
            plt.show()

        return self

    def transform(self, x_ary):
        X_columns = ["PCA_{}".format(i) for i in range(1, self.best_k+1)]
        return pd.DataFrame(self.selector.transform(x_ary), index=x_ary.index, columns=X_columns)

selector = PCASelector()

X_select = selector.fit(x_ary=X_std,verbose=True, plot=True).transform(x_ary=X_std)

# In[] K-Means Clustering (With HappyML's Class)
import time
from sklearn.cluster import KMeans

class KMeansCluster:
    __cluster = None
    __best_k = None
    __max_k = None
    __strategy = None
    __random_state = None
    __centroids = None
    
    def __init__(self, best_k="auto", max_k=10, random_state=int(time.time())):
        if type(best_k) is int:
            self.__strategy = "fixed"
            self.best_k = best_k
        else:
            self.__strategy = "auto"
            self.best_k = 8
        
        self.__max_k = max_k
        self.__random_state = random_state
        
        self.__cluster = KMeans(n_clusters=self.best_k, max_iter=300, n_init=10, init="k-means++", random_state=self.__random_state)
    
    @property
    def cluster(self):
        return self.__cluster
    
    @cluster.setter
    def cluster(self, cluster):
        self.__cluster = cluster
    
    @property
    def best_k(self):
        return self.__best_k
    
    @best_k.setter
    def best_k(self, best_k):
        if (type(best_k) is int) and (best_k >= 1):
            self.__best_k = best_k
        else:
            self.__best_k = 1
    
    @property
    def centroids(self):
        return self.__centroids
    
    def fit(self, x_ary, verbose=False, plot=False):
        if self.__strategy == "auto":
            wcss = []
            for i in range(1, self.__max_k+1):
                kmeans = KMeans(n_clusters=i, max_iter=300, n_init=10, init="k-means++", random_state=self.__random_state)
                kmeans.fit(x_ary)
                wcss.append(kmeans.inertia_)
            
            scaler = MinMaxScaler(feature_range=(0, len(wcss)-1))
            wcss_scaled = scaler.fit_transform(np.array(wcss).reshape(-1, 1)).ravel()
            for i in range(1, wcss_scaled.shape[0]):
                if (wcss_scaled[i-1]-wcss_scaled[i]) < 1:
                    break
            self.best_k = i
            
            if verbose:
                print("The best clusters = {}".format(self.best_k))
            
            if plot:
                plt.plot(range(1, len(wcss)+1), wcss, color="blue")
                plt.scatter(x=self.best_k, y=wcss[self.best_k], color="red")
                plt.title("The Best Cluster")
                plt.xlabel("# of Clusters")
                plt.ylabel("WCSS")
                plt.show()
        
        # Fit the Model
        self.cluster = KMeans(n_clusters=self.best_k, random_state=self.__random_state)
        self.cluster.fit(x_ary)
        self.__centroids = self.cluster.cluster_centers_
        
        return self
    
    def predict(self, x_ary, y_column="Result"):
        return pd.DataFrame(self.cluster.predict(x_ary), index=x_ary.index, columns=[y_column])

cluster = KMeansCluster()
Y_pred = cluster.fit(x_ary=X_select, verbose=True, plot=True).predict(x_ary=X_select, y_column="Customer Type")


# In[]  Attach the Y_pred to Dataset & Save as .CSV file
def combine(dataset, y_pred):
    return dataset.join(y_pred)

dataset = combine(X, Y_pred)
dataset.to_csv("Creditcard_answers.csv", index=False)

# In[] Visualization 
import matplotlib.cm as cm

def cluster_drawer(x, y, centroids, title="", font=""):
    # Check for x has only two columns
    if x.shape[1] != 2:
        print("ERROR: x must have only two features to draw!!")
        return None

    # Change y from DataFrame to NDArray
    y_ndarray = y.values.ravel()

    # Get how many classes in y
    y_unique = np.unique(y_ndarray)

    # Iterate all classes in y
    colors = cm.rainbow(np.linspace(0, 1, len(y_unique)))
    for val, col in zip(y_unique, colors):
        plt.scatter(x.iloc[y_ndarray==val, 0], x.iloc[y_ndarray==val, 1], s=50, c=col, label="Cluster {}".format(val))

    # Draw Centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c="black", marker="^", label="Centroids")

    # Labels & Legends
    # for showing Chinese characters
    if font != "":
        plt.rcParams['font.sans-serif']=[font]
        plt.rcParams['axes.unicode_minus'] = False

    plt.title(title)
    plt.xlabel(x.columns[0])
    plt.ylabel(x.columns[1])
    plt.legend(loc="best")
    plt.show()


selector = PCASelector(best_k=2)

X_std = selector.fit(x_ary=X_std,verbose=True, plot=True).transform(x_ary=X_std)

cluster_drawer(x=X_std, y=Y_pred, centroids=cluster.centroids, title="Customers Segmentation")
