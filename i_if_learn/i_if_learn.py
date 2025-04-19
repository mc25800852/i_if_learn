## @file: i_IF_learn.py
## @brief: This file contains the implementation of the i_IFP_learn algorithm.

## Import necessary libraries
import os
import math
import umap
import random
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import cosine_distances
from scipy.optimize import linear_sum_assignment
from scipy import stats
from scipy.stats import kstest,f_oneway,norm,f,multinomial
from scipy.stats import kruskal
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

# =========================
# Basic methods
# =========================

def calculate_accuracy(y_pred,y_true):
    """
    Calculate the accuracy of predicted labels against true labels using the Hungarian algorithm.
    Args:
        y_pred (np.ndarray): Predicted labels.
        y_true (np.ndarray): True labels.
    Returns:
        float: Accuracy of the predicted labels.
    """
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost_matrix = np.zeros((len(labels_true), len(labels_pred)))
    for i, true_label in enumerate(labels_true):
        for j, pred_label in enumerate(labels_pred):
            matches = np.sum((y_true == true_label) & (y_pred == pred_label))
            cost_matrix[i, j] = -matches
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_matches = -cost_matrix[row_ind, col_ind].sum()
    accuracy = total_matches / len(y_true)
    return accuracy

def read_data(file_path, delimiters):
    """
    Read data from a file with multiple possible delimiters.
    Args:
        file_path (str): Path to the data file.
        delimiters (list): List of possible delimiters.
    Returns:
        np.ndarray: Data read from the file.
    """
    for delimiter in delimiters:
        try:
            return np.loadtxt(file_path, delimiter=delimiter)
        except ValueError:
            continue
    raise ValueError(f"Failed to read {file_path} with given delimiters")

def replace_inf_with_extremes(arr):
    """
    Replace inf and -inf values in an array with the maximum and minimum finite values.
    Args:
        arr (np.ndarray): Input array.
    Returns:
        np.ndarray: Array with inf and -inf replaced.
    """
    arr = np.array(arr, dtype=float)  
    finite_values = arr[np.isfinite(arr)]     
    if len(finite_values) == 0:
        raise ValueError("Array doesn't have finite values, can't replace inf or -inf")
            
    max_val = np.max(finite_values)
    min_val = np.min(finite_values)

    arr[arr == np.inf] = max_val  
    arr[arr == -np.inf] = min_val      
    return arr 

def qqplot(data, title="QQ Plot"):
    """
    Create a QQ plot to visualize the distribution of data.
    Args:
        data (np.ndarray): Input data.
        title (str): Title of the plot.
    """
    data = np.array(data)
    N = len(data)
    sorted_data = np.sort(data)
    theoretical_quantiles = np.linspace(1/(N+1), N/(N+1), N)

    plt.figure(figsize=(6, 6))
    plt.scatter(theoretical_quantiles, sorted_data, color='blue', label="Data Points")

    plt.plot([0, 1], [0, 1], color='black', linestyle='-', linewidth=1, label="y=x Line")
    
    plt.xlabel("Theoretical Quantiles (Uniform)")
    plt.ylabel("Sample Quantiles")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# =========================
# IF-PCA methods
# =========================

def calculate_hct_threshold(pi,n,p):
    """
    Calculate the threshold for the HCT (higher criticism threshold) method.
    Args:
        pi (np.ndarray): Array of p-values.
        n (int): Number of samples.
        p (int): Number of features.
    Returns:
        int: Threshold for the HCT method.
    """
    sorted_pi = np.sort(pi)
    HC_p_j = np.zeros(p)
    for j in range(1, p + 1):
        HC_p_j[j - 1] = math.sqrt(p) * (j / p - sorted_pi[j - 1]) / math.sqrt(
            max((math.sqrt(n) * (j / p - sorted_pi[j - 1])), 0) + j / p)

    valid_indices = np.where(np.logical_and(sorted_pi >math.log(p)/p, np.arange(1, p+1) < p/2))[0]
    if valid_indices.size > 0:
        j_hat = valid_indices[np.argmax(HC_p_j[valid_indices])]
        print("j_hat",j_hat)
    else:
        j_hat=None
    return j_hat

def ks_statistic_single_sample(X):
    """
    Perform the Kolmogorov-Smirnov test for each feature in the dataset.
    Args:
        X (np.ndarray): Input data.
    Returns:
        tuple: KS statistics and p-values for each feature.
    """
    p_features = X.shape[1]
    ks_stats = []
    p_values=[]
    for j in range(p_features):
        ks_stat,pv= kstest(X[:, j], 'norm')
        p_values.append(pv)
        ks_stats.append(ks_stat)

    return ks_stats,p_values

def simulate_ks_distribution(n, rep=100000):
    """
    Simulate the Kolmogorov-Smirnov distribution.
    Args:
        n (int): Number of samples.
        rep (int): Number of repetitions for simulation.
    Returns:
        np.ndarray: Simulated KS statistics.
    """
    ks_values = np.zeros(rep)
    
    for i in range(rep):
        
        x = np.random.randn(n)
        x = (x - np.mean(x)) / np.std(x)  
        pi = np.sort(stats.norm.cdf(x))  
        kk = np.arange(0, n + 1) / n     
        
        # 计算 KS 统计量
        ks_stat = np.max(np.abs(kk[1:n+1] - pi)) 
        ks_stat = max(ks_stat, np.max(np.abs(kk[:n] - pi))) 
        
        ks_values[i] = ks_stat

        '''
        # or use
        ks_stat,_= kstest(x, 'norm')
        ks_values[i] = ks_stat
        '''
    
    return ks_values

def adjust_ks_stats(ks_data, ks_simulated):
    """
    Adjust the KS statistics based on the simulated distribution.
    Args:
        ks_data (np.ndarray): KS statistics from the data.
        ks_simulated (np.ndarray): Simulated KS statistics.
    Returns:
        np.ndarray: Adjusted KS statistics.
    """
    ks_data = np.array(ks_data)
    ks_simulated = np.array(ks_simulated)
    mean_ks = np.mean(ks_data)
    std_ks = np.std(ks_data)
    mean_sim = np.mean(ks_simulated)
    std_sim = np.std(ks_simulated)
    
    ks_adjusted = (ks_data - mean_ks) / std_ks * std_sim + mean_sim  
    return ks_adjusted

def calculate_pi(ks_simulated,ks_adjusted):
    """
    Calculate the p-values based on the KS statistics.
    Args:
        ks_simulated (np.ndarray): Simulated KS statistics.
        ks_adjusted (np.ndarray): Adjusted KS statistics.
    Returns:
        np.ndarray: p-values for each feature.
    """
    p_values = []
    for ks in ks_adjusted:
        count = np.sum(ks_simulated <= ks)
        pi = 1 - count / len(ks_simulated)
        p_values.append(pi)
    p_values = np.array(p_values)

    return p_values

def if_pca(X,K,threshold=None):
    """
    Perform IF-PCA on the input data.
    Args:
        X (np.ndarray): Input data.
        K (int): Number of clusters for KMeans.
        threshold (int, optional): Number of top features to select. If None, it will be calculated.
    Returns:
        tuple: Clusters: cluster, indices of influential features: influential_features_indices, KS statistics: ks_data, adjusted p-values of KS test: pi.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)/np.sqrt(X.shape[0])*np.sqrt((X.shape[0]-1))
    n,p=X_scaled.shape
    ks_simulated = simulate_ks_distribution(n)

    ks_data,_=ks_statistic_single_sample(X_scaled)

    ks_adjusted = adjust_ks_stats(ks_data, ks_simulated)
    pi=calculate_pi(ks_simulated,ks_adjusted)

    if threshold is None:
        threshold = calculate_hct_threshold(pi,n,p)
    
    influential_features_indices = np.argsort(ks_adjusted)[-threshold:]

    influential_features = X_scaled[:, influential_features_indices]

    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(influential_features,influential_features.T))

    V = eigenvectors[:, np.argsort(eigenvalues)[::-1][:K-1]]

    kmeans = KMeans(n_clusters=K,n_init=30)

    clusters = kmeans.fit_predict(V)

    return clusters, influential_features_indices,ks_data,pi

# =========================
# Prerequisite methods
# =========================

def calculate_anova(X, labels):
    """
    Perform ANOVA on the input data.
    Args:
        X (np.ndarray): Input data.
        labels (np.ndarray): Labels for the data.
    Returns:
        tuple: F-statistics and p-values for each feature.
    """
    num_features = X.shape[1]
    F_stats = []
    p_values=[]

    if (len(np.unique(labels))==1):
        print("There is only one group")
    for i in range(num_features):
        feature_values = [X[labels == label, i] for label in np.unique(labels)]
        F_stat, pv = f_oneway(*feature_values)
        if F_stat < 0:
            F_stat = 0
        pv = 1.0 if np.isnan(pv) else pv
        F_stats.append(F_stat)
        p_values.append(pv)
    return F_stats,p_values

def multiple_random_sampling(X, true_labels, K,component, prev_influential_indices, num_trials):
    """
    Perform multiple random sampling for KMeans clustering.
    Args:
        X (np.ndarray): Input data.
        true_labels (np.ndarray): True labels for the data.
        K (int): Number of clusters.
        component (int): Number of components for spectral embedding.
        prev_influential_indices (list): Indices of influential features.
        num_trials (int): Number of trials for random sampling.
    Returns:
        list: List of results from each trial.
    """
    results = []
    n, p = X.shape
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)/np.sqrt(X.shape[0])*np.sqrt((X.shape[0]-1))
    kmeans = KMeans(n_clusters=K,n_init=30)

    for i in range(num_trials):
        random_indices = random.sample(range(p), len(prev_influential_indices))
        random_features = Z[:, random_indices]
        
        cosine_dist = cosine_distances(random_features)
        gamma = 1  
        affinity_matrix = np.exp(-gamma * cosine_dist ** 2)
        
        embedding = SpectralEmbedding(
            n_components=component,
            affinity='precomputed',
            n_neighbors=8
        )
        X_random = embedding.fit_transform(affinity_matrix)
        
        # kmeans = KMeans(n_clusters=K, n_init=30, random_state=46)
        labels_random = kmeans.fit_predict(X_random)
        
        error = 1 - calculate_accuracy(labels_random, true_labels)
        f_random, _ = calculate_anova(X, labels_random)
        f_random = np.array(f_random)
        random_f_data = f_random[random_indices]
        
        results.append({
            'random_indices': random_indices,
            'error': error,
            'random_f_data': random_f_data
        })
        
        # print(f"Iteration {i+1}: error = {error}")
    return results

# =========================
# Non-parametric methods
# =========================

def simulate_kruskal_distribution(n, K, rep=100000):
    """
    Simulate the Kruskal-Wallis distribution.
    Args:
        n (int): Number of samples.
        K (int): Number of groups.
        rep (int): Number of repetitions for simulation.
    Returns:
        np.ndarray: Simulated Kruskal-Wallis statistics.
    """
    H_values = np.zeros(rep)
    for i in range(rep):
        x = np.random.randn(n)
        labels = [random.randint(0, K - 1) for _ in range(n)]
        data = pd.DataFrame({'values': x, 'group': labels})

        try:
            H_stat = stats.kruskal(*(data[data['group'] == j]['values'] for j in range(K))).statistic
        except ValueError:
            H_stat = 0 

        H_values[i] = H_stat

    H_values = H_values[~np.isnan(H_values)]
    
    return H_values

def calculate_kruskal_anova(X, labels):
    """
    Perform Kruskal-Wallis test on the input data.
    Args:
        X (np.ndarray): Input data.
        labels (np.ndarray): Labels for the data.
    Returns:
        tuple: H-statistics and p-values for each feature.
    """
    num_features = X.shape[1]
    H_stats = []
    p_values = []

    if len(np.unique(labels)) == 1:
        print("There is only one group")
        return None, None

    for i in range(num_features):
        feature_values = [X[labels == label, i] for label in np.unique(labels)]
        try:
            H_stat, pv = kruskal(*feature_values)
        except ValueError:
            H_stat, pv = 0, 1.0  # fallback for degenerate cases
        H_stats.append(H_stat)
        p_values.append(pv)

    return H_stats, p_values

def multiple_random_sampling_kw(X, true_labels, K, component, prev_influential_indices,num_trials):
    """
    Perform multiple random sampling for Kruskal-Wallis test.
    Args:
        X (np.ndarray): Input data.
        true_labels (np.ndarray): True labels for the data.
        K (int): Number of clusters.
        component (int): Number of components for spectral embedding
        kmeans (KMeans): KMeans object for clustering.
        prev_influential_indices (list): Indices of influential features.
        num_trials (int): Number of trials for random sampling.
    Returns:
        list: List of results from each trial.
    """
    results = []
    n, p = X.shape
    component = K + 2
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)/np.sqrt(X.shape[0])*np.sqrt((X.shape[0]-1))
    kmeans = KMeans(n_clusters=K,n_init=30)
    
    for i in range(num_trials):
        random_indices = random.sample(range(p), len(prev_influential_indices))
        random_features = Z[:, random_indices]
        
        cosine_dist = cosine_distances(random_features)
        gamma = 1  
        affinity_matrix = np.exp(-gamma * cosine_dist ** 2)
        
        embedding = SpectralEmbedding(
            n_components=component,
            affinity='precomputed',
            n_neighbors=8
        )
        X_random = embedding.fit_transform(affinity_matrix)
        
        labels_random = kmeans.fit_predict(X_random)
        
        error = 1 - calculate_accuracy(labels_random, true_labels)
        f_random, _ = calculate_kruskal_anova(X, labels_random)
        f_random = np.array(f_random)
        random_f_data = f_random[random_indices]
        
        results.append({
            'random_indices': random_indices,
            'error': error,
            'random_h_data': random_f_data
        })
        
        # print(f"Iteration {i+1}: error = {error}")
    return results


# =========================
# Main methods
# =========================

def compute_values(scores):
    """
    Compute the values of b, c, ScoreMaximum, and weights (unmapped) based on the input scores.
    Args:
        scores (list): List of scores.
    Returns:
        tuple: b, c, ScoreMaximum, weights.
    """
    
    scores = np.array(scores) 
    two_thirds = int(len(scores) * 2 / 3)
    top_scores = scores[-two_thirds:]
    scores=top_scores
    S_len = len(scores)
    b = np.sqrt(2 * np.log(np.log(S_len)))
    c = 2 * np.log(np.log(S_len)) + 0.5 * np.log(np.log(np.log(S_len))) - np.log(4 * np.pi)/2
    
    ScoreMaximum = max(scores)
    weights = np.exp(-np.exp(c - b * ScoreMaximum))
    
    return b, c, ScoreMaximum, weights

def i_if_learn(X,labels,K,prev_influential_indices,constant,ksp,method,component,random_results,max_iter=10,convergence_threshold = 0.10):
    """
    Perform the i_if_learn algorithm on the input data.
    Args:
        X (np.ndarray): Input data.
        labels (np.ndarray): Labels given by initialization method (IF-PCA).
        K (int): Number of clusters for KMeans.
        prev_influential_indices (list): Indices of influential features from initialization (IF-PCA).
        constant (str | float): Controls the feature selection strategy. Three types of values are supported:
            - 'hct': Uses the HCT method to determine the threshold adaptively to select top features.
            - float > 1: Interpreted as a percentage. For example, `constant=10` selects the top 10% of features with the lowest p-values.
            - float <= 1: Uses a sparsity-based theoretical threshold given by `-constant * sqrt(log(p))`, selects features with p-values below this threshold.
        ksp (float): adjusted p-value for KS test.
        method (str): Method for dimensionality reduction ('laplacian', 'UMAP', or 'PCA').
        component (int): Number of components for spectral embedding.
        random_results (list): Results from multiple random sampling.
        max_iter (int, optional): Maximum number of iterations. Default is 10.
        convergence_threshold (float, optional): Convergence threshold. Default is 0.10.
    Returns:
        top_k_indices (np.ndarray): Indices of the selected influential features.
        labels (np.ndarray): Final cluster labels.
    """
    n,p=X.shape
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)/np.sqrt(X.shape[0])*np.sqrt((X.shape[0]-1))
    prev_influential_indices = np.array(prev_influential_indices, dtype=int)
    df1=K-1
    df2=n-K

    kmeans = KMeans(n_clusters=K,n_init=30)
    for iteration in range(max_iter):
        print(f".............Iteration {iteration + 1}/{max_iter}...............")

        f_data,_=calculate_anova(X,labels)
        f_data=replace_inf_with_extremes(f_data)
        f_data=np.array(f_data)

        mean_theoretical = stats.f.mean(dfn=df1, dfd=df2)
        std_theoretical = stats.f.std(dfn=df1, dfd=df2)
        f_adj=(f_data-np.mean(f_data))/np.std(f_data)*std_theoretical+mean_theoretical
        fp = 1 - stats.f.cdf(f_adj, df1, df2)

        influential_f_data = f_data[prev_influential_indices]

        num_influential = len(influential_f_data)           
        random_all = np.concatenate([res['random_f_data'] for res in random_results], axis=0)  

        pij = np.array([
            np.mean(inf_j < random_all)
            for inf_j in influential_f_data
        ])
        sorted_indices = np.argsort(pij)
        sorted_pi = pij[sorted_indices]
        j_vals = np.arange(1, num_influential + 1)   
        scores = np.sqrt(num_influential) * ( j_vals / n- sorted_pi ) / np.sqrt(sorted_pi * (1 - sorted_pi) )
        b, c, ScoreMaximum, w1 = compute_values(scores)
        # print("b", b, "c", c, "ScoreMaximum", ScoreMaximum, "weight", w1)

        w=1-(1-w1)/(1-w1+0.6)

        zf=np.array(norm.ppf(fp))
        zks=np.array(norm.ppf(ksp))
        zks=replace_inf_with_extremes(zks)
        zf=replace_inf_with_extremes(zf)
        S=w*zf+(1-w)*zks
        # print("weight",w)

        z_scores=S/(w**2+(1-w)**2)
        

        if(constant=='hct'):
            pi=norm.cdf(z_scores)
            threshold = calculate_hct_threshold(pi,n,p)
            current_influential_indices = np.argsort(z_scores)[:threshold] 
            influential_features = Z[:, current_influential_indices]

        elif(constant>1):
            percent = constant/100  
            num_features = p
            threshold = int(num_features * percent)
            current_influential_indices = np.argsort(z_scores)[:threshold] 
            influential_features = Z[:, current_influential_indices]

        else:
            threshold = -constant*np.sqrt(np.log(p))
            influential_features = Z[:, z_scores<= threshold]
            current_influential_indices = np.where(z_scores<= threshold)[0]

        if(method=='laplacian'):
            cosine_dist = cosine_distances(influential_features)
            gamma = 1 
            affinity_matrix = np.exp(-gamma * cosine_dist**2)
            embedding = SpectralEmbedding(
                n_components=component,    
                affinity='precomputed',  
                n_neighbors=8
            )
            X_low = embedding.fit_transform(affinity_matrix)

        elif(method=='UMAP'):
            eigenvalues, eigenvectors = np.linalg.eigh(np.dot(influential_features,influential_features.T))
            V = eigenvectors[:, np.argsort(eigenvalues)[::-1][:component]] 
            umap_model = umap.UMAP(n_neighbors=5,n_components=component, metric='cosine',angular_rp_forest=True,init=np.array(V)) 
            X_low = umap_model.fit_transform(influential_features)

        else:
            eigenvalues, eigenvectors = np.linalg.eigh(np.dot(influential_features,influential_features.T))
            X_low = eigenvectors[:, np.argsort(eigenvalues)[::-1][:component]]

        clusters = kmeans.fit_predict(X_low)
        labels = clusters
        # accuracy = calculate_accuracy(labels, true_labels)
        # error=1-accuracy
        # print("error rate", error)

        if prev_influential_indices is not None:
            changed_features = np.setdiff1d(current_influential_indices, prev_influential_indices)
            feature_change_ratio = len(changed_features) / len(prev_influential_indices)
            print(f"Feature change ratio: {feature_change_ratio * 100:.2f}%")

            if feature_change_ratio < convergence_threshold:
                print(f"Converged after {iteration + 1} iterations with feature change ratio {feature_change_ratio * 100:.2f}%")
                break

        prev_influential_indices = current_influential_indices

    return current_influential_indices,labels

