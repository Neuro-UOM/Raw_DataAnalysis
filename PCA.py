import pandas as pd

df = pd.read_csv(
    filepath_or_buffer='raw_nadun.csv',
    header=None,
    sep=',')

#df.columns=['F3 Value','F3 Quality','FC5 Value','FC5 Quality','F7 Value','F7 Quality','T7 Value','T7 Quality','P7 Value','P7 Quality','O1 Value','O1 Quality','O2 Value','O2 Quality','P8 Value','P8 Quality','T8 Value','T8 Quality','F8 Value','F8 Quality','AF4 Value','AF4 Quality','FC6 Value','FC6 Quality','F4 Value','F4 Quality','AF3 Value','AF3 Quality','X Value','Y Value','Class']
df.columns=['T7 Value','P7 Value','O1 Value','O2 Value','P8 Value','T8 Value','Class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

df.tail()


X = df.ix[:,0:6].values
y = df.ix[:,6].values

import numpy as np

np.isnan(X).any()

np.isinf(X).any()
X = np.nan_to_num(X)

from matplotlib import pyplot as plt
import numpy as np
import math

label_dict = {1: 'null',
              2: 'red',
              3: 'green'}

feature_dict = {0: 'T7',
                1: 'P7',
                2: 'O1',
                3: 'O2',
                4: 'P8',
                5: 'T8'}

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(8, 6))
    for cnt in range(6):
        plt.subplot(2, 2, cnt+1)
        for lab in ('null', 'red', 'green'):
            plt.hist(X[y==lab, cnt],
                     label=lab,
                     bins=10,
                     alpha=0.3,)
        plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)

    plt.tight_layout()
    plt.show()


'''
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

import numpy as np
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)





cor_mat2 = np.corrcoef(X.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat2)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)




# Fourier 
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

X_fft = scipy.fftpack.fft(X_std)

X_fft.tofile('fft_version.csv',sep=',',format='(%s+%sj)') 
'''