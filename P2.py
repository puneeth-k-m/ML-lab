import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = pd.read_csv("housing.csv")
numeric_data = data.select_dtypes(include=[float,int])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm',fmt=' .3f',linewidths=0.1)
plt.title('correlation matrix of california housing features')
plt.show()

sns.pairplot(data,diag_kind='kde',plot_kws={'alpha':0.5})
plt.suptitle('Pair of California Housing Features',y=1.5)
plt.show()
