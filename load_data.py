# libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
# split
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.datasets import load_iris


# load data 'penguins'
df = pd.read_csv('penguins.csv')
df.columns

# show values in one column
df['species'].value_counts()

#####################################################################################
######### data analysis

#null values
df.isnull().sum()
df['sex'].value_counts()

# info
df.info()

# groub by 'species'
df.groupby(['species'])['sex'].value_counts()

# drop null vaues
df.dropna(inplace=True)

####################################################################################
######### data visualization

sns.pairplot(df, hue='species')

####################################################################################
######### split data

X = df.drop('species', axis=1)
Y = df['species']

# one hot encoding
X = pd.get_dummies(X) 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=Y)
 
####################################################################################
########



