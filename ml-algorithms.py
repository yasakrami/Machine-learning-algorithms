
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]
kmeans = KMeans(n_clusters=3, random_state=11)
kmeans.fit(iris.data)
tsne_data = TSNE(n_components=2 ,random_state=11)
iris_tsne = tsne_data.fit_transform(iris.data)
iris_tsne_df = pd.DataFrame(iris_tsne, columns=['Component1', 'Component2'])
iris_tsne_df['species'] = iris_df.species
axes = sns.scatterplot(data=iris_tsne_df, x='Component1', y='Component2', hue='species', legend='brief') 
iris_centers = tsne_data.fit_transform(kmeans.cluster_centers_)
dots = plt.scatter(iris_centers[:,0], iris_centers[:,1], s=100, c='k')
#Question a's answer: According to what can be seen from the result, in the class we have used PCA cluster method and here
#Tsne method is used. In PCA the dispersion of the results were much less than TSNE. Also in the TSNE it will show closer
#points to each other 

from sklearn.datasets import fetch_california_housing
import seaborn as sns
california= fetch_california_housing()
calif_df = pd.DataFrame(california.data, columns=california.feature_names)
sns.set_style('whitegrid')
grid = sns.pairplot(data=calif_df, vars=calif_df.columns[0:4])

print("Part C")
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]
kmeans = KMeans(n_clusters=3, random_state=11)
knc = KNeighborsClassifier()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=11)  
knc.fit(X=x_train, y=y_train)
print("The accuracy is", f'{knc.score(x_test, y_test):.2%}')

print("Part D")
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
diabets= sklearn.datasets.load_diabetes()
print(diabets.DESCR)
linear_regression = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(diabets.data, diabets.target, random_state=11)
linear_regression.fit(X=X_train, y=y_train)
print("The linear coefficient is",linear_regression.coef_)
print("The linear intercepts are",linear_regression.intercept_)
for i, name in enumerate(diabets.feature_names):
    print(f'{name:>10}: {linear_regression.coef_[i]}')  
predict = linear_regression.predict(X_test)
expect = y_test
dataFrame = pd.DataFrame()
dataFrame['Predict'] = pd.Series(predict)
dataFrame['Expect'] = pd.Series(expect)
figure = plt.figure(figsize=(9, 9))
axes = sns.scatterplot(data= dataFrame, x=dataFrame["Expect"], y=dataFrame["Predict"], hue=dataFrame["Predict"], palette='cool')
start = min(expect.min(), predict.min())
end = max(expect.max(), predict.max())
axes.set_xlim(start, end)
axes.set_ylim(start, end)
line = plt.plot([start, end], [start, end], 'k--')


print("Part E")
import graphviz 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
titanic = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/carData/TitanicSurvival.csv', index_col=False)
df= pd.DataFrame(titanic)
new_df= df.drop(["Unnamed: 0","sex"], axis=1)
new_df['sex'] = encoder.fit_transform(df['sex']) 
new_df['survived'] = encoder.fit_transform(df['survived'])
new_df["passengerClass"]= new_df["passengerClass"].replace({"1st": "1", "2nd": "2", "3rd":"3"})
neww_df = new_df.dropna()
print(neww_df.head())
x = neww_df.drop(['survived'], axis=1).values
y = neww_df['survived'].values
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.1, random_state=0)
clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names= None,  class_names=['survived','not survived'],  filled=True, rounded=True,  special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

