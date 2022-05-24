import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

data_set = pd.read_csv('dataset/wine.csv')

# X - Caracteristicas
X = data_set[['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids',
              'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280_OD315', 'Proline']]

# Y - Rótulo
Y = data_set['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

modelo = SVC()
modelo.fit(X_train, Y_train)

predicoes_treino = modelo.predict(X_train)
matriz_conf_treino = confusion_matrix(Y_train, predicoes_treino)

ax = sns.heatmap(matriz_conf_treino, annot=True, cmap='Greens', fmt='d',
                 xticklabels=['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
                              'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue',
                              'OD280_OD315', 'Proline']
                 , yticklabels=['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols',
                                'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue',
                                'OD280_OD315', 'Proline'])
ax.set(ylabel='Real', xlabel='Predito', title='Dados de Treino')
plt.show()
