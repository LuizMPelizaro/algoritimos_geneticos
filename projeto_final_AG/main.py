import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
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
                 xticklabels=['Class_1', 'Class_2', 'Class_3']
                 , yticklabels=['Class_1', 'Class_2', 'Class_3'])
ax.set(ylabel='Real', xlabel='Predito', title='Dados de Treino')

plt.show()

matriz_conf_treino_norm = confusion_matrix(Y_train, predicoes_treino, normalize='true')
ax2 = sns.heatmap(matriz_conf_treino_norm, annot=True, cmap='Greens', fmt='.2%',
                  xticklabels=['Class_1', 'Class_2', 'Class_3']
                  , yticklabels=['Class_1', 'Class_2', 'Class_3'])
ax2.set(ylabel='Real', xlabel='Predito', title='Dados de Treino')

# Calculando a predição com os dados de teste
predicoes_teste = modelo.predict(X_test)

# Calcula a matriz de confusão
matriz_conf_teste = confusion_matrix(Y_test, predicoes_teste)

matriz_conf_teste_norm = confusion_matrix(Y_test, predicoes_teste, normalize='true')

ax3 = sns.heatmap(matriz_conf_teste, annot=True, cmap='Greens', fmt='d',
                  xticklabels=['Class_1', 'Class_2', 'Class_3']
                  , yticklabels=['Class_1', 'Class_2', 'Class_3'])
ax3.set(ylabel='Real', xlabel='Predito', title='Dados de Treino')

ax4 = sns.heatmap(matriz_conf_teste_norm, annot=True, cmap='Greens', fmt='.2%',
                  xticklabels=['Class_1', 'Class_2', 'Class_3']
                  , yticklabels=['Class_1', 'Class_2', 'Class_3'])
ax4.set(ylabel='Real', xlabel='Predito', title='Dados de Treino')

print(classification_report(Y_test, predicoes_teste))

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear', 'sigmoid']}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, Y_train)

print(grid.best_params_)

best_model = grid.best_estimator_
best_predictions = best_model.predict(X_test)
matriz_conf_final = confusion_matrix(Y_test, best_predictions, normalize='true')

ax5 = sns.heatmap(matriz_conf_final, annot=True, cmap='Greens', fmt='.2%',
                  xticklabels=['Class_1', 'Class_2', 'Class_3']
                  , yticklabels=['Class_1', 'Class_2', 'Class_3'])

ax5.set(ylabel='Real', xlabel='Predito', title='Dados de Treino')

print(classification_report(Y_test, best_predictions))
