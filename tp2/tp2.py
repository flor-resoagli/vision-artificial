from sklearn import tree
from joblib import dump, load
import csv


# dataset
X = csv.reader('./descriptores.csv')

# etiquetas, correspondientes a las muestras
Y = [0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1]

# entrenamiento
clasificador = tree.DecisionTreeClassifier().fit(X, Y)

# visualización del árbol de decisión resultante
tree.plot_tree(clasificador)

# guarda el modelo en un archivo
dump(clasificador, 'filename.joblib')

# en otro programa, se puede cargar el modelo guardado
clasificadorRecuperado = load('filename.joblib') 

