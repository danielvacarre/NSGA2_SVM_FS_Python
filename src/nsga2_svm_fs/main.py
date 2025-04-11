from numpy.random import rand
from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler

from src.nsga2_svm_fs.nsga2_svmfs import NSGA2_SVMFS

datos = read_csv("14Arrythmia_sinceros.txt",sep=";", header=None)
first_column = datos.columns[0]
datos = datos.drop(columns=[first_column])
columnas = ['y1']
inputs = ['x' + str(i + 1) for i in range(len(datos.columns) - 1)]
columnas = columnas + inputs
datos.columns = columnas

# Especifica las columnas a normalizar

output = 'y1'

# Elimina columnas no deseadas
datos_norm = datos[inputs]

# Normaliza los datos
scaler = MinMaxScaler()
datos_norm = DataFrame(scaler.fit_transform(datos_norm), columns=inputs)
datos_norm[output] = datos[output]

costs = rand(260)
num_features = 5
population_size = 100
method = 'MC-COST'

algorithm = NSGA2_SVMFS(method, datos_norm, costs, population_size, inputs, output, num_features)
algorithm.run(train='iter', num_iter=10)

algorithm.draw_solution()


