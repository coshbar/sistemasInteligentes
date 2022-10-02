
import pandas as pd
from sklearn import preprocessing
from pickle import dump
data = pd.read_csv('/normalizar.csv', sep = ';')
#normalizacao
data_normalized = (data - data.min()) / (data.max() - data.min())
#padronizacao
data_patterned = (data - data.mean()) / data.std()
#normalizacao e modelo
normalizer = preprocessing.MinMaxScaler()
normalized_data_model = normalizer.fit(data)
#salvar
dump(normalized_data_model, open('normalized_data_model', 'wb'))
