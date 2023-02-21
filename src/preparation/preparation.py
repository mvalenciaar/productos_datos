# Import Libraries
import pandas as pd
import numpy as np
from pandas import ExcelFile


def read_file(data,data_dupl,miss_values, data_dtypes,filter):
    data = pd.read_csv('card_transdata.csv')
    #print(data)
    data['used_chip'] = data['used_chip'].astype(int)
    data['repeat_retailer'] = data['repeat_retailer'].astype(int)
    data['used_pin_number'] = data['used_pin_number'].astype(int)
    data['online_order'] = data['online_order'].astype(int)
    data['fraud'] = data['fraud'].astype(int)
    data_desc = data.describe().T
    #print(data_desc)

    # Duplicated values
    data_dupl = data.duplicated().sum()
    print(f'La cantidad de valores duplicados son: {data_dupl}')

    #Find the null values 
    miss_values = data.isnull().sum()
    print(f'Los valores nulos son: {miss_values}')

    #Know the datatype
    data_dtypes = data.dtypes
    print(f'El tipo de datos para este conjunto de datos es: {data_dtypes}')

    # Filter data
    # Filtro para saber qué clientes hicieron compras sólo en comercio físico
    filter = data[data['used_chip'] == 1].head(10)
    print(f'Clientes que hicieron compra en comercio físico: {filter}')

    return data,data_dupl,miss_values, data_dtypes,filter