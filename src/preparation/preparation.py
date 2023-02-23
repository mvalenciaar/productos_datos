# Import Libraries
import pandas as pd
import numpy as np
from pandas import ExcelFile
import os

# Parámetros de simulación


def load_file_card():
    ''' This function loads the data and reads the file '''
    module_path = os.path.dirname(__file__)
    filename = os.path.join(module_path, "card_transdata.csv")
    data = pd.read_csv(filename, sep=',')
    return data


def cleansing_data():
    ''' This function is responsible for data cleansing '''
    clean_data = load_file_card()
    clean_data['used_chip'] = clean_data['used_chip'].astype(int)
    clean_data['repeat_retailer'] = clean_data['repeat_retailer'].astype(int)
    clean_data['used_pin_number'] = clean_data['used_pin_number'].astype(int)
    clean_data['online_order'] = clean_data['online_order'].astype(int)
    clean_data['fraud'] = clean_data['fraud'].astype(int)

    #Describe
    data_desc = clean_data.describe().T
    print(f'Describe de la data: {data_desc}')   

    # Duplicated values
    data_dupl = clean_data.duplicated().sum()
    print(f'Cantidad de valores duplicados: {data_dupl}')
    
    #Find the null values 
    miss_values = clean_data.isnull().sum()
    print(f'Los valores nulos son: {miss_values}')

    #Know the datatype
    data_dtypes = clean_data.dtypes
    print(f'El tipo de datos para este conjunto de datos es: {data_dtypes}')

    # Filter data
    # Filtro para saber qué clientes hicieron compras sólo en comercio físico
    filter = clean_data[clean_data['used_chip'] == 1].head(10)
    print(f'Clientes que hicieron compra en comercio físico: {filter}')

    return clean_data

def purchases_cardholder():
    ''' This function displays the purchases made by cardholders '''
    data_purch = cleansing_data()
    data_purchase = data_purch[data_purch.columns[0]].count()
    
    return data_purchase

def purchases_cardholder(n):
    ''' This function displays the n purchases made by cardholders '''
    data_purch = cleansing_data()
    data_purch_n = data_purch.head(n)
    return data_purch_n


def onsite_transactions_validation():
    ''' This function  returns the validations of transactions made on site '''
    val_tran = cleansing_data()
    filt_data_onsite = val_tran[val_tran['used_chip'] == 1]
    return filt_data_onsite   

def online_transactions_validation():
    ''' This function  returns the validations of transactions made online '''
    val_tran = cleansing_data()
    filt_data_online = val_tran[val_tran['online_order'] == 1]
    return filt_data_online   

def onsite_final_state():
    ''' This function shows if the transaction is fraudulent on site '''
    data_final = onsite_transactions_validation()
    valid_data_onsite = data_final[data_final['fraud'] == 1]
    return valid_data_onsite

def online_final_state():
    ''' This function shows if the transaction is fraudulent online '''
    data_final = online_transactions_validation()
    valid_data_online = data_final[data_final['fraud'] == 1]
    return valid_data_online
