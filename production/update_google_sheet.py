import os
import gspread
import pandas as pd

from dotenv import load_dotenv
from df2gspread import df2gspread as d2g 

from oauth2client.service_account import ServiceAccountCredentials


# Загрузка переменных окружения из файла .env
load_dotenv()

# Получение значений переменных окружения
my_mail = os.getenv('GOOGLE_SHEET_EMAIL')
table_id = os.getenv('GOOGLE_SHEET_PREDICTIONS_ID')

path_to_credentials = '../crdentials.json'

# Authorization 
scope = ['https://spreadsheets.google.com/feeds', 
        'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name(path_to_credentials, scope) 
gs = gspread.authorize(credentials)

# Читаем csv и обновляем Google Sheet 
with open('../data/cars_data_predictons.csv', 'r', encoding='utf-8') as f:
    result = f.read()

gs.import_csv(table_id, result)