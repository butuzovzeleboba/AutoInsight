import os
import gspread
import pandas as pd

from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from oauth2client.service_account import ServiceAccountCredentials


# Загрузка переменных окружения из файла .env
load_dotenv()

# Получение значений переменных окружения
my_mail = os.getenv('GOOGLE_SHEET_EMAIL')
table_url = os.getenv('GOOGLE_SHEET_CLEAN_DATA_URL')

path_to_credentials = '../crdentials.json'

# Authorization 
scope = ['https://spreadsheets.google.com/feeds', 
        'https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name(path_to_credentials, scope) 
gs = gspread.authorize(credentials)

# Get this table 
work_sheet = gs.open_by_url(table_url) 

# Select 1st sheet 
sheet1 = work_sheet.sheet1 

# Get data in python lists format 
data = sheet1.get_all_values() 

# Get header from data 
headers = data.pop(0) 

# Create df 
df = pd.DataFrame(data, columns=headers) 
df_train, df_test = train_test_split(df, random_state=42, test_size=0.25)

df.to_csv('../data/cars_data_clear.csv')
df_train.to_csv('../data/train.csv')
df_test.to_csv('../data/test.csv')
   