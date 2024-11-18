import pandas as pd

from rapidfuzz import fuzz, utils
from category_dicts import car_brand_to_country, subject_to_state_of_russia

from sklearn.model_selection import train_test_split


df_full = pd.read_csv('../data/cars_data_clear.csv')

# Функция для извлечения федерального округа места продажи авто
def get_region(adress):
    for subregion in subject_to_state_of_russia.keys():
        if (
            subregion in adress
            or fuzz.WRatio(subregion, adress, processor=utils.default_process) >= 90
        ):
            return subject_to_state_of_russia.get(subregion)
    return 'Другое'

df_full['region'] = df_full['city'].apply(get_region)

# Функция для извлечения марки автомобиля
def extract_brand(model):
    for brand in car_brand_to_country.keys():
        if (
            brand in model
            or fuzz.token_set_ratio(brand, model, processor=utils.default_process) >= 95
        ):
            return car_brand_to_country.get(brand)
    return 'Другое'

df_full['brand'] = df_full['model'].apply(extract_brand)
df_full['is_dealer'] = df_full['is_dealer'].apply(lambda x: x=='TRUE')

numeric_columns = [
    'year', 'engine',
    'mileage', 'power_horse',
    'price'
]

df_full[numeric_columns] = df_full[numeric_columns].apply(pd.to_numeric)

features_ml = [
    'price', 'ad_description', 'is_dealer',
    'year', 'engine', 'transmission',
    'mileage', 'power_horse', 'car_body',
    'wheel_drive', 'fuel_type', 'brand',
    'region'
]

df_full = df_full[features_ml]

df_full.columns = [
    'price', 'text', 'is_dealer',
    'year', 'engine', 'transmission',
    'mileage', 'power_horse', 'car_body',
    'wheel_drive', 'fuel_type', 'brand',
    'region'
]

dummy_features = [
    'transmission', 'car_body',
    'wheel_drive','fuel_type', 'brand', 
    'region'
]

df_full = pd.get_dummies(df_full, columns=dummy_features, drop_first=True)

df_train, df_test = train_test_split(df_full, test_size=0.25, random_state=42)

df_full.to_csv('../data/ml_data.csv', index=False)
df_train.to_csv('../data/train_data.csv', index=False)
df_test.to_csv('../data/test_data.csv', index=False)
