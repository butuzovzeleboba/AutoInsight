import pandas as pd

from rapidfuzz import fuzz, utils
from constants import (
    CAR_BRANDS_TO_COUNTRY, SUBJECTS_TO_STATE_OF_RUSSIA,
    DUMMY_FEATURES, MODEL_FEATURES, PRICE_CLASSES_AMOUNT
)

from sklearn.model_selection import train_test_split


df_full = pd.read_csv('../data/cars_data_clear.csv')

# Функция для извлечения федерального округа места продажи авто
def get_region(adress):
    for subregion in SUBJECTS_TO_STATE_OF_RUSSIA.keys():
        if (
            subregion in adress
            or fuzz.WRatio(subregion, adress, processor=utils.default_process) >= 90
        ):
            return SUBJECTS_TO_STATE_OF_RUSSIA.get(subregion)
    return 'Другое'

df_full['region'] = df_full['city'].apply(get_region)

# Функция для извлечения марки автомобиля
def extract_brand(model):
    for brand in CAR_BRANDS_TO_COUNTRY.keys():
        if (
            brand in model
            or fuzz.token_set_ratio(brand, model, processor=utils.default_process) >= 95
        ):
            return CAR_BRANDS_TO_COUNTRY.get(brand)
    return 'Другое'

df_full['brand'] = df_full['model'].apply(extract_brand)
df_full['is_dealer'] = df_full['is_dealer'].apply(lambda x: x=='TRUE')

numeric_features = [
    'year', 'engine',
    'mileage', 'power_horse',
    'price'
]

df_full[numeric_features] = df_full[numeric_features].apply(pd.to_numeric)

features_need = [
    'price', 'ad_description', 'is_dealer',
    'year', 'engine', 'transmission',
    'mileage', 'power_horse', 'car_body',
    'wheel_drive', 'fuel_type', 'brand',
    'region'
]

df_full = df_full[features_need]

df_full.columns = MODEL_FEATURES

# КОдируем категориальные переменные
df_full = pd.get_dummies(df_full, columns=DUMMY_FEATURES, drop_first=True)

df_train, df_test = train_test_split(df_full, test_size=0.25, random_state=42)

# Усекаем выбросы для трейна
df_train = df_train[df_train.price <= df_train.price.quantile(0.999)]
df_train = df_train[df_train.price >= df_train.price.quantile(0.001)]

# Переводим непрерывную цену в категории цены, для nlp модели
df_train['price_class'] = pd.qcut(
    df_train['price'], q=PRICE_CLASSES_AMOUNT, labels=range(0, PRICE_CLASSES_AMOUNT)
)
df_test['price_class'] = pd.qcut(
    df_test['price'], q=PRICE_CLASSES_AMOUNT, labels=range(0, PRICE_CLASSES_AMOUNT)
)

# Сохраняем все в папку data
df_full.to_csv('../data/ml_data.csv', index=False)
df_train.to_csv('../data/train_data.csv', index=False)
df_test.to_csv('../data/test_data.csv', index=False)
