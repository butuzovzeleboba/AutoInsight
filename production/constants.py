PRICE_CLASSES_AMOUNT = 10

MODEL_FEATURES = [
    'price', 'text', 'is_dealer',
    'year', 'engine', 'transmission',
    'mileage', 'power_horse', 'car_body',
    'wheel_drive', 'fuel_type', 'brand',
    'region'
]

DUMMY_FEATURES = [
    'transmission', 'car_body',
    'wheel_drive','fuel_type', 'brand', 
    'region'
]

CAR_BRANDS_TO_COUNTRY = {
    'ВАЗ': 'Россия',
    'Toyota': 'Япония',
    'Tesla': 'США',
    'Acura': 'Япония',
    'Hyundai': 'Южная Корея',
    'Ford': 'США',
    'Opel': 'Германия',
    'Mazda': 'Япония',
    'Renault': 'Франция',
    'Mercedes-Benz': 'Германия',
    'Volkswagen': 'Германия',
    'OMODA': 'Китай',
    'Audi': 'Германия',
    'Honda': 'Япония',
    'Volvo': 'Швеция',
    'Kia': 'Южная Корея',
    'Dodge': 'США',
    'Skoda': 'Чехия',
    'Peugeot': 'Франция',
    'Geely': 'Китай',
    'Changan': 'Китай',
    'Lexus': 'Япония',
    'Citroen': 'Франция',
    'LIFAN': 'Китай',
    'LiXiang': 'Китай',
    'Mitsubishi': 'Япония',
    'Chery': 'Китай',
    'Dongfeng': 'Китай',
    'Chevrolet': 'США',
    'Chrysler': 'США',
    'BMW': 'Германия',
    'Great Wall': 'Китай',
    'Nissan': 'Япония',
    'Land Rover': 'Великобритания', 
    'Nissan': 'Япония',
    'Suzuki': 'Япония',
    'Новый': 'Россия',
    'УАЗ': 'Россия',
    'ТагАЗ': 'Россия',
    'ГАЗ': 'Россия',
    'FIAT': 'Италия',
    'Datsun': 'Япония',
    'Daewoo': 'Южная Корея', 
    'Genesis': 'Южная Корея',
    'JAC': 'Китай',
    'Rover': 'Великобритания',
    'Infiniti': 'Япония',
    'Subaru': 'Япония',
    'Porsche': 'Германия',
    'MINI': 'Великобритания',
    'SsangYong': 'Южная Корея',
    'ВИС': 'Россия',
    'Haima': 'Китай',
    'Jeep': 'США',
    'RAM': 'США',
    'Dodge': 'США',
    'Jaguar': 'Великобритания',
    'Cadillac': 'США',
    'Vortex': 'Россия',
    'HAVAL': 'Китай',
    'ЗАЗ': 'Украина',
    'Ravon': 'Узбекистан',
    'Daihatsu': 'Япония',
    'EXEED': 'Китай',
    'BYD': 'Китай',
    'Москвич': 'Россия',
    'Iran': 'Иран',
    'ИЖ': 'Россия',
    'SEAT': 'Испания',
    'Saab': 'Швеция',
    'Brilliance': 'Китай',
    'FAW': 'Китай',
    'Zeekr': 'Китай',
    "Pagani": "Италия",
    "Ferrari": "Италия",
    "Lamborghini": "Италия",
    "Aston Martin": "Великобритания",
    "Lotus": "Великобритания",
    "McLaren": "Великобритания",
    "Buick": "США",
    "Lincoln": "США",
    "Alfa Romeo": "Италия",
    "Peugeot": "Франция",
    "Seat": "Испания",
    "Skoda": "Чехия",
    "Tata": "Индия",
    "Mahindra": "Индия",
    "Proton": "Малайзия",
    "Voyah": "Китай",
    "ZX": "Китай",
    "Derways": "Россия",
    "Doninvest": "Россия",
    "Jetour": "Китай",
    "Dacia": "Румыния",
    "Maserati": "Италия",
    "JAECOO": "Китай",
    "ZOTYE": "Китай",
    "Smart": "Германия",
    "Bugatti": "Франция",
    "Tianye": "Китай",
    "Богдан": "Украина",
    "Wey": "Китай",
    "Hafei": "Китай",
    "Jetta": "Китай",
    "Pontiac": "США",
    "Isuzu": "Япония",
    "JMC": "Китай",
    "Chana": "Китай",
    "LDV": "Великобритания",
    "BAIC": "Китай",
    "Iveco": "Италия",
    "Saturn": "США",
    "Tank": "Китай",
    "Trabant": "Германия",
    "GMC": "США",
    "GAC": "Китай",
    "Landwind": "Китай",
    "ЛуАЗ": "Украина",
    "Kaiyi": "Китай",

}

SUBJECTS_TO_STATE_OF_RUSSIA = {
    "Республика Адыгея": "Южный",
    "Адыгейская Республика": "Южный",
    "Республика Башкортостан": "Приволжский",
    "Башкортостанская Республика": "Приволжский",
    "Республика Бурятия": "Сибирский",
    "Бурятская Республика": "Сибирский",
    "Республика Алтай": "Сибирский",
    "Алтайская Республика": "Сибирский",
    "Республика Дагестан": "Южный",
    "Дагестанская Республика": "Южный",
    "Республика Ингушетия": "Южный",
    "Ингушская Республика": "Южный",
    "Республика Кабардино-Балкарская": "Южный",
    "Кабардино-Балкарская Республика": "Южный",
    "Республика Калмыкия": "Южный",
    "Калмыцкая Республика": "Южный",
    "Республика Карачаево-Черкесская": "Южный",
    "Карачаево-Черкесская Республика": "Южный",
    "Республика Карелия": "Северо-Западный",
    "Карельская Республика": "Северо-Западный",
    "Республика Коми": "Северо-Западный",
    "Коми Республика": "Северо-Западный",
    "Республика Крым": "Южный",
    "Крымская Республика": "Южный",
    "Республика Марий Эл": "Приволжский",
    "Марийская Республика": "Приволжский",
    "Республика Мордовия": "Приволжский",
    "Мордовская Республика": "Приволжский",
    "Республика Саха": "Дальневосточный",
    "Республика Саха (Якутия)": "Дальневосточный",
    "Якутская Республика": "Дальневосточный",
    "Республика Северная Осетия": "Южный",
    "Северная Осетия": "Южный",
    "Республика Северная Осетия - Алания": "Южный",
    "Республика Татарстан": "Приволжский",
    "Татарстанская Республика": "Приволжский",
    "Республика Тыва": "Сибирский",
    "Тывинская Республика": "Сибирский",
    "Республика Удмуртская": "Приволжский",
    "Удмуртская Республика": "Приволжский",
    "Республика Хакасия": "Сибирский",
    "Хакасская Республика": "Сибирский",
    "Республика Чечня": "Южный",
    "Чеченская Республика": "Южный",
    "Чувашская Республика": "Приволжский",
    "Чувашская Республика - Чувашия": "Приволжский",
    "Краснодарский край": "Южный",
    "Красноярский край": "Сибирский",
    "Алтайский край": "Сибирский",
    "Приморский край": "Дальневосточный",
    "Ставропольский край": "Южный",
    "Хабаровский край": "Дальневосточный",
    "Забайкальский край": "Дальневосточный",
    "Амурская обл.": "Дальневосточный",
    "Архангельская обл.": "Северо-Западный",
    "Астраханская обл.": "Южный",
    "Белгородская обл.": "Центральный",
    "Брянская обл.": "Центральный",
    "Владимирская обл.": "Центральный",
    "Волгоградская обл.": "Южный",
    "Вологодская обл.": "Северо-Западный",
    "Воронежская обл.": "Центральный",
    "Донецкая Народная Республика": "Южный",
    "Ивановская обл.": "Центральный",
    "Иркутская обл.": "Сибирский",
    "Калининградская обл.": "Северо-Западный",
    "Калужская обл.": "Центральный",
    "Камчатский край": "Дальневосточный",
    "Кемеровская обл.": "Сибирский",
    "Кировская обл.": "Приволжский",
    "Костромская обл.": "Центральный",
    "Курганская обл.": "Уральский",
    "Курская обл.": "Центральный",
    "Ленинградская обл.": "Северо-Западный",
    "Липецкая обл.": "Центральный",
    "Луганская Народная Республика": "Южный",
    "Магаданская обл.": "Дальневосточный",
    "Московская обл.": "Центральный",
    "Мурманская обл.": "Северо-Западный",
    "Нижегородская обл.": "Приволжский",
    "Новгородская обл.": "Северо-Западный",
    "Новосибирская обл.": "Сибирский",
    "Омская обл.": "Сибирский",
    "Оренбургская обл.": "Приволжский",
    "Орловская обл.": "Центральный",
    "Пензенская обл.": "Приволжский",
    "Пермский край": "Приволжский",
    "Псковская обл.": "Северо-Западный",
    "Ростовская обл.": "Южный",
    "Рязанская обл.": "Центральный",
    "Самарская обл.": "Приволжский",
    "Саратовская обл.": "Приволжский",
    "Сахалинская обл.": "Дальневосточный",
    "Свердловская обл.": "Уральский",
    "Смоленская обл.": "Центральный",
    "Тамбовская обл.": "Центральный",
    "Тверская обл.": "Центральный",
    "Томская обл.": "Сибирский",
    "Тульская обл.": "Центральный",
    "Тюменская обл.": "Уральский",
    "Ульяновская обл.": "Приволжский",
    "Челябинская обл.": "Уральский",
    "Ярославская обл.": "Центральный",
    "Санкт-Петербург": "Северо-Западный",
    "Севастополь": "Южный",
    "Москва": "Центральный",
    "Еврейская АО": "Дальневосточный",
    "Ненецкий АО": "Северо-Западный",
    "Ханты-Мансийский АО - Югра": "Уральский",
    "Ханты-Мансийский АО": "Уральский",
    "Херсонская обл.": "Южный",
    "Чукотский АО": "Дальневосточный",
    "Ямало-Ненецкий АО": "Уральский",
}
