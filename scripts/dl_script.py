# Author: Alvaro Garcia
# Description: Script of DL Model for ML/DL Project 
# Date: June 29th, 2025
#####################################################

# Using data from:
# https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data

### Importing Libraries #################################################
import numpy as np 
import pandas as pd 
import re

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.metrics import mean_absolute_error

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
########################################################################

### Setup plotting (used when exploring models)
# plt.style.use('seaborn-whitegrid')
# # Set Matplotlib defaults
# plt.rc('figure', autolayout=True)
# plt.rc('axes', labelweight='bold', labelsize='large',
#        titleweight='bold', titlesize=18, titlepad=10)
# plt.rc('animation', html='html5')


# Get data
data = pd.read_csv('./used_cars.csv')



### Data Cleaning ##################################################
# Cleaning data here:
clean = data.copy()

# Fixing clean_title NaN's
clean.clean_title = clean.clean_title.fillna("Missing")

# Fixing fuel types: NaN -> Electric
clean.fuel_type = clean.fuel_type.fillna("Electric")

# Fixing accident: NaN -> Unknown
clean.accident = clean.accident.fillna("Unknown")
# No more Null Values at this point

# Fixing Numbers from str -> Floats
clean["milage"] = (
    clean["milage"]
      .str.replace(r"[^\d.]", "", regex=True)  # keep only digits + dot
      .astype(float)
)
clean["price"] = (
    clean["price"]
      .str.replace(r"[^\d.]", "", regex=True)  # keep only digits + dot
      .astype(float)
)

# fixing obvious outlier:
clean.loc[(clean["model"] == "Quattroporte Base") &
    (clean["price"] > 2_000_000),
    "price"] = 29540

# Copying a 1st set of columns for the model
df = clean[['brand', 'model', 'model_year', 'milage', 'accident', 'clean_title', 'price']]

# brand and model are combined (high cardinality for both, and no point in deleting models, so just combined)
df["brand_model"] = df["brand"] + " " + df["model"]
# and old columns are dropped:
df = df.drop(['brand', 'model'], axis=1)

# car year is changed to car age (2024 is year where data is obtained)
df["car_age"] = 2024 - df["model_year"]
df = df.drop(['model_year'], axis=1)



###############################



### Second part of preparing data: using apply and regex functions:
# Working on another dataframe:
extra = clean[['engine', 'fuel_type', 'transmission']]

## For engine:
    # Engine had a lot of valuable information packed together, 
    # so I decided to split it into different columns:
# engine fix 1: taking the hp values (creating function)
def extract_hp(row):
    """
    Takes a row, looks at row['engine'], extracts the number that
    appears immediately before 'HP', stores it in row['engine_hp'],
    and deletes that substring from row['engine'].

    Returns the *modified* row so .apply(axis=1) can overwrite the DF.
    """
    engine_str = row['engine']
    
    # regex: one or more digits (optionally with . and more digits) + optional spaces + HP
    match = re.search(r'(\d+(?:\.\d+)?)\s*HP', engine_str, flags=re.I)
    
    if match:
        hp_value = float(match.group(1))
        # Remove the matched substring from the engine string
        start, end = match.span()
        engine_str = (engine_str[:start] + engine_str[end:]).strip()
    else:
        hp_value = np.nan  # couldn’t parse
        # leave engine_str unchanged
    
    # Write back into the row
    row['engine_hp'] = hp_value
    row['engine'] = engine_str
    return row
extra = extra.apply(extract_hp, axis=1)

# engine fix 2: taking the liter values (creating function)
def extract_liter(row):
    engine_str = row['engine']
    
    # regex: one or more digits (optionally with . and more digits) + optional spaces + L
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:Liters?|L\b)', engine_str, flags=re.I)
    
    if match:
        l_value = float(match.group(1))
        # Remove the matched substring from the engine string
        start, end = match.span()
        engine_str = (engine_str[:start] + engine_str[end:]).strip()
    else:
        l_value = np.nan  # couldn’t parse
        # leave engine_str unchanged
    
    # Write back into the row
    row['engine_liter'] = l_value
    row['engine'] = engine_str
    return row
extra = extra.apply(extract_liter, axis=1)

# engine fix 3: taking the cylinder values (creating function)
CYL_PATTERN = re.compile(
    r'(?:^|\s)'                              # start or space
    r'(?:V|Flat|Straight|I)?\s*-?\s*'        # optional layout: V, Flat, Straight, I
    r'(\d+)(?!V)'                            # number NOT followed by V (blocks 48V)
    r'(?:\s*(?:Cylinder|Cyl\b))?',           # optional "Cylinder"/"Cyl"
    flags=re.I
)
def extract_cyl(row):
    engine_str = row['engine']
    
    # regex: one or more digits (optionally with . and more digits) + optional spaces + L
    match = CYL_PATTERN.search(engine_str)
    
    if match:
        cyl_value = int(match.group(1))
        # Remove the matched substring from the engine string
        start, end = match.span()
        engine_str = (engine_str[:start] + engine_str[end:]).strip()
    else:
        cyl_value = np.nan  # couldn’t parse
    
    # Write back into the row
    row['engine_cyl'] = cyl_value
    row['engine'] = engine_str
    return row
extra = extra.apply(extract_cyl, axis=1)

## For fueltype: left as it is as it has low cardinality

## For transmission:
# transmission fix 1: taking speeds
SPEED_PATTERN = re.compile(
    r'''
    \b               # word boundary
    (\d{1,2})        # group 1: 1–2 digits
    \s*              # optional space
    (?:[- ]?         # optional dash or space
       (?:           # one of the speed keywords
          [Ss]peed | # "Speed" (any case)
          [Ss]pd     # "Spd"
       )
    )
    \b
    ''',
    flags=re.VERBOSE | re.IGNORECASE
)
BARE_DIGIT = re.compile(r'^\s*(\d{1,2})\s*$')
def extract_spd(row):
    trans_str = row['transmission']
    
    # regex
    match = SPEED_PATTERN.search(trans_str) or BARE_DIGIT.search(trans_str)
    
    if match:
        spd_value = int(match.group(1))
        # Remove the matched substring from the transmission string
        start, end = match.span()
        trans_str = (trans_str[:start] + trans_str[end:]).replace('--', '-').strip(" -")
    else:
        spd_value = np.nan  # couldn’t parse
    
    # Write back into the row
    row['trans_spd'] = spd_value
    row['transmission'] = trans_str.strip()
    return row
extra = extra.apply(extract_spd, axis=1)

# transmission fix 2: taking the transmission types
DUAL_TOKENS   = re.compile(r'\b(dual|dct|cvt.*manual|at/mt)\b', re.I)
MANUAL_TOKENS = re.compile(r'\b(manual|m/t|mt)\b', re.I)
AUTO_TOKENS   = re.compile(r'\b(auto|a/t|at|automatic|cvt)\b', re.I)
def map_trans_type(s):
    """
    Returns: 'dual', 'manual', 'auto', or 'unknown'
    """
    if s is None:
        return "unknown"
    
    s = str(s).lower()
    
    if DUAL_TOKENS.search(s):
        return "dual"
    
    if MANUAL_TOKENS.search(s):
        return "manual"
    
    if AUTO_TOKENS.search(s):
        return "auto"
    
    return "unknown"
extra["trans_type"] = extra["transmission"].apply(map_trans_type)
####################################################################




### Model Training ##################################################
cols = extra[['engine_hp', 'engine_liter', 'engine_cyl', 'fuel_type', 'trans_type', 'trans_spd']]

# final dataframe we'll use
df = pd.concat([df, cols], axis=1) 

# Setting up data for Model:
X = df.copy()
y = X.pop('price')

features_num = [ 'milage', 'car_age', 'engine_hp', 
                'engine_liter', 'engine_cyl', 'trans_spd']
features_cat = [ 'accident', 'clean_title', 'brand_model',
                 'fuel_type', 'trans_type'] 

transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"), # there are a few missing values
    StandardScaler()
)
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="unknown"),
    OneHotEncoder(handle_unknown='ignore')
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat)
)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75, random_state=3)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

# Scaling target 'y' as well
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1))
y_valid_scaled = scaler_y.transform(y_valid.to_numpy().reshape(-1, 1))

input_shape = [X_train.shape[1]]
# print("Input shape: {}".format(input_shape))    # [1618]


###############################



# Training Model:
myDL = keras.Sequential([
    layers.Dense(256, use_bias=False, input_shape=input_shape),
    # layers.BatchNormalization(),
    layers.ReLU(),

    layers.Dropout(0.1),
    
    layers.Dense(128, use_bias=False),
    # layers.BatchNormalization(),
    layers.ReLU(),
    
    layers.Dropout(0.1),
    
    layers.Dense(64, use_bias=False),
    # layers.BatchNormalization(),
    layers.ReLU(),

    layers.Dropout(0.1),
    
    layers.Dense(32),
    layers.ReLU(),
    
    layers.Dense(1)
])

# Evaluating Model:
myDL.compile(
    optimizer='adam',
    loss='mae'
)

# Early stopping
early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True
)

EPOCHS = 200
history = myDL.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=32,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1,
    # shuffle=False
)

### Used to understand different models
# # Print loss graph (mae)
# history_df = pd.DataFrame(history.history)
# history_df.loc[:, ['loss', 'val_loss']].plot()
# print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

### RESULTS:
# Different runs: 10785.8486 -> 10725.3945 -> 10449.0264 -> 10456.6221
########################################################################


### Save Preprocessor and Model:
import joblib
joblib.dump(preprocessor, "dl_preprocessor.joblib") # Save preprocessor 

myDL.save("dl_model.keras") # Save model

### To load:
# model = keras.models.load_model("dl_model.keras")
# prediction = model.predict(raw_df)