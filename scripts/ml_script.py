# Author: Alvaro Garcia
# Description: Script of ML Model for ML/DL Project 
# Date: June 29th, 2025
#####################################################

# Using data from:
# https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data

# Run with 
# streamlit run streamlit_app.py --server.headless true --server.enableCORS false --server.address=0.0.0.0 
# For WSL system

### Importing Libraries #################################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

from sklearn.linear_model import LinearRegression

from sklearn.metrics import make_scorer, mean_absolute_error

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
# Custom K‑fold target encoder 
class KFoldTargetEncoder(TargetEncoder):
    def __init__(self, cols=None, smoothing=0.3, **kw):
        super().__init__(cols=cols, smoothing=smoothing, **kw)


from xgboost import XGBRegressor
########################################################################


#  Main file (used to differentiate when exporting KFold Module)
if __name__ == "__main__":
    # Get data
    data = pd.read_csv('./used_cars.csv')



    ### Data Cleaning c##################################################
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

    # Preparing a function:
    def log_to_price_mae(y_true_log, y_pred_log):
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
        return mean_absolute_error(y_true, y_pred)

    ### Model (Version 3.1)
    # X and y
    X = df[[
        "brand_model", "accident", "clean_title",
        "milage", "car_age",
        "engine_hp", "engine_liter", "engine_cyl",
        "fuel_type", "trans_spd", "trans_type"
    ]]
    y = df.price
    y_log = np.log1p(y)

    # Preprocessor
    brand_model_enc = KFoldTargetEncoder(cols=["brand_model"], smoothing=0.3)
    ohe_small = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_cols = ["milage", "car_age", "engine_hp", "engine_liter", "engine_cyl", "trans_spd"]

    preprocess = ColumnTransformer([
            ("brand_model_enc", brand_model_enc, ["brand_model"]),
            ("cat_small", ohe_small, ["accident", "clean_title", "fuel_type", "trans_type"]),
            ("num", Pipeline([
                            ("impute", SimpleImputer(strategy="median")),
                            ("pass", "passthrough")
                            ]),
                            numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Model
    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=3,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("prep", preprocess),
        ("xgb", xgb),
    ])

    ### Evalute model (used for testing models)
    # cv = KFold(n_splits=5, shuffle=True, random_state=3)

    # mae_scorer = make_scorer(log_to_price_mae, greater_is_better=False)

    # mae_scores = -1 * cross_val_score(
    #     pipe,
    #     X, y_log,
    #     cv=cv,
    #     scoring=mae_scorer,
    #     n_jobs=-1,
    # )

    # print(f"MAE per fold:  {np.round(mae_scores, 0)}")
    # print(f"Mean ± std MAE: {mae_scores.mean():,.0f} ± {mae_scores.std():,.0f}")
    ### RESULTS:
    # MAE per fold:  [ 8379. 12392. 13812.  9180. 10714.]
    # Mean ± std MAE: 10,896 ± 2,003
    ########################################################################


    ### Save model:
    import joblib

    pipe.fit(X, y_log)

    joblib.dump(pipe, "ml_model.joblib")

    ### To load:
    # pipe = joblib.load("ml_model.joblib")
    # prediction = pipe.predict(raw_df)
