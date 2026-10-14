import os, joblib, warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from scipy.stats import kendalltau
from packaging.version import Version
from pathlib import Path

warnings.filterwarnings('ignore')

if Version(xgb.__version__) < Version('2.0.0'):
    raise RuntimeError('Gunakan XGBoost >= 2.0.0 untuk multi_output_tree.')

# =========================
# 1. KONFIGURASI
# =========================

BASE_PATH = Path(__file__).resolve().parent

DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH = BASE_PATH / "output"
MODEL_PATH = OUTPUT_PATH / "models"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 123
TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR = 2025

DATASETS = {
    "jogja": {
        2022: "susenas_jogja_2022.csv",
        2023: "susenas_jogja_2023.csv",
        2024: "susenas_jogja_2024.csv",
        2025: "susenas_jogja_2025.csv",
    },
    "jateng": {
        2022: "susenas_jateng_2022.csv",
        2023: "susenas_jateng_2023.csv",
        2024: "susenas_jateng_2024.csv",
        2025: "susenas_jateng_2025.csv",
    }
}


# Fixed dari kode acuan yang kamu berikan
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'random_state': RANDOM_STATE,
    'colsample_bytree': 0.3,
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 700,
    'reg_lambda': 0.01,
    'subsample': 0.7,
}

RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 5,
    'min_samples_split': 3,
    'min_samples_leaf': 3,
    'bootstrap': True,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}

# Satu daftar prediktor untuk SEMUA model
predictors = [
    'h_hhcount','h_nmale','h_nfemale','h_nage04','h_nage519','h_nage2064','h_nage65up',
    'h_ngrad_sd','h_ngrad_smp','h_ngrad_sma','h_ngrad_d','h_ngrad_s','h_notgrad',
    'h_sec1_stat1','h_sec1_stat2','h_sec1_stat3','h_sec1_stat4','h_sec1_stat5','h_sec1_stat6',
    'h_sec2_stat1','h_sec2_stat2','h_sec2_stat3','h_sec2_stat4','h_sec2_stat5','h_sec2_stat6',
    'h_sec3_stat1','h_sec3_stat2','h_sec3_stat3','h_sec3_stat4','h_sec3_stat5','h_sec3_stat6',
    'h_neverschool','h_stillschool','h_notschool',
    'h_house1','h_house2','h_house3','h_house4',
    'h_floor1','h_floor2','h_floor3','h_floor4',
    'h_wall1','h_wall2','h_wall3','h_wall4',
    'h_roof1','h_roof2','h_roof3','h_roof4',
    'h_dwater1','h_dwater2','h_dwater3','h_dwater4','h_dwater5',
    'h_lighting1','h_lighting2','h_lighting3',
    'h_epower1','h_epower2','h_epower3',
    'h_cookingfuel1','h_cookingfuel2','h_cookingfuel3','h_cookingfuel4','h_cookingfuel5',
    'h_toiltype1','h_toiltype2','h_toiltype3','h_toiltype4','h_toiltype5','h_toiltype6',
    'h_septic1','h_septic2','h_septic3','h_septic4','h_septic5',
    'h_asset_lpg5kg','h_asset_fridge','h_asset_ac','h_asset_wheater','h_asset_phone',
    'h_asset_computer','h_asset_jewelry','h_asset_motorcycle','h_asset_boat','h_asset_motorboat',
    'h_asset_car','h_asset_tv','h_asset_smartphone','h_asset_land',
    'h_lnluaslantai','h_nfamily','h_notmarried','h_married','h_divorced','h_widowed'
]

needed_cols = predictors + ['R102','KAPITA','lpcexp','lfood','lnonfood']

# =========================
# 2. GARIS KEMISKINAN / DEFLATOR
# =========================
GK = {
    'jogja': {
        2022: {'urban': {'total':543821,'food':392290,'nonfood':151531}, 'rural': {'total':455583,'food':338551,'nonfood':117032}},
        2023: {'urban': {'total':598925,'food':431962,'nonfood':166963}, 'rural': {'total':499942,'food':366712,'nonfood':133230}},
        2024: {'urban': {'total':628626,'food':455545,'nonfood':173081}, 'rural': {'total':524188,'food':385903,'nonfood':138285}},
        2025: {'urban': {'total':653421,'food':474766,'nonfood':178655}, 'rural': {'total':545206,'food':406045,'nonfood':139161}},
    },
    'jateng': {
        2022: {'urban': {'total':442479,'food':328539,'nonfood':113940}, 'rural': {'total':434785,'food':330526,'nonfood':104259}},
        2023: {'urban': {'total':485511,'food':365201,'nonfood':120310}, 'rural': {'total':469003,'food':355699,'nonfood':113304}},
        2024: {'urban': {'total':517415,'food':390855,'nonfood':126560}, 'rural': {'total':495627,'food':375173,'nonfood':120454}},
        2025: {'urban': {'total':546576,'food':414287,'nonfood':132289}, 'rural': {'total':527934,'food':400113,'nonfood':127821}},
    }
}

def normalize_target_by_gk(df, year, prov):
    df = df.copy()
    area = np.where(df['R105'].eq(1), 'urban', 'rural')
    for target, comp in [('lpcexp','total'), ('lfood','food'), ('lnonfood','nonfood')]:
        gk_t = np.array([GK[prov][year][a][comp] for a in area])
        gk_25 = np.array([GK[prov][2025][a][comp] for a in area])
        df[target] = df[target] + np.log(gk_25 / gk_t)
    return df

def prepare_year(prov, year, filename):
    df = pd.read_csv(DATA_PATH / filename, sep=None, engine="python")
    df['food_pc'] = df['FOOD'] / df['h_hhcount']
    df['nonfood_pc'] = df['NONFOOD'] / df['h_hhcount']
    df['lfood'] = np.log(df['food_pc'])
    df['lnonfood'] = np.log(df['nonfood_pc'])
    df = normalize_target_by_gk(df, year, prov)
    return df

# =========================
# 3. METRIK
# =========================
def compute_decile_ranks(x):
    x = np.asarray(x)
    idx = np.argsort(x)
    ranks = np.zeros(len(x), dtype=int)
    size = len(x) // 10
    for i in range(10):
        s = i * size
        e = (i + 1) * size if i < 9 else len(x)
        ranks[idx[s:e]] = i
    return ranks

def targeting_error(actual, predicted):
    cm = confusion_matrix(actual, predicted, labels=[0,1])
    tp, fn = cm[0,0], cm[0,1]
    fp, tn = cm[1,0], cm[1,1]
    ie = fp / (tp + fp) if (tp + fp) else np.nan
    ee = fn / (tp + fn) if (tp + fn) else np.nan
    return ie, ee

def evaluate_prediction(test_df, pred):
    d = test_df.copy()
    d['pred'] = pred
    d['decile'] = compute_decile_ranks(d['KAPITA'])
    d['decile_pred'] = compute_decile_ranks(d['pred'])

    out = {
        'RMSE': np.sqrt(mean_squared_error(np.log(d['KAPITA']), np.log(d['pred']))),
        'taudcl': kendalltau(d['decile'], d['decile_pred']).correlation,
    }
    for pct, cutoff in [(20,1),(40,3),(60,5)]:
        actual = np.where(d['decile'] <= cutoff, 0, 1)
        predicted = np.where(d['decile_pred'] <= cutoff, 0, 1)
        ie, ee = targeting_error(actual, predicted)
        out[f'IE{pct}'] = ie
        out[f'EE{pct}'] = ee
        out[f'Acc{pct}'] = accuracy_score(actual, predicted)
        out[f'tauGroup{pct}'] = kendalltau(actual, predicted).correlation
    return out

# =========================
# 4. MODEL FACTORIES
# =========================
def new_xgb(**extra):
    return xgb.XGBRegressor(**XGB_PARAMS, **extra)

def new_rf():
    return RandomForestRegressor(**RF_PARAMS)

# =========================
# 5. TRAIN SATU KABUPATEN
# =========================
def train_district_models(tr, prov, kab):
    X = tr[predictors]
    Y = tr[['lfood','lnonfood']]
    models = {}

    # A. Baseline total
    models['Total_XGB'] = new_xgb().fit(X, tr['lpcexp'])

    # B. Single-output components -> nanti food + nonfood
    models['XGB_Components'] = {
        'food': new_xgb().fit(X, tr['lfood']),
        'nonfood': new_xgb().fit(X, tr['lnonfood'])
    }
    models['RF_Components'] = {
        'food': new_rf().fit(X, tr['lfood']),
        'nonfood': new_rf().fit(X, tr['lnonfood'])
    }

    # C. Problem transformation
    models['MultiOutputRegressor_XGB'] = MultiOutputRegressor(new_xgb()).fit(X, Y)
    models['MultiOutputRegressor_RF'] = MultiOutputRegressor(new_rf()).fit(X, Y)

    # D. Algorithm adaptation / native multi-output
    mot_params = XGB_PARAMS.copy()
    mot_params.update({'tree_method':'hist', 'multi_strategy':'multi_output_tree'})
    models['MultiOutputTree_XGB'] = xgb.XGBRegressor(**mot_params).fit(X, Y)
    models['Multivariate_RF'] = new_rf().fit(X, Y)

    # Simpan semua model kabupaten dalam SATU file
    bundle = {
        'province': prov,
        'district': kab,
        'train_years': TRAIN_YEARS,
        'predictors': predictors,
        'xgb_params': XGB_PARAMS,
        'rf_params': RF_PARAMS,
        'models': models,
    }
    joblib.dump(bundle, os.path.join(MODEL_PATH / f"models_{prov}_kab_{kab}.pkl"))
    return models

# =========================
# 6. PREDIKSI SEMUA MODEL
# =========================
def predict_all(models, test_df):
    X = test_df[predictors]
    pred = {}

    pred['Total_XGB'] = np.exp(models['Total_XGB'].predict(X))

    for name in ['XGB_Components','RF_Components']:
        food = np.exp(models[name]['food'].predict(X))
        nonfood = np.exp(models[name]['nonfood'].predict(X))
        pred[name] = food + nonfood

    for name in ['MultiOutputRegressor_XGB','MultiOutputRegressor_RF',
                 'MultiOutputTree_XGB','Multivariate_RF']:
        p = models[name].predict(X)
        pred[name] = np.exp(p[:,0]) + np.exp(p[:,1])

    return pred

# =========================
# 7. MAIN: TRAIN 2022-2024, TEST 2025
# =========================
results = []
predictions_2025 = []

for prov, files in DATASETS.items():
    print(f'\\n===== {prov.upper()} =====')

    data = {y: prepare_year(prov, y, f) for y, f in files.items()}
    train_df = pd.concat([data[y][needed_cols] for y in TRAIN_YEARS], ignore_index=True)
    test_df = data[TEST_YEAR][needed_cols].copy()

    for kab in sorted(train_df['R102'].dropna().unique()):
        tr = train_df[train_df['R102'].eq(kab)].copy()
        ts = test_df[test_df['R102'].eq(kab)].copy()
        if ts.empty or len(tr) < 30:
            continue

        print(f'Kab {kab}: train={len(tr)}, test={len(ts)}')
        models = train_district_models(tr, prov, kab)
        all_pred = predict_all(models, ts)

        for model_name, pred in all_pred.items():
            met = evaluate_prediction(ts, pred)
            results.append({
                'province': prov, 'district': kab,
                'train': str(TRAIN_YEARS), 'test': TEST_YEAR,
                'model': model_name, 'train_size': len(tr), 'test_size': len(ts),
                **met
            })
            predictions_2025.append(pd.DataFrame({
                'province': prov, 'district': kab,
                'model': model_name,
                'KAPITA': ts['KAPITA'].to_numpy(),
                'pred': pred
            }))

# =========================
# 8. OUTPUT
# =========================
df_perf = pd.DataFrame(results)
df_pred = pd.concat(predictions_2025, ignore_index=True)

metric_order = [
    'province','district','train','test','model','RMSE',
    'IE20','EE20','IE40','EE40','IE60','EE60',
    'Acc20','Acc40','Acc60','taudcl',
    'tauGroup20','tauGroup40','tauGroup60','train_size','test_size'
]
df_perf = df_perf[metric_order]

df_perf.to_excel(os.path.join(OUTPUT_PATH, 'hasil_model_fixed_2025.xlsx'), index=False)
df_pred.to_excel(os.path.join(OUTPUT_PATH, 'prediksi_per_ruta_2025.xlsx'), index=False)

summary = (df_perf.groupby(['province','model'], as_index=False)
           .agg(mean_RMSE=('RMSE','mean'), mean_IE40=('IE40','mean'),
                mean_EE40=('EE40','mean'), mean_tau=('taudcl','mean'),
                n_kabupaten=('district','nunique'))
           .sort_values(['province','mean_RMSE']))
summary.to_excel(os.path.join(OUTPUT_PATH, 'summary_model_fixed_2025.xlsx'), index=False)

print('\\nSELESAI. Model tersimpan di:', MODEL_PATH)
print(summary.to_string(index=False))
