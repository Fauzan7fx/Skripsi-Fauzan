# ============================================================
# OOF RESIDUAL FINAL - XGBOOST + RANDOM FOREST
# Disesuaikan dengan PMT_FIXED_TRAIN_AND_TEST.py
# Training: 2022-2024
# OOF: 3-fold per kabupaten/kota
# Residual dihitung dari model single-output food dan nonfood
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# =========================
# 1. KONFIGURASI
# =========================

BASE_PATH = Path(__file__).resolve().parent
DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH = BASE_PATH / "output" / "oof_residual"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 123
TRAIN_YEARS = [2022, 2023, 2024]

DATASETS = {
    "jogja": {
        2022: "susenas_jogja_2022.csv",
        2023: "susenas_jogja_2023.csv",
        2024: "susenas_jogja_2024.csv",
    },
    "jateng": {
        2022: "susenas_jateng_2022.csv",
        2023: "susenas_jateng_2023.csv",
        2024: "susenas_jateng_2024.csv",
    },
}

PROV_LABEL = {
    "jogja": "D.I. Yogyakarta",
    "jateng": "Jawa Tengah",
}

# Fixed parameter sama dengan script final
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "colsample_bytree": 0.3,
    "learning_rate": 0.01,
    "max_depth": 5,
    "n_estimators": 700,
    "reg_lambda": 0.01,
    "subsample": 0.7,
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "min_samples_split": 3,
    "min_samples_leaf": 3,
    "bootstrap": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
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

# =========================
# 2. GARIS KEMISKINAN / DEFLATOR
# Sama dengan script final
# =========================

GK = {
    'jogja': {
        2022: {'urban': {'total':543821,'food':392290,'nonfood':151531},
               'rural': {'total':455583,'food':338551,'nonfood':117032}},
        2023: {'urban': {'total':598925,'food':431962,'nonfood':166963},
               'rural': {'total':499942,'food':366712,'nonfood':133230}},
        2024: {'urban': {'total':628626,'food':455545,'nonfood':173081},
               'rural': {'total':524188,'food':385903,'nonfood':138285}},
        2025: {'urban': {'total':653421,'food':474766,'nonfood':178655},
               'rural': {'total':545206,'food':406045,'nonfood':139161}},
    },
    'jateng': {
        2022: {'urban': {'total':442479,'food':328539,'nonfood':113940},
               'rural': {'total':434785,'food':330526,'nonfood':104259}},
        2023: {'urban': {'total':485511,'food':365201,'nonfood':120310},
               'rural': {'total':469003,'food':355699,'nonfood':113304}},
        2024: {'urban': {'total':517415,'food':390855,'nonfood':126560},
               'rural': {'total':495627,'food':375173,'nonfood':120454}},
        2025: {'urban': {'total':546576,'food':414287,'nonfood':132289},
               'rural': {'total':527934,'food':400113,'nonfood':127821}},
    }
}


def normalize_target_by_gk(df, year, prov):
    df = df.copy()
    area = np.where(df["R105"].eq(1), "urban", "rural")

    for target, comp in [
        ("lpcexp", "total"),
        ("lfood", "food"),
        ("lnonfood", "nonfood"),
    ]:
        gk_t = np.array([GK[prov][year][a][comp] for a in area])
        gk_25 = np.array([GK[prov][2025][a][comp] for a in area])
        df[target] = df[target] + np.log(gk_25 / gk_t)

    return df


def prepare_year(prov, year, filename):
    path = DATA_PATH / filename

    df = pd.read_csv(
        path,
        sep=None,
        engine="python"
    )

    df["food_pc"] = df["FOOD"] / df["h_hhcount"]
    df["nonfood_pc"] = df["NONFOOD"] / df["h_hhcount"]

    df["lfood"] = np.log(df["food_pc"])
    df["lnonfood"] = np.log(df["nonfood_pc"])

    df = normalize_target_by_gk(df, year, prov)
    df["year"] = year

    return df


# =========================
# 3. OOF PER KABUPATEN/KOTA
# =========================

def run_oof_district(tr, prov, kab):
    tr = tr.copy().reset_index(drop=True)

    valid_target = (
        np.isfinite(tr["lfood"])
        & np.isfinite(tr["lnonfood"])
    )
    tr = tr.loc[valid_target].reset_index(drop=True)

    n = len(tr)

    if n < 30:
        return None, None

    X = tr[predictors]
    y_food = tr["lfood"].to_numpy()
    y_nonfood = tr["lnonfood"].to_numpy()

    pred_xgb_food = np.full(n, np.nan)
    pred_xgb_nonfood = np.full(n, np.nan)
    pred_rf_food = np.full(n, np.nan)
    pred_rf_nonfood = np.full(n, np.nan)

    fold_no = np.zeros(n, dtype=int)

    kf = KFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    for fold, (idx_train, idx_valid) in enumerate(kf.split(X), start=1):

        # -------------------------
        # XGBOOST
        # -------------------------
        xgb_food = xgb.XGBRegressor(**XGB_PARAMS)
        xgb_nonfood = xgb.XGBRegressor(**XGB_PARAMS)

        xgb_food.fit(
            X.iloc[idx_train],
            y_food[idx_train]
        )
        xgb_nonfood.fit(
            X.iloc[idx_train],
            y_nonfood[idx_train]
        )

        pred_xgb_food[idx_valid] = xgb_food.predict(
            X.iloc[idx_valid]
        )
        pred_xgb_nonfood[idx_valid] = xgb_nonfood.predict(
            X.iloc[idx_valid]
        )

        # -------------------------
        # RANDOM FOREST
        # -------------------------
        rf_food = RandomForestRegressor(**RF_PARAMS)
        rf_nonfood = RandomForestRegressor(**RF_PARAMS)

        rf_food.fit(
            X.iloc[idx_train],
            y_food[idx_train]
        )
        rf_nonfood.fit(
            X.iloc[idx_train],
            y_nonfood[idx_train]
        )

        pred_rf_food[idx_valid] = rf_food.predict(
            X.iloc[idx_valid]
        )
        pred_rf_nonfood[idx_valid] = rf_nonfood.predict(
            X.iloc[idx_valid]
        )

        fold_no[idx_valid] = fold

    # Residual XGB
    resid_xgb_food = y_food - pred_xgb_food
    resid_xgb_nonfood = y_nonfood - pred_xgb_nonfood

    # Residual RF
    resid_rf_food = y_food - pred_rf_food
    resid_rf_nonfood = y_nonfood - pred_rf_nonfood

    # Korelasi aktual target
    corr_actual = np.corrcoef(
        y_food,
        y_nonfood
    )[0, 1]

    # Korelasi residual OOF
    corr_resid_xgb = np.corrcoef(
        resid_xgb_food,
        resid_xgb_nonfood
    )[0, 1]

    corr_resid_rf = np.corrcoef(
        resid_rf_food,
        resid_rf_nonfood
    )[0, 1]

    summary = {
        "province": prov,
        "Provinsi": PROV_LABEL[prov],
        "district": kab,
        "N": n,
        "corr_actual_lfood_lnonfood": corr_actual,
        "corr_residual_oof_xgb": corr_resid_xgb,
        "corr_residual_oof_rf": corr_resid_rf,
    }

    detail = pd.DataFrame({
        "province": prov,
        "Provinsi": PROV_LABEL[prov],
        "district": kab,
        "year": tr["year"].to_numpy(),
        "fold": fold_no,

        "actual_lfood": y_food,
        "actual_lnonfood": y_nonfood,

        "pred_oof_xgb_lfood": pred_xgb_food,
        "pred_oof_xgb_lnonfood": pred_xgb_nonfood,
        "residual_xgb_food": resid_xgb_food,
        "residual_xgb_nonfood": resid_xgb_nonfood,

        "pred_oof_rf_lfood": pred_rf_food,
        "pred_oof_rf_lnonfood": pred_rf_nonfood,
        "residual_rf_food": resid_rf_food,
        "residual_rf_nonfood": resid_rf_nonfood,
    })

    return summary, detail


# =========================
# 4. MAIN
# =========================

summary_rows = []
detail_rows = []

for prov, files in DATASETS.items():

    print(f"\n===== {PROV_LABEL[prov]} =====")

    data = {
        year: prepare_year(prov, year, filename)
        for year, filename in files.items()
    }

    train_df = pd.concat(
        [data[year] for year in TRAIN_YEARS],
        ignore_index=True
    )

    required_cols = predictors + [
        "R102",
        "lfood",
        "lnonfood",
        "year"
    ]

    missing_cols = [
        col for col in required_cols
        if col not in train_df.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Kolom tidak ditemukan untuk {prov}: {missing_cols}"
        )

    for kab in sorted(
        train_df["R102"].dropna().unique()
    ):

        tr = train_df[
            train_df["R102"].eq(kab)
        ].copy()

        if len(tr) < 30:
            print(
                f"Skip {kab}: train < 30"
            )
            continue

        summary, detail = run_oof_district(
            tr,
            prov,
            kab
        )

        if summary is None:
            continue

        summary_rows.append(summary)
        detail_rows.append(detail)

        print(
            f"Kab {kab}: "
            f"N={summary['N']}, "
            f"corr aktual={summary['corr_actual_lfood_lnonfood']:.4f}, "
            f"OOF XGB={summary['corr_residual_oof_xgb']:.4f}, "
            f"OOF RF={summary['corr_residual_oof_rf']:.4f}"
        )


summary_oof = pd.DataFrame(summary_rows)

if not detail_rows:
    raise RuntimeError("Tidak ada hasil OOF yang berhasil dihitung.")

detail_oof = pd.concat(
    detail_rows,
    ignore_index=True
)


# =========================
# 5. RINGKASAN PROVINSI
# =========================

summary_province = (
    summary_oof
    .groupby(
        ["province", "Provinsi"],
        as_index=False
    )
    .agg(
        n_kabupaten=(
            "district",
            "nunique"
        ),

        mean_corr_actual=(
            "corr_actual_lfood_lnonfood",
            "mean"
        ),
        median_corr_actual=(
            "corr_actual_lfood_lnonfood",
            "median"
        ),

        mean_corr_residual_oof_xgb=(
            "corr_residual_oof_xgb",
            "mean"
        ),
        median_corr_residual_oof_xgb=(
            "corr_residual_oof_xgb",
            "median"
        ),
        min_corr_residual_oof_xgb=(
            "corr_residual_oof_xgb",
            "min"
        ),
        max_corr_residual_oof_xgb=(
            "corr_residual_oof_xgb",
            "max"
        ),

        mean_corr_residual_oof_rf=(
            "corr_residual_oof_rf",
            "mean"
        ),
        median_corr_residual_oof_rf=(
            "corr_residual_oof_rf",
            "median"
        ),
        min_corr_residual_oof_rf=(
            "corr_residual_oof_rf",
            "min"
        ),
        max_corr_residual_oof_rf=(
            "corr_residual_oof_rf",
            "max"
        ),
    )
)


# =========================
# 6. POOLED PER PROVINSI
# Tambahan diagnostik, bukan hasil utama
# =========================

pooled_rows = []

for prov in DATASETS:

    d = detail_oof[
        detail_oof["province"].eq(prov)
    ]

    pooled_rows.append({
        "province": prov,
        "Provinsi": PROV_LABEL[prov],
        "N_ruta": len(d),

        "pooled_corr_actual":
            d["actual_lfood"].corr(
                d["actual_lnonfood"]
            ),

        "pooled_corr_residual_oof_xgb":
            d["residual_xgb_food"].corr(
                d["residual_xgb_nonfood"]
            ),

        "pooled_corr_residual_oof_rf":
            d["residual_rf_food"].corr(
                d["residual_rf_nonfood"]
            ),
    })

pooled_province = pd.DataFrame(
    pooled_rows
)


# =========================
# 7. SAVE
# =========================

output_file = (
    OUTPUT_PATH
    / "hasil_OOF_residual_XGB_RF_final.xlsx"
)

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    summary_oof.round(6).to_excel(
        writer,
        sheet_name="per_kabupaten",
        index=False
    )

    summary_province.round(6).to_excel(
        writer,
        sheet_name="ringkasan_provinsi",
        index=False
    )

    pooled_province.round(6).to_excel(
        writer,
        sheet_name="pooled_provinsi",
        index=False
    )

    detail_oof.round(6).to_excel(
        writer,
        sheet_name="residual_per_ruta",
        index=False
    )


print("\n===== SELESAI =====")
print("Output:", output_file)

print("\nRINGKASAN PROVINSI")
print(
    summary_province
    .round(4)
    .to_string(index=False)
)

print("\nKORELASI POOLED PROVINSI")
print(
    pooled_province
    .round(4)
    .to_string(index=False)
)
