import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Concatenate, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

MAX_LAG       = 16      # jendela lag maksimum (K = 16 ~ 8 jam @ 30 menit)
N_REGIONS     = 6
T_TEST        = 240     # jumlah observasi uji per region per periode

N_GRU_LAYERS  = 2       # Tabel 6
HIDDEN_UNITS  = 128     # Tabel 6  (d)
DROPOUT       = 0.2     # Tabel 6  (p)
LEARNING_RATE = 1e-3    # Tabel 6
BATCH_SIZE    = 32      # Tabel 6
MAX_EPOCHS    = 200     # Tabel 6
PATIENCE      = 15      # Tabel 6
VAL_SPLIT     = 0.10    # 10% terakhir (kronologis) untuk early stopping

CONFIGS_TO_RUN = ["SW", "PACF", "PLCMF"]

OUTPUT_DIR = "hasil_MIMO_GRU_jurnal"
REGIONS    = [f"region_{i}" for i in range(1, 7)]

PERIOD_CONFIG = {
    "period_1": {"train_csv": "trainingpagi.csv",  "test_csv": "testingpagi.csv",  "label": "Period 1"},
    "period_2": {"train_csv": "trainingsiang.csv", "test_csv": "testingsiang.csv", "label": "Period 2"},
    "period_3": {"train_csv": "trainingmalam.csv", "test_csv": "testingmalam.csv", "label": "Period 3"},
}

FEATURE_CONFIG = {
    "period_1": {
        "region_1": {"own_lag": [1,2,3,4,16,17], "cross_lag": {
            "region_2": [1,2,3,4,5,9,10,11,13,14],
            "region_3": [1,2,3,4,7,10,11,12,15,16],
            "region_4": [1,2,4,5,6,7,9,10,12,13,14,15,16],
            "region_5": [1,3,4,6,7,8,14,15],
            "region_6": [1,2,3,4,5,6,9,10,12,13,14,15]}},
        "region_2": {"own_lag": [1,4], "cross_lag": {
            "region_1": [1,2,3,4,5,6,9,13,14,15,16],
            "region_3": [1,2,3,4,5,6,7,9,10,11,14,15,16],
            "region_4": [1,2,3,4,5,6,7,8,9,11,12,13,14,15],
            "region_5": [1,2,3,4,6,9,10,11,14,15,16],
            "region_6": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}},
        "region_3": {"own_lag": [1,5,6,16], "cross_lag": {
            "region_1": [1,2,3,4,5,6,7,8,9,10,11,12,15,16],
            "region_2": [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16],
            "region_4": [1,2,4,5,6,7,8,9,10,11,12,13,14,15],
            "region_5": [2,3,4,5,6,11,12,13,14,16],
            "region_6": [2,3,4,6,7,8,9,10,11,12,13,14,15,16]}},
        "region_4": {"own_lag": [1,2,5,6,10,17], "cross_lag": {
            "region_1": [1,2,3,4,5,7,8,9,10,12,13,14,15,16],
            "region_2": [1,2,3,5,6,7,8,9,10,11,14],
            "region_3": [1,3,4,7,8,9,10,12,13,15,16],
            "region_5": [1,2,3,4,5,6,7,11,12,13,14,16],
            "region_6": [1,2,3,4,7,8,9,10,11,12,13,14,15,16]}},
        "region_5": {"own_lag": [1,2,16], "cross_lag": {
            "region_1": [1,2,4,5,7,8,9,10,11,12,15,16],
            "region_2": [1,2,3,4,5,6,8,9,10,11,12,14,15,16],
            "region_3": [1,2,3,4,5,7,8,9,10,11,13,14,15],
            "region_4": [1,2,3,5,6,9,10,12,13,14,15,16],
            "region_6": [1,2,3,4,5,6,7,8,9,10,12,13,14,15]}},
        "region_6": {"own_lag": [1,4,12,18,20], "cross_lag": {
            "region_1": [1,2,3,4,5,6,8,9,10,11,12,13,14,15,16],
            "region_2": [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16],
            "region_3": [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16],
            "region_4": [1,2,3,5,7,9,10,12,13,14,15,16],
            "region_5": [2,3,4,5,6,8,9,13,14,16]}},
    },
    "period_2": {
        "region_1": {"own_lag": [1,7,9,10,16,17,23,25,26,96,112], "cross_lag": {
            "region_2": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
            "region_3": [1,2,3,4,7,9,14,16],
            "region_4": [2,3,6,7,10,11,13,14,15],
            "region_5": [1,3,4,5,6,7,8,9,14,16],
            "region_6": [1,2,3,4,5,6,8,9,11,12,13,14,15,16]}},
        "region_2": {"own_lag": [2,9,112], "cross_lag": {
            "region_1": [1,2,3,4,5,7,8,9,10,11,12,13,14,16],
            "region_3": [1,2,3,4,6,7,8,9,10,11,12,13,14],
            "region_4": [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16],
            "region_5": [6,7,8,9,10,12,13,14,16],
            "region_6": [2,3,4,5,6,7,8,9,11,12,13,14,16]}},
        "region_3": {"own_lag": [1,6,7,9,10,16], "cross_lag": {
            "region_1": [1,2,4,5,6,7,8,9,10,11,13,15,16],
            "region_2": [1,2,3,4,5,6,7,8,9,10,11,12,14,15],
            "region_4": [1,2,3,4,6,8,9,11,13,14,15,16],
            "region_5": [2,3,6,7,9,10,11,14,16],
            "region_6": [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]}},
        "region_4": {"own_lag": [1,2,7,16,17,18], "cross_lag": {
            "region_1": [2,4,5,7,8,9,10,11,15,16],
            "region_2": [1,2,5,6,7,8,9,10,11,14,15,16],
            "region_3": [1,2,3,4,6,7,9,12,13,14,15,16],
            "region_5": [3,4,5,6,7,9,10,13,14,16],
            "region_6": [1,2,4,5,6,7,8,9,10,12,13,14,15,16]}},
        "region_5": {"own_lag": [1,6,23,26], "cross_lag": {
            "region_1": [2,5,7,8,9,10,11,16],
            "region_2": [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16],
            "region_3": [1,3,4,5,6,7,9,12,14],
            "region_4": [1,2,3,7,8,9,11,13,14,15,16],
            "region_6": [2,3,4,6,7,8,9,10,11,12,13,14,15,16]}},
        "region_6": {"own_lag": [1,10,14], "cross_lag": {
            "region_1": [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16],
            "region_2": [1,2,4,5,6,7,8,9,10,11,13,14,15,16],
            "region_3": [2,3,4,5,7,8,9,10,13,14,16],
            "region_4": [2,3,4,7,8,9,11,13,14,15,16],
            "region_5": [1,4,5,6,7,9,10,13,14,16]}},
    },
    "period_3": {
        "region_1": {"own_lag": [1,4,5,15,16,17,18], "cross_lag": {
            "region_2": [1,2,4,6,8,9,10,11,12,13,14,15,16],
            "region_3": [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16],
            "region_4": [1,2,3,4,5,6,7,8,9,10,11,12,13,15],
            "region_5": [1,3,7,9,10],
            "region_6": [1,2,4,6,7,8,9,10,12,13,14,15,16]}},
        "region_2": {"own_lag": [1,5,6,10], "cross_lag": {
            "region_1": [1,2,3,4,5,7,8,9,10,11,12,13,14,15,16],
            "region_3": [1,2,3,4,5,6,7,9,11,12,13,14,15,16],
            "region_4": [1,2,3,4,6,7,8,9,10,11,12,13,14,15],
            "region_5": [2,3,4,5,6,8,9,10,11,12,13,16],
            "region_6": [1,2,3,4,5,6,7,9,10,11,12,13,14,16]}},
        "region_3": {"own_lag": [1,2,16], "cross_lag": {
            "region_1": [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16],
            "region_2": [1,2,4,5,6,7,9,10,12,15],
            "region_4": [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16],
            "region_5": [1,2,3,4,5,6,8,9,10,11,12,13,14,16],
            "region_6": [1,2,3,4,5,6,7,8,10,12,13,14,15,16]}},
        "region_4": {"own_lag": [1,2,3,4,5,6,15,16,17,19], "cross_lag": {
            "region_1": [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16],
            "region_2": [1,2,3,4,5,6,7,8,10,11,12,15],
            "region_3": [1,2,3,5,7,8,9,10,11,12,13,14,15,16],
            "region_5": [2,3,6,9,10,11,12,13,14,15,16],
            "region_6": [1,2,3,4,5,6,7,8,9,10,12,13,14,15,16]}},
        "region_5": {"own_lag": [1,2,3], "cross_lag": {
            "region_1": [1,2,3,4,5,7,8,10,11,12,13,14,15,16],
            "region_2": [1,2,3,5,6,7,9,11,12,15],
            "region_3": [1,2,3,5,6,10,11,12,13,14,15],
            "region_4": [1,2,3,4,5,6,7,9,10,12,13,14,16],
            "region_6": [1,2,3,4,5,8,9,10,11,12,13,14,15,16]}},
        "region_6": {"own_lag": [1,2,4,16,17], "cross_lag": {
            "region_1": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
            "region_2": [1,2,4,5,6,7,9,10,11,12,15],
            "region_3": [1,3,4,5,6,7,9,11,12,13,14,15,16],
            "region_4": [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16],
            "region_5": [2,3,4,5,6,8,9,10,11,12,13,14,16]}},
    },
}


# =========================================================================
# DATA LOADING
# =========================================================================
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    rename_map = {f"region-{i}": f"region_{i}" for i in range(1, 7)}
    df = df.rename(columns=rename_map)
    return df.dropna().reset_index(drop=True)


def build_masks(feature_cfg: dict, mode: str) -> list:
    masks = []
    for region in REGIONS:
        M = np.zeros((MAX_LAG, N_REGIONS), dtype="float32")
        r_idx = REGIONS.index(region)

        if mode == "SW":
            M[:, :] = 1.0

        elif mode == "PACF":
            for lag in feature_cfg[region]["own_lag"]:
                if 1 <= lag <= MAX_LAG:
                    M[MAX_LAG - lag, r_idx] = 1.0

        elif mode == "PLCMF":
            for lag in feature_cfg[region]["own_lag"]:
                if 1 <= lag <= MAX_LAG:
                    M[MAX_LAG - lag, r_idx] = 1.0
            for src, lags in feature_cfg[region]["cross_lag"].items():
                if src not in REGIONS:
                    continue
                s_idx = REGIONS.index(src)
                for lag in lags:
                    if 1 <= lag <= MAX_LAG:
                        M[MAX_LAG - lag, s_idx] = 1.0
        else:
            raise ValueError(f"Mode tidak dikenal: {mode}")

        masks.append(M)
    return masks


# =========================================================================
# PEMBENTUKAN SAMPEL ONE-STEP-AHEAD
#   X[i] = riwayat 16 langkah (tertua -> terbaru), Y[i] = 6 region di t
# =========================================================================
def make_samples(scaled: np.ndarray):
    n = len(scaled)
    n_samples = n - MAX_LAG
    if n_samples <= 0:
        raise ValueError(f"Data terlalu pendek: {n}")

    X = np.zeros((n_samples, MAX_LAG, N_REGIONS), dtype="float32")
    Y = np.zeros((n_samples, N_REGIONS),         dtype="float32")
    for i in range(n_samples):
        t = i + MAX_LAG
        X[i] = scaled[t - MAX_LAG:t]   # baris 0 = t-16 (lag 16) ... baris 15 = t-1 (lag 1)
        Y[i] = scaled[t]
    return X, Y


# =========================================================================
# LAYER MASK KONSTAN (non-trainable)
# =========================================================================
class FeatureMask(Layer):
   
    def __init__(self, mask, **kwargs):
        super().__init__(**kwargs)
        self.mask_np = np.asarray(mask, dtype="float32")

    def build(self, input_shape):
        self.mask_w = self.add_weight(
            name="mask",
            shape=self.mask_np.shape,
            initializer=tf.constant_initializer(self.mask_np),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.mask_w

def build_mimo_model(masks: list) -> Model:
    inp = Input(shape=(MAX_LAG, N_REGIONS), name="history")

    region_outputs = []
    for i, region in enumerate(REGIONS):
        x = FeatureMask(masks[i], name=f"mask_{region}")(inp)
        x = GRU(HIDDEN_UNITS, return_sequences=True,  name=f"gru1_{region}")(x)
        x = Dropout(DROPOUT, name=f"drop1_{region}")(x)
        x = GRU(HIDDEN_UNITS, return_sequences=False, name=f"gru2_{region}")(x)
        x = Dropout(DROPOUT, name=f"drop2_{region}")(x)
        region_outputs.append(
            Dense(1, activation="linear", name=f"out_{region}")(x)
        )

    output = Concatenate(name="mimo_output")(region_outputs)   # (batch, 6)

    model = Model(inputs=inp, outputs=output, name="MIMO_GRU")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model

def rolling_one_step(model: Model, train_scaled: np.ndarray,
                     test_scaled: np.ndarray) -> np.ndarray:
    combined = np.vstack([train_scaled, test_scaled]).astype("float32")
    n_train  = len(train_scaled)
    n_test   = len(test_scaled)

    X = np.zeros((n_test, MAX_LAG, N_REGIONS), dtype="float32")
    for tt in range(n_test):
        t = n_train + tt
        X[tt] = combined[t - MAX_LAG:t]

    return model.predict(X, batch_size=256, verbose=0)   # (n_test, 6) skala [0,1]

def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calc_mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def calc_smape(y_true, y_pred):
    return float(np.mean(2 * np.abs(y_true - y_pred) /
                         (np.abs(y_true) + np.abs(y_pred))) * 100)

def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = []
    test_plot   = {}   # (mode, period, region) -> (y_true, y_pred)

    for mode in CONFIGS_TO_RUN:
        print(f"\n{'#'*70}\n# KONFIGURASI: GRU-{mode}\n{'#'*70}")

        for period_key, cfg in PERIOD_CONFIG.items():
            print(f"\n=== {cfg['label']} | GRU-{mode} ===")

            train_df  = load_data(cfg["train_csv"])
            test_df   = load_data(cfg["test_csv"])
            train_arr = train_df[REGIONS].values.astype("float32")
            test_arr  = test_df[REGIONS].values.astype("float32")

            # --- Min-Max [0,1], fit pada training, terapkan ke test ---
            scaler   = MinMaxScaler().fit(train_arr)
            train_sc = scaler.transform(train_arr).astype("float32")
            test_sc  = scaler.transform(test_arr).astype("float32")

            # --- mask seleksi fitur ---
            masks    = build_masks(FEATURE_CONFIG[period_key], mode)
            n_active = int(sum(int(m.sum()) for m in masks))
            print(f"  Fitur aktif (Sigma mask 6 region): {n_active}")

            # --- sampel one-step-ahead + split validasi kronologis ---
            X, Y  = make_samples(train_sc)
            split = int(len(X) * (1.0 - VAL_SPLIT))
            X_tr, X_val = X[:split], X[split:]
            Y_tr, Y_val = Y[:split], Y[split:]
            print(f"  Sampel latih: {len(X_tr)} | validasi: {len(X_val)}")

            # --- bangun & latih model MIMO ---
            tf.keras.backend.clear_session()
            model = build_mimo_model(masks)
            es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                               restore_best_weights=True, verbose=1)
            t0 = time.time()
            model.fit(
                X_tr, Y_tr,
                validation_data=(X_val, Y_val),
                epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                callbacks=[es], verbose=0, shuffle=False,
            )
            print(f"  Training selesai dalam {time.time() - t0:.1f}s")

            # --- metrik training (one-step in-sample) ---
            pred_tr = scaler.inverse_transform(
                model.predict(X, batch_size=256, verbose=0))
            true_tr = scaler.inverse_transform(Y)

            # --- metrik testing (rolling one-step) ---
            pred_te = scaler.inverse_transform(
                rolling_one_step(model, train_sc, test_sc))
            true_te = test_arr[:T_TEST]

            for j, region in enumerate(REGIONS):
                all_metrics.append({
                    "config": mode, "period": period_key, "region": region,
                    "split": "train",
                    "RMSE": round(calc_rmse(true_tr[:, j], pred_tr[:, j]), 4),
                    "MAPE": round(calc_mape(true_tr[:, j], pred_tr[:, j]), 4),
                    "SMAPE": round(calc_smape(true_tr[:, j], pred_tr[:, j]), 4),
                })

                r_rmse  = calc_rmse(true_te[:, j], pred_te[:, j])
                r_mape  = calc_mape(true_te[:, j], pred_te[:, j])
                r_smape = calc_smape(true_te[:, j], pred_te[:, j])
                all_metrics.append({
                    "config": mode, "period": period_key, "region": region,
                    "split": "test",
                    "RMSE": round(r_rmse, 4), "MAPE": round(r_mape, 4),
                    "SMAPE": round(r_smape, 4),
                })
                test_plot[(mode, period_key, region)] = (true_te[:, j], pred_te[:, j])
                print(f"  [TEST] {region}: RMSE={r_rmse:8.2f}  "
                      f"MAPE={r_mape:6.3f}%  SMAPE={r_smape:6.3f}%")

                pd.DataFrame({
                    "step": np.arange(1, T_TEST + 1),
                    "actual": true_te[:, j], "predicted": pred_te[:, j],
                }).to_csv(
                    os.path.join(OUTPUT_DIR,
                                 f"forecast_{mode}_{period_key}_{region}.csv"),
                    index=False,
                )

    # --- simpan & ringkas ---
    mdf = pd.DataFrame(all_metrics)
    mdf.to_csv(os.path.join(OUTPUT_DIR, "all_metrics.csv"), index=False)
    _print_summary(mdf)
    _plot_test(test_plot)
    print(f"\nOutput tersimpan di: {OUTPUT_DIR}/")
    print("SELESAI")

def _print_summary(mdf: pd.DataFrame):
    te = mdf[mdf.split == "test"]

    print(f"\n{'='*70}\nRINGKASAN METRIK UJI (rata-rata per konfigurasi/periode)\n{'='*70}")
    for mode in CONFIGS_TO_RUN:
        for pk in PERIOD_CONFIG:
            sub = te[(te.config == mode) & (te.period == pk)]
            print(f"  GRU-{mode:6s} {pk}: "
                  f"MAPE={sub.MAPE.mean():6.3f}%  "
                  f"SMAPE={sub.SMAPE.mean():6.3f}%  "
                  f"RMSE={sub.RMSE.mean():8.2f}")
        overall = te[te.config == mode]
        print(f"  GRU-{mode:6s} {'OVERALL':9s}: MAPE={overall.MAPE.mean():6.3f}%  "
              f"RMSE={overall.RMSE.mean():8.2f}\n")

    if len(CONFIGS_TO_RUN) > 1:
        print(f"{'-'*70}\nJumlah kemenangan (RMSE minimum) per konfigurasi\n"
              f"[per {len(PERIOD_CONFIG)*len(REGIONS)} skenario = 6 region x 3 periode, satu window]\n{'-'*70}")
        wins = {m: 0 for m in CONFIGS_TO_RUN}
        for pk in PERIOD_CONFIG:
            for region in REGIONS:
                best, best_val = None, np.inf
                for m in CONFIGS_TO_RUN:
                    v = te[(te.config == m) & (te.period == pk) &
                           (te.region == region)].RMSE.values
                    if len(v) and v[0] < best_val:
                        best_val, best = v[0], m
                if best is not None:
                    wins[best] += 1
        total = len(PERIOD_CONFIG) * len(REGIONS)
        for m in CONFIGS_TO_RUN:
            print(f"  GRU-{m:6s}: {wins[m]}/{total}")


# =========================================================================
# PLOT
# =========================================================================
def _plot_test(test_plot: dict):
    for mode in CONFIGS_TO_RUN:
        fig, axes = plt.subplots(3, 6, figsize=(30, 12))
        for i, pk in enumerate(PERIOD_CONFIG):
            for j, region in enumerate(REGIONS):
                ax = axes[i, j]
                y_true, y_pred = test_plot[(mode, pk, region)]
                ax.plot(y_true, lw=1.1, color="#2563EB", label="Actual")
                ax.plot(y_pred, lw=1.1, color="#DC2626", ls="--", label="Forecast")
                ax.set_title(
                    f"{PERIOD_CONFIG[pk]['label']} | {region}\n"
                    f"MAPE={calc_mape(y_true, y_pred):.2f}%", fontsize=9)
                ax.grid(True, alpha=0.3)
                if i == 2: ax.set_xlabel("Test step (1-240)", fontsize=8)
                if j == 0: ax.set_ylabel("Load (MW)", fontsize=8)
                if i == 0 and j == 0: ax.legend(fontsize=8)
        fig.suptitle(f"GRU-{mode}: Rolling one-step-ahead test forecast (240 titik)",
                     fontsize=14, y=1.01)
        fig.tight_layout()
        fn = os.path.join(OUTPUT_DIR, f"plot_test_{mode}.png")
        fig.savefig(fn, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot: {fn}")


# =========================================================================
if __name__ == "__main__":
    run_pipeline()
