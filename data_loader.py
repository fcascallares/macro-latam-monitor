"""Data loader — reads the Monitor.xlsx and returns structured DataFrames per country."""

import pandas as pd
import numpy as np
from pathlib import Path

EXCEL_PATH = Path(__file__).parent / "Monitor.xlsx"

FULL_COUNTRIES = ["BRA", "CHL", "MEX"]
SIMPLE_COUNTRIES = ["COL", "ARG", "PER"]
ALL_COUNTRIES = FULL_COUNTRIES + SIMPLE_COUNTRIES

COUNTRY_NAMES = {"BRA": "Brasil", "CHL": "Chile", "MEX": "México", "COL": "Colombia", "ARG": "Argentina", "PER": "Perú"}
COUNTRY_FLAGS = {"BRA": "🇧🇷", "CHL": "🇨🇱", "MEX": "🇲🇽", "COL": "🇨🇴", "ARG": "🇦🇷", "PER": "🇵🇪"}
COUNTRY_ETF = {"BRA": "EWZ", "CHL": "ECH", "MEX": "EWW", "COL": "GXG", "ARG": "ARGT", "PER": "EPU"}

FULL_VARIABLES = ["Actividad", "Crédito", "Inflación", "Tasa de política", "Desempleo", "TCRM", "Resultado primario", "Resultado financiero"]
SIMPLE_VARIABLES = ["Actividad", "Crédito", "Inflación", "Tasa de política", "Desempleo"]

VARIABLE_COLORS = {
    "Actividad": "#3D85C6", "Crédito": "#6FA8DC", "Inflación": "#E06666",
    "Tasa de política": "#F6B26B", "Desempleo": "#8E7CC3", "TCRM": "#76A5AF",
    "Resultado primario": "#93C47D", "Resultado financiero": "#6AA84F",
}

FULL_COL_MAP = {
    "fecha": 0,
    "Actividad_nivel": 1, "Crédito_nivel": 2, "TPM_nivel": 3,
    "Inflación_nivel": 4, "Desempleo_nivel": 5, "TCRM_nivel": 6,
    "ResPrimario_nivel": 7, "ResFinanciero_nivel": 8,
    "Actividad_crec": 9, "Actividad_acel_t1": 10, "Actividad_señal_t1": 11,
    "Actividad_acel_t3": 12, "Actividad_señal_t3": 13, "Actividad_acel_t6": 14, "Actividad_señal_t6": 15,
    "Crédito_crec": 16, "Crédito_acel_t1": 17, "Crédito_señal_t1": 18,
    "Crédito_acel_t3": 19, "Crédito_señal_t3": 20, "Crédito_acel_t6": 21, "Crédito_señal_t6": 22,
    "TPM_crec": 23, "TPM_acel_t1": 24, "TPM_señal_t1": 25,
    "TPM_acel_t3": 26, "TPM_señal_t3": 27, "TPM_acel_t6": 28, "TPM_señal_t6": 29,
    "Inflación_crec": 31, "Inflación_acel_t1": 32, "Inflación_señal_t1": 33,
    "Inflación_acel_t3": 34, "Inflación_señal_t3": 35, "Inflación_acel_t6": 36, "Inflación_señal_t6": 37,
    "Desempleo_crec": 38, "Desempleo_acel_t1": 39, "Desempleo_señal_t1": 40,
    "Desempleo_acel_t3": 41, "Desempleo_señal_t3": 42, "Desempleo_acel_t6": 43, "Desempleo_señal_t6": 44,
    "TCRM_crec": 45, "TCRM_acel_t1": 46, "TCRM_señal_t1": 47,
    "TCRM_acel_t3": 48, "TCRM_señal_t3": 49, "TCRM_acel_t6": 50, "TCRM_señal_t6": 51,
    "ResPrimario_crec": 52, "ResPrimario_acel_t1": 53, "ResPrimario_señal_t1": 54,
    "ResPrimario_acel_t3": 55, "ResPrimario_señal_t3": 56, "ResPrimario_acel_t6": 57, "ResPrimario_señal_t6": 58,
    "ResFinanciero_crec": 59, "ResFinanciero_acel_t1": 60, "ResFinanciero_señal_t1": 61,
    "ResFinanciero_acel_t3": 62, "ResFinanciero_señal_t3": 63, "ResFinanciero_acel_t6": 64, "ResFinanciero_señal_t6": 65,
}

SIMPLE_COL_MAP = {
    "fecha": 0,
    "Actividad_nivel": 1, "Crédito_nivel": 2, "TPM_nivel": 3, "Inflación_nivel": 4, "Desempleo_nivel": 5,
    "Actividad_yoy": 6, "Actividad_crec": 7, "Actividad_acel_t1": 8, "Actividad_señal_t1": 9,
    "Crédito_yoy": 10, "Crédito_crec": 11, "Crédito_acel_t1": 12, "Crédito_señal_t1": 13,
    "TPM_yoy": 14, "TPM_crec": 15, "TPM_acel_t1": 16, "TPM_señal_t1": 17,
    "Inflación_yoy": 18, "Inflación_crec": 19, "Inflación_acel_t1": 20, "Inflación_señal_t1": 21,
    "Desempleo_yoy": 22, "Desempleo_crec": 23, "Desempleo_acel_t1": 24, "Desempleo_señal_t1": 25,
}


def _load_sheet_raw(sheet_name):
    return pd.read_excel(EXCEL_PATH, sheet_name, header=None)


def _extract_series(data_rows, col_idx, dates):
    return pd.to_numeric(data_rows.iloc[:, col_idx], errors="coerce").values


def load_full_country(code):
    raw = _load_sheet_raw(code)
    data_rows = raw.iloc[5:].copy()
    data_rows = data_rows[data_rows.iloc[:, 0].notna()].copy()
    dates = pd.to_datetime(data_rows.iloc[:, 0])
    result = {}
    niveles = pd.DataFrame(index=dates)
    for var, key in [("Actividad","Actividad_nivel"),("Crédito","Crédito_nivel"),("TPM","TPM_nivel"),
                     ("Inflación","Inflación_nivel"),("Desempleo","Desempleo_nivel"),("TCRM","TCRM_nivel"),
                     ("Res. Primario","ResPrimario_nivel"),("Res. Financiero","ResFinanciero_nivel")]:
        niveles[var] = _extract_series(data_rows, FULL_COL_MAP[key], dates)
    niveles.index.name = "Fecha"
    result["niveles"] = niveles.sort_index()

    var_configs = [("Actividad","Actividad"),("Crédito","Crédito"),("Inflación","Inflación"),
                   ("Tasa de política","TPM"),("Desempleo","Desempleo"),("TCRM","TCRM"),
                   ("Resultado primario","ResPrimario"),("Resultado financiero","ResFinanciero")]
    for var_name, prefix in var_configs:
        df = pd.DataFrame(index=dates)
        k = f"{prefix}_crec"
        if k in FULL_COL_MAP:
            df["Crecimiento"] = _extract_series(data_rows, FULL_COL_MAP[k], dates)
        for h in ["t1","t3","t6"]:
            ak, sk = f"{prefix}_acel_{h}", f"{prefix}_señal_{h}"
            if ak in FULL_COL_MAP:
                df[f"Acel_{h}"] = _extract_series(data_rows, FULL_COL_MAP[ak], dates)
            if sk in FULL_COL_MAP:
                df[f"Señal_{h}"] = data_rows.iloc[:, FULL_COL_MAP[sk]].values
        df.index.name = "Fecha"
        result[var_name] = df.sort_index()
    return result


def load_simple_country(code):
    raw = _load_sheet_raw(code)
    data_rows = raw.iloc[5:].copy()
    data_rows = data_rows[data_rows.iloc[:, 0].notna()].copy()
    dates = pd.to_datetime(data_rows.iloc[:, 0])
    result = {}
    niveles = pd.DataFrame(index=dates)
    for var, key in [("Actividad","Actividad_nivel"),("Crédito","Crédito_nivel"),("TPM","TPM_nivel"),
                     ("Inflación","Inflación_nivel"),("Desempleo","Desempleo_nivel")]:
        niveles[var] = _extract_series(data_rows, SIMPLE_COL_MAP[key], dates)
    niveles.index.name = "Fecha"
    result["niveles"] = niveles.sort_index()

    var_configs = [("Actividad","Actividad"),("Crédito","Crédito"),("Inflación","Inflación"),
                   ("Tasa de política","TPM"),("Desempleo","Desempleo")]
    for var_name, prefix in var_configs:
        df = pd.DataFrame(index=dates)
        k = f"{prefix}_crec"
        if k in SIMPLE_COL_MAP:
            df["Crecimiento"] = _extract_series(data_rows, SIMPLE_COL_MAP[k], dates)
        ak, sk = f"{prefix}_acel_t1", f"{prefix}_señal_t1"
        if ak in SIMPLE_COL_MAP:
            df["Acel_t1"] = _extract_series(data_rows, SIMPLE_COL_MAP[ak], dates)
        if sk in SIMPLE_COL_MAP:
            df["Señal_t1"] = data_rows.iloc[:, SIMPLE_COL_MAP[sk]].values
        df.index.name = "Fecha"
        result[var_name] = df.sort_index()
    return result


def load_resumen():
    raw = _load_sheet_raw("RESUMEN")
    rows = []
    for i in range(3, 29):
        r = raw.iloc[i]
        pais, variable = r.iloc[1], r.iloc[2]
        if pd.isna(pais) or pd.isna(variable):
            continue
        rows.append({
            "País": str(pais).strip(), "Variable": str(variable).strip(),
            "Fecha": r.iloc[3],
            "Índice": r.iloc[4] if pd.notna(r.iloc[4]) else "",
            "Crecimiento": r.iloc[5] if pd.notna(r.iloc[5]) else np.nan,
            "Acel_t1": r.iloc[6] if pd.notna(r.iloc[6]) else np.nan,
            "Acel_t3": r.iloc[7] if pd.notna(r.iloc[7]) else np.nan,
            "Acel_t6": r.iloc[8] if pd.notna(r.iloc[8]) else np.nan,
            "Señal_t1": r.iloc[9] if pd.notna(r.iloc[9]) else "",
            "Señal_t3": r.iloc[10] if pd.notna(r.iloc[10]) else "",
            "Señal_t6": r.iloc[11] if pd.notna(r.iloc[11]) else "",
        })
    return pd.DataFrame(rows)


def load_targets():
    raw = _load_sheet_raw("RESUMEN")
    rows = []
    for i in range(36, 41):
        r = raw.iloc[i]
        if pd.isna(r.iloc[1]):
            continue
        rows.append({
            "País": str(r.iloc[1]).strip(),
            "Inflación_actual": r.iloc[2] if pd.notna(r.iloc[2]) else np.nan,
            "Meta_inflación": r.iloc[3] if pd.notna(r.iloc[3]) else np.nan,
            "Rango": str(r.iloc[4]) if pd.notna(r.iloc[4]) else "",
            "TPM_actual": r.iloc[5] if pd.notna(r.iloc[5]) else np.nan,
            "TPM_esperada": r.iloc[6] if pd.notna(r.iloc[6]) else np.nan,
        })
    return pd.DataFrame(rows)


def load_country(code):
    if code in FULL_COUNTRIES:
        return load_full_country(code)
    return load_simple_country(code)
