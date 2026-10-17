import glob
import json
import os
import platform
import re
import sys
import time
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")  # Guardar graficas sin necesitar ventana
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# CONFIGURACION

CARPETA = os.path.dirname(os.path.abspath(__file__))

DATASET_KAGGLE = "ricardocassiano/projeto-python-tarefa-pratica-11-2"
NOMBRE_CSV = "Python_M11_support material .csv"

CARPETA_MODELOS = os.path.join(CARPETA, "modelos_comparativos")
CARPETA_GRAFICOS = os.path.join(CARPETA, "Graficos")
RUTA_CSV_RESULTADOS = os.path.join(CARPETA, "resultados_comparacion.csv")
RUTA_CSV_VC_RESUMEN = os.path.join(CARPETA, "resultados_validacion_cruzada.csv")
RUTA_CSV_VC_DETALLE = os.path.join(CARPETA, "resultados_validacion_cruzada_detalle.csv")
RUTA_CONFIG = os.path.join(CARPETA_MODELOS, "config_comparativo.json")

TARGET = "default"  # 0 = no moroso, 1 = moroso

COLUMNAS_CATEGORICAS = [
    "sexo", "escolaridade", "estado_civil", "salario_anual", "tipo_cartao"
]
COLUMNAS_NUMERICAS = [
    "idade", "dependentes", "meses_de_relacionamento", "qtd_produtos",
    "iteracoes_12m", "meses_inativo_12m", "limite_credito",
    "valor_transacoes_12m", "qtd_transacoes_12m"
]
COLUMNAS_MODELO = COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS

# PARÁMETROS PARA ELIMINAR VALORES ATÍPICOS (REGLA IQR)
# Valores atipicos: se detectan con la regla del rango intercuartilico (IQR) en las
# variables continuas. Las variables de conteo pequeno (dependentes, qtd_produtos,
# iteracoes_12m, meses_inativo_12m) no se revisan porque sus valores extremos
# (p. ej. 0 o 6) son validos y la regla IQR los marcaria por error.
COLUMNAS_ATIPICOS = [
    "idade", "meses_de_relacionamento", "limite_credito",
    "valor_transacoes_12m", "qtd_transacoes_12m"
]
FACTOR_IQR = 1.5  # limites = [Q1 - factor*IQR, Q3 + factor*IQR]. Sube a 3.0 para ser menos estricto.

TEST_SIZE = 0.20
RANDOM_STATE = 42
UMBRAL = 0.50  # Un cliente se clasifica como moroso si P(moroso) >= UMBRAL

# PARÁMETROS DE LA VALIDACIÓN CRUZADA (K-FOLD)
# Validacion cruzada: K-fold estratificado repetido n veces (particionamiento n veces)
K_FOLDS = 5
N_REPETICIONES = 3

METRICAS_VC = ["accuracy", "precision_moroso", "recall_moroso", "f1_moroso", "roc_auc"]

CRITERIO = "F1-score de la clase moroso (desempate: Recall moroso, luego ROC-AUC)"

AZUL_OSCURO = "#0f2a4a"
AZUL = "#2563eb"
NARANJA = "#f59e0b"


# LECTURA Y LIMPIEZA DE DATOS

def localizar_csv(ruta_manual=None):
    if ruta_manual:
        if not os.path.isfile(ruta_manual):
            raise FileNotFoundError(f"No existe el archivo indicado: {ruta_manual}")
        return ruta_manual, "ruta indicada por el usuario"

    local = os.path.join(CARPETA, NOMBRE_CSV)
    if os.path.isfile(local):
        return local, "carpeta del proyecto"

    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError(
            "No se encontro el CSV en la carpeta del proyecto y 'kagglehub' no esta "
            "instalado.\nInstalalo con:  pip install kagglehub\n"
            "o copia el CSV a la carpeta del proyecto."
        ) from e

    carpeta_dataset = kagglehub.dataset_download(DATASET_KAGGLE)
    candidatos = glob.glob(os.path.join(carpeta_dataset, "**", "*.csv"), recursive=True)
    if not candidatos:
        raise FileNotFoundError(f"No hay archivos .csv en {carpeta_dataset}")
    exacto = [c for c in candidatos if os.path.basename(c) == NOMBRE_CSV]
    return (exacto or candidatos)[0], "descarga de Kaggle (kagglehub)"


def leer_csv(ruta):
    ultimo_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        for sep in (",", ";"):
            try:
                df = pd.read_csv(ruta, sep=sep, encoding=encoding)
            except Exception as e:  # noqa: BLE001
                ultimo_error = e
                continue
            df.columns = [str(c).strip() for c in df.columns]
            if TARGET in df.columns:
                return df
    raise ValueError(f"No se pudo leer el CSV con la columna '{TARGET}'. Ultimo error: {ultimo_error}")


def convertir_numero(valor):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return np.nan
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return float(valor)

    texto = re.sub(r"[R$\s]", "", str(valor))
    if texto == "" or texto.lower() in {"na", "nan", "n/a", "null", "none"}:
        return np.nan

    if "," in texto:
        texto = texto.replace(".", "").replace(",", ".")
    elif texto.count(".") > 1:
        texto = texto.replace(".", "")

    try:
        return float(texto)
    except ValueError:
        return np.nan


# LIMPIEZA DE DATOS: DUPLICADOS Y VALORES ATÍPICOS
def limpiar_datos(df):
    """Limpieza comun a los seis modelos.

    Pasos: validacion de columnas y tipos -> eliminacion de duplicados ->
    eliminacion de valores atipicos (regla IQR).
    Devuelve X, y, un resumen de la limpieza y los datos previos a quitar atipicos
    (solo para graficar el antes/despues).
    """
    df = df.copy()
    resumen = {"filas_originales": int(len(df))}

    # VALIDACIÓN DE COLUMNAS REQUERIDAS
    faltan = [c for c in COLUMNAS_MODELO + [TARGET] if c not in df.columns]
    if faltan:
        raise ValueError(
            f"Faltan columnas en el CSV: {faltan}\nColumnas encontradas: {list(df.columns)}"
        )

    # CONVERSIÓN DE LA VARIABLE OBJETIVO
    # Variable objetivo
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    antes = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    resumen["filas_sin_objetivo"] = int(antes - len(df))
    if len(df) < antes:
        print(f"  Aviso: {antes - len(df)} filas sin valor en '{TARGET}' fueron descartadas.")
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"'{TARGET}' debe contener solo 0 y 1: {sorted(df[TARGET].unique())}")

    # LIMPIEZA DE ESPACIOS EN VARIABLES CATEGÓRICAS
    # Categoricas: solo se quitan espacios (se conservan los valores originales, p. ej. 'na')
    for col in COLUMNAS_CATEGORICAS:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

    # CONVERSIÓN DE VARIABLES NUMÉRICAS
    # Numericas: conversion robusta (el identificador 'id' NO se usa como predictor)
    for col in COLUMNAS_NUMERICAS:
        df[col] = df[col].map(convertir_numero).astype(float)

    # ===== ELIMINACIÓN DE DUPLICADOS =====
    # 1) Eliminacion de duplicados.
    # Se comparan las variables del modelo + el objetivo e se ignora 'id', porque dos
    # clientes con exactamente los mismos datos pero distinto 'id' son un registro repetido.
    criterio_dup = COLUMNAS_MODELO + [TARGET]
    mascara_dup = df.duplicated(subset=criterio_dup, keep="first")
    resumen["duplicados_eliminados"] = int(mascara_dup.sum())
    df = df.loc[~mascara_dup].copy()
    resumen["filas_tras_duplicados"] = int(len(df))

    # ===== ELIMINACIÓN DE VALORES ATÍPICOS (REGLA IQR) =====
    # 2) Eliminacion de valores atipicos (regla IQR)
    datos_previos = df[COLUMNAS_ATIPICOS + [TARGET]].copy()
    resumen["morosos_antes"] = int(df[TARGET].sum())
    resumen["factor_iqr"] = FACTOR_IQR
    resumen["columnas_atipicos"] = list(COLUMNAS_ATIPICOS)

    limites = {}
    atipicos_por_columna = {}
    mascara_atipico = pd.Series(False, index=df.index)
    for col in COLUMNAS_ATIPICOS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        minimo = q1 - FACTOR_IQR * iqr
        maximo = q3 + FACTOR_IQR * iqr
        fuera = (df[col] < minimo) | (df[col] > maximo)  # los NaN no cuentan como atipicos
        limites[col] = {
            "q1": float(q1), "q3": float(q3),
            "limite_inferior": float(minimo), "limite_superior": float(maximo),
        }
        atipicos_por_columna[col] = int(fuera.sum())
        mascara_atipico |= fuera

    resumen["limites_iqr"] = limites
    resumen["atipicos_por_columna"] = atipicos_por_columna
    resumen["atipicos_eliminados"] = int(mascara_atipico.sum())
    df = df.loc[~mascara_atipico].reset_index(drop=True)
    resumen["filas_finales"] = int(len(df))
    resumen["morosos_despues"] = int(df[TARGET].sum())

    X = df[COLUMNAS_MODELO].copy()
    y = df[TARGET].copy()
    return X, y, resumen, datos_previos


# RESUMEN DE LA LIMPIEZA (DUPLICADOS Y ATÍPICOS)
def imprimir_resumen_limpieza(res):
    print(f"  Filas originales ............................ {res['filas_originales']}")
    if res["filas_sin_objetivo"]:
        print(f"  Filas sin valor en '{TARGET}' (descartadas) .... {res['filas_sin_objetivo']}")
    print(f"  Duplicados eliminados ....................... {res['duplicados_eliminados']}")
    print(f"  Filas tras quitar duplicados ................ {res['filas_tras_duplicados']}")
    print(f"  Atipicos eliminados (IQR x {res['factor_iqr']}) ............. {res['atipicos_eliminados']}")
    print(f"  Filas finales ............................... {res['filas_finales']}")
    print("  Atipicos detectados por columna (una fila puede contarse en varias):")
    for col, n in res["atipicos_por_columna"].items():
        lim = res["limites_iqr"][col]
        print(f"    {col}: {n}  (rango valido: {lim['limite_inferior']:.2f} a {lim['limite_superior']:.2f})")

    pct_antes = res["morosos_antes"] / max(res["filas_tras_duplicados"], 1) * 100
    pct_despues = res["morosos_despues"] / max(res["filas_finales"], 1) * 100
    print(f"  % de morosos antes de quitar atipicos: {pct_antes:.2f}%  |  despues: {pct_despues:.2f}%")

    if res["filas_tras_duplicados"] and res["atipicos_eliminados"] / res["filas_tras_duplicados"] > 0.15:
        print("  AVISO: se elimino mas del 15% de las filas como atipicas. Si baja mucho la cantidad "
              "de morosos,\n         sube FACTOR_IQR (por ejemplo a 3.0) para ser menos estricto.")


# INFORME DE LOS DATOS YA LIMPIOS
def informe_datos(X, y):
    print(f"  Filas: {len(X)}  |  Columnas predictoras: {X.shape[1]}")
    morosos = int(y.sum())
    print(f"  No morosos (0): {len(y) - morosos}  |  Morosos (1): {morosos}  "
          f"({morosos / len(y) * 100:.2f}% morosos)")

    faltantes = X.isna().sum()
    faltantes = faltantes[faltantes > 0]
    print("  Valores faltantes por columna:",
          "ninguno" if faltantes.empty else dict(faltantes))

    print("  Categorias encontradas:")
    for col in COLUMNAS_CATEGORICAS:
        print(f"    {col}: {sorted(X[col].dropna().unique().tolist())}")

    print("  Resumen de columnas numericas clave (verifica que los rangos sean logicos):")
    resumen = X[["limite_credito", "valor_transacoes_12m"]].describe().loc[
        ["min", "50%", "max"]
    ]
    print("    " + resumen.round(2).to_string().replace("\n", "\n    "))


# PIPELINES Y MODELOS

# PREPROCESAMIENTO: IMPUTACIÓN DE FALTANTES, CODIFICACIÓN Y ESCALADO
def crear_preprocesador(escalar):
    categoricas = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore")),
    ])

    pasos_numericos = [("imputador", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_numericos.append(("escalador", StandardScaler()))

    return ColumnTransformer([
        ("categoricas", categoricas, COLUMNAS_CATEGORICAS),
        ("numericas", Pipeline(pasos_numericos), COLUMNAS_NUMERICAS),
    ])


# DEFINICIÓN DE LOS SEIS MODELOS
def definir_modelos():
    return [
        {
            # MODELO: RANDOM FOREST
            "nombre": "Random Forest",
            "archivo": "random_forest.pkl",
            "escalar": False,
            "pesos": False,
            "clasificador": RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_STATE,
                class_weight="balanced", n_jobs=-1,
            ),
        },
        {
            # MODELO: REGRESIÓN LOGÍSTICA
            "nombre": "Regresión logística",
            "archivo": "regresion_logistica.pkl",
            "escalar": True,
            "pesos": False,
            "clasificador": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
            ),
        },
        {
            # MODELO: ÁRBOL DE DECISIÓN
            "nombre": "Árbol de decisión",
            "archivo": "arbol_decision.pkl",
            "escalar": False,
            "pesos": False,
            "clasificador": DecisionTreeClassifier(
                max_depth=8, min_samples_leaf=5,
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
        },
        {
            # MODELO: GRADIENT BOOSTING
            "nombre": "Gradient Boosting",
            "archivo": "gradient_boosting.pkl",
            "escalar": False,
            "pesos": True,  # GradientBoostingClassifier no tiene class_weight
            "clasificador": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=3,
                random_state=RANDOM_STATE,
            ),
        },
        {
            # MODELO: SVM
            "nombre": "SVM",
            "archivo": "svm.pkl",
            "escalar": True,
            "pesos": False,
            # SVC(probability=True) esta obsoleto en scikit-learn >= 1.9; esta es la
            # forma equivalente (calibracion sigmoide de Platt) valida en todas las versiones.
            "clasificador": CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
                method="sigmoid", cv=5, ensemble=False,
            ),
        },
        {
            # MODELO: KNN
            "nombre": "KNN",
            "archivo": "knn.pkl",
            "escalar": True,
            "pesos": False,  # KNN no admite pesos de clase
            "clasificador": KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1,
            ),
        },
    ]


# CREACIÓN DEL PIPELINE (PREPROCESAMIENTO + MODELO)
def crear_pipeline(cfg):
    # clone() garantiza un modelo nuevo y sin entrenar en cada ajuste (cada fold)
    return Pipeline([
        ("preprocesamiento", crear_preprocesador(cfg["escalar"])),
        ("modelo", clone(cfg["clasificador"])),
    ])


# AJUSTE DEL MODELO (CON PESOS DE CLASE SI APLICA)
def ajustar_pipeline(pipeline, cfg, X_ent, y_ent):
    if cfg["pesos"]:
        pesos = compute_sample_weight("balanced", y_ent)
        pipeline.fit(X_ent, y_ent, modelo__sample_weight=pesos)
    else:
        pipeline.fit(X_ent, y_ent)


# MÉTRICAS: ACCURACY, PRECISIÓN, RECALL, F1, ROC-AUC Y MATRIZ DE CONFUSIÓN
def calcular_metricas(y_real, probabilidad):
    """Misma regla de decision para todos: P(moroso) >= UMBRAL."""
    prediccion = (probabilidad >= UMBRAL).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_real, prediccion, labels=[0, 1]).ravel()

    try:
        auc = float(roc_auc_score(y_real, probabilidad))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_real, prediccion)),
        "precision_moroso": float(precision_score(y_real, prediccion, pos_label=1, zero_division=0)),
        "recall_moroso": float(recall_score(y_real, prediccion, pos_label=1, zero_division=0)),
        "f1_moroso": float(f1_score(y_real, prediccion, pos_label=1, zero_division=0)),
        "roc_auc": auc,
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
    }


# ENTRENAMIENTO FINAL Y EVALUACIÓN EN EL CONJUNTO DE PRUEBA
def entrenar_y_evaluar(cfg, X_train, X_test, y_train, y_test):
    """Entrenamiento final (con todo el 80%) y evaluacion en el 20% de prueba."""
    pipeline = crear_pipeline(cfg)

    inicio = time.perf_counter()
    ajustar_pipeline(pipeline, cfg, X_train, y_train)
    tiempo = time.perf_counter() - inicio

    probabilidad = pipeline.predict_proba(X_test)[:, 1]
    metricas = {"modelo": cfg["nombre"], "archivo": cfg["archivo"]}
    metricas.update(calcular_metricas(y_test, probabilidad))
    metricas["tiempo_entrenamiento_s"] = round(float(tiempo), 2)
    return pipeline, metricas, probabilidad


# VALIDACIÓN CRUZADA K-FOLD ESTRATIFICADA REPETIDA N VECES
def validacion_cruzada(cfg, X, y):
    """K-fold estratificado repetido N veces (particionamiento n veces).

    En cada una de las K x N iteraciones se entrena un modelo nuevo con K-1 folds
    y se mide en el fold restante. Devuelve una fila de metricas por iteracion.
    """
    # PARTICIONAMIENTO N VECES (K-FOLD REPETIDO)
    particionador = RepeatedStratifiedKFold(
        n_splits=K_FOLDS, n_repeats=N_REPETICIONES, random_state=RANDOM_STATE
    )
    filas = []
    for n, (idx_ent, idx_val) in enumerate(particionador.split(X, y)):
        pipeline = crear_pipeline(cfg)
        ajustar_pipeline(pipeline, cfg, X.iloc[idx_ent], y.iloc[idx_ent])
        probabilidad = pipeline.predict_proba(X.iloc[idx_val])[:, 1]
        m = calcular_metricas(y.iloc[idx_val], probabilidad)

        fila = {
            "modelo": cfg["nombre"],
            "repeticion": n // K_FOLDS + 1,
            "fold": n % K_FOLDS + 1,
        }
        fila.update({k: m[k] for k in METRICAS_VC})
        filas.append(fila)
    return filas


# RESUMEN DE LA VALIDACIÓN CRUZADA (MEDIA Y DESVIACIÓN ESTÁNDAR)
def resumir_validacion_cruzada(filas):
    """Media y desviacion estandar de cada metrica sobre las K x N iteraciones."""
    df = pd.DataFrame(filas)
    resumen = {}
    for metrica in METRICAS_VC:
        base = metrica.replace("_moroso", "")
        resumen[f"cv_{base}_media"] = float(df[metrica].mean())
        resumen[f"cv_{base}_std"] = float(df[metrica].std(ddof=1))
    return resumen


# SALIDAS: TABLAS

# TABLA DE MÉTRICAS (RANKING DE MODELOS)
def construir_tabla(lista_metricas):
    tabla = pd.DataFrame(lista_metricas)
    tabla = tabla.sort_values(
        by=["f1_moroso", "recall_moroso", "roc_auc"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    tabla.insert(0, "ranking", np.arange(1, len(tabla) + 1))
    return tabla


# IMPRESIÓN DE LA TABLA DE MÉTRICAS (CONJUNTO DE PRUEBA)
def imprimir_tabla(tabla):
    vista = tabla[[
        "ranking", "modelo", "accuracy", "precision_moroso", "recall_moroso",
        "f1_moroso", "roc_auc", "verdaderos_negativos", "falsos_positivos",
        "falsos_negativos", "verdaderos_positivos", "tiempo_entrenamiento_s",
    ]].copy()

    for col in ["accuracy", "precision_moroso", "recall_moroso", "f1_moroso", "roc_auc"]:
        vista[col] = (vista[col] * 100).map(lambda v: f"{v:.2f}%")

    vista.columns = [
        "#", "Modelo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
        "TN", "FP", "FN", "TP", "Tiempo(s)",
    ]
    print(vista.to_string(index=False))


def texto_media_std(media, std):
    return f"{media * 100:.2f}% ± {std * 100:.2f}%"


# IMPRESIÓN DE LA TABLA DE VALIDACIÓN CRUZADA
def imprimir_tabla_vc(tabla):
    vista = pd.DataFrame({
        "#": tabla["ranking"],
        "Modelo": tabla["modelo"],
        "Accuracy": [texto_media_std(m, s) for m, s in zip(tabla["cv_accuracy_media"], tabla["cv_accuracy_std"])],
        "Precision": [texto_media_std(m, s) for m, s in zip(tabla["cv_precision_media"], tabla["cv_precision_std"])],
        "Recall": [texto_media_std(m, s) for m, s in zip(tabla["cv_recall_media"], tabla["cv_recall_std"])],
        "F1": [texto_media_std(m, s) for m, s in zip(tabla["cv_f1_media"], tabla["cv_f1_std"])],
        "ROC-AUC": [texto_media_std(m, s) for m, s in zip(tabla["cv_roc_auc_media"], tabla["cv_roc_auc_std"])],
    })
    print(vista.to_string(index=False))


# GUARDADO DE LA VALIDACIÓN CRUZADA EN CSV
def guardar_csv_vc(detalle_vc, tabla):
    detalle = pd.DataFrame(detalle_vc)
    detalle.to_csv(RUTA_CSV_VC_DETALLE, index=False, encoding="utf-8-sig")

    columnas = ["ranking", "modelo"] + [
        c for c in tabla.columns if c.startswith("cv_")
    ]
    tabla[columnas].to_csv(RUTA_CSV_VC_RESUMEN, index=False, encoding="utf-8-sig")


# SALIDAS: GRAFICOS (carpeta Graficos)

def _caja(ax, cx, cy, w, h, texto, color, tam=9.5):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.5",
        linewidth=1.4, edgecolor=AZUL_OSCURO, facecolor=color,
    ))
    ax.text(cx, cy, texto, ha="center", va="center", fontsize=tam,
            color="#111827", linespacing=1.4)
    return (cx, cy, w, h)


def _abajo(caja, dx=0.0):
    cx, cy, w, h = caja
    return (cx + dx, cy - h / 2 - 0.6)


def _arriba(caja, dx=0.0):
    cx, cy, w, h = caja
    return (cx + dx, cy + h / 2 + 0.6)


def _flecha(ax, origen, destino):
    ax.annotate(
        "", xy=destino, xytext=origen,
        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.8, shrinkA=0, shrinkB=0),
    )


# GRÁFICO: DIAGRAMA DEL PROCESO
def graficar_diagrama_proceso(ruta_png):
    fig, ax = plt.subplots(figsize=(9.5, 14))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 150)
    ax.axis("off")

    c_datos, c_limpieza, c_vc, c_prueba, c_final, c_salida = (
        "#dbeafe", "#fef3c7", "#e0e7ff", "#fce7f3", "#dcfce7", "#e5e7eb"
    )

    b1 = _caja(ax, 50, 141, 58, 9,
               "1. Conjunto de datos (Kaggle)\n14 variables + objetivo «default»", c_datos)
    b2 = _caja(ax, 50, 126, 58, 9,
               "2. Lectura y validación\nconversión de tipos · objetivo 0/1", c_datos)
    b3 = _caja(ax, 50, 109, 70, 17,
               "3. Limpieza de datos\n• Eliminación de duplicados\n"
               "• Eliminación de valores atípicos (regla IQR)\n"
               "• Faltantes: imputación dentro del pipeline", c_limpieza)
    b4 = _caja(ax, 50, 91, 58, 9,
               "4. Partición estratificada\n80 % entrenamiento · 20 % prueba", c_datos)
    b5a = _caja(ax, 26, 72, 44, 15,
                f"5. Validación cruzada\nK-fold estratificado (K = {K_FOLDS})\n"
                f"repetido n = {N_REPETICIONES} veces\n(solo sobre el entrenamiento)", c_vc)
    b5b = _caja(ax, 76, 72, 40, 15,
                "Conjunto de prueba\n20 % reservado\n(no participa en el\najuste de los modelos)", c_prueba)
    b6 = _caja(ax, 26, 53, 44, 13,
               "6. Entrenamiento final\n6 modelos con pipeline:\nimputación + codificación\n+ escalado + clasificador", c_final)
    b7 = _caja(ax, 50, 34, 70, 11,
               "7. Evaluación en el conjunto de prueba\nAccuracy · Precisión · Recall · F1 · ROC-AUC", c_final)
    b8 = _caja(ax, 50, 19, 70, 11,
               "8. Resultados\ntabla de métricas · matrices de confusión · gráficos", c_salida)
    b9 = _caja(ax, 50, 5, 58, 8,
               "9. Aplicación Tkinter\npredicción con los 6 modelos y mejor modelo ★", c_salida)

    _flecha(ax, _abajo(b1), _arriba(b2))
    _flecha(ax, _abajo(b2), _arriba(b3))
    _flecha(ax, _abajo(b3), _arriba(b4))
    _flecha(ax, _abajo(b4, -8), _arriba(b5a, 0))
    _flecha(ax, _abajo(b4, 8), _arriba(b5b, 0))
    _flecha(ax, _abajo(b5a), _arriba(b6))
    _flecha(ax, _abajo(b6), _arriba(b7, -24))
    _flecha(ax, _abajo(b5b), _arriba(b7, 26))
    _flecha(ax, _abajo(b7), _arriba(b8))
    _flecha(ax, _abajo(b8), _arriba(b9))

    ax.set_title("Diagrama del proceso de entrenamiento y comparación de modelos",
                 fontsize=13, fontweight="bold", color=AZUL_OSCURO, pad=10)
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# GRÁFICO: ESQUEMA DE LA VALIDACIÓN CRUZADA K-FOLD
def graficar_esquema_validacion_cruzada(ruta_png):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, K_FOLDS + 3.2)
    ax.axis("off")

    x0, ancho_total = 14, 56
    ancho = ancho_total / K_FOLDS

    ax.text(x0 + ancho_total / 2, K_FOLDS + 2.3, "Repetición 1 (se reparte el entrenamiento en K partes)",
            ha="center", fontsize=10.5, fontweight="bold", color=AZUL_OSCURO)

    for i in range(K_FOLDS):
        y = K_FOLDS - i
        ax.text(x0 - 1.5, y + 0.4, f"Iteración {i + 1}", ha="right", va="center", fontsize=9.5)
        for j in range(K_FOLDS):
            validacion = (i == j)
            ax.add_patch(Rectangle(
                (x0 + j * ancho, y), ancho - 0.6, 0.8,
                facecolor=NARANJA if validacion else AZUL, edgecolor="white",
            ))
            ax.text(x0 + j * ancho + (ancho - 0.6) / 2, y + 0.4,
                    "Validación" if validacion else "Entrena",
                    ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    for j in range(K_FOLDS):
        ax.text(x0 + j * ancho + (ancho - 0.6) / 2, K_FOLDS + 1.2, f"Fold {j + 1}",
                ha="center", fontsize=9, color="#374151")

    ax.text(
        73, K_FOLDS / 2 + 0.9,
        f"• Cada fila es un entrenamiento\n  nuevo del modelo.\n\n"
        f"• K = {K_FOLDS} iteraciones por repetición.\n\n"
        f"• Se repite n = {N_REPETICIONES} veces con\n  particiones distintas.\n\n"
        f"• Total: {K_FOLDS} × {N_REPETICIONES} = {K_FOLDS * N_REPETICIONES} evaluaciones\n  por modelo.\n\n"
        "• Resultado = media ± desviación\n  estándar de las métricas.",
        fontsize=9.5, va="center", ha="left", color="#111827", linespacing=1.15,
    )
    ax.text(x0 + ancho_total / 2, 0.2,
            "Las particiones son estratificadas: cada fold conserva la proporción de morosos.",
            ha="center", fontsize=9, color="#6b7280", style="italic")

    ax.set_title("Validación cruzada K-fold repetida (particionamiento n veces)",
                 fontsize=13, fontweight="bold", color=AZUL_OSCURO)
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# GRÁFICO: LIMPIEZA DE DATOS (DUPLICADOS Y ATÍPICOS)
def graficar_limpieza(res, ruta_png):
    fig, ejes = plt.subplots(1, 3, figsize=(17, 5.2))

    # (a) Filas en cada etapa
    etapas = ["Originales", "Sin duplicados", "Sin atípicos"]
    valores = [res["filas_originales"], res["filas_tras_duplicados"], res["filas_finales"]]
    barras = ejes[0].bar(etapas, valores, color=["#93c5fd", AZUL, AZUL_OSCURO])
    for barra, v in zip(barras, valores):
        ejes[0].text(barra.get_x() + barra.get_width() / 2, v, f"{v:,}",
                     ha="center", va="bottom", fontweight="bold", fontsize=10)
    ejes[0].set_ylim(0, max(valores) * 1.12)
    ejes[0].set_title(f"Filas en cada etapa\n(duplicados: {res['duplicados_eliminados']} · "
                      f"atípicos: {res['atipicos_eliminados']})", fontsize=11)
    ejes[0].set_ylabel("Cantidad de filas")

    # (b) Atipicos por columna
    columnas = list(res["atipicos_por_columna"].keys())
    cantidades = [res["atipicos_por_columna"][c] for c in columnas]
    barras = ejes[1].barh(columnas, cantidades, color=NARANJA)
    maximo = max(cantidades) if max(cantidades) > 0 else 1
    for barra, v in zip(barras, cantidades):
        ejes[1].text(v + maximo * 0.01, barra.get_y() + barra.get_height() / 2, str(v),
                     va="center", fontsize=10, fontweight="bold")
    ejes[1].set_xlim(0, maximo * 1.15)
    ejes[1].invert_yaxis()
    ejes[1].set_title(f"Valores atípicos detectados por columna\n(regla IQR × {res['factor_iqr']})", fontsize=11)
    ejes[1].set_xlabel("Cantidad de filas fuera de rango")

    # (c) Porcentaje de morosos antes / despues
    pct_antes = res["morosos_antes"] / max(res["filas_tras_duplicados"], 1) * 100
    pct_despues = res["morosos_despues"] / max(res["filas_finales"], 1) * 100
    barras = ejes[2].bar(["Antes de quitar\natípicos", "Después de quitar\natípicos"],
                         [pct_antes, pct_despues], color=["#fca5a5", "#dc2626"])
    for barra, v, n in zip(barras, [pct_antes, pct_despues],
                           [res["morosos_antes"], res["morosos_despues"]]):
        ejes[2].text(barra.get_x() + barra.get_width() / 2, v, f"{v:.2f}%\n({n} morosos)",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    ejes[2].set_ylim(0, max(pct_antes, pct_despues) * 1.3 + 1)
    ejes[2].set_title("Proporción de clientes morosos", fontsize=11)
    ejes[2].set_ylabel("% de morosos")

    for eje in ejes:
        eje.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Limpieza de datos: duplicados y valores atípicos", fontsize=14,
                 fontweight="bold", color=AZUL_OSCURO)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)


# GRÁFICO: BOXPLOTS DE VALORES ATÍPICOS (ANTES Y DESPUÉS)
def graficar_boxplots_atipicos(datos_previos, X_final, ruta_png):
    columnas = COLUMNAS_ATIPICOS
    fig, ejes = plt.subplots(1, len(columnas), figsize=(3.4 * len(columnas), 5))

    for eje, col in zip(np.atleast_1d(ejes), columnas):
        antes = datos_previos[col].dropna().values
        despues = X_final[col].dropna().values
        caja = eje.boxplot(
            [antes, despues], patch_artist=True, showmeans=True,
            flierprops=dict(marker="o", markersize=3, markerfacecolor="#ef4444",
                            markeredgecolor="none", alpha=0.5),
        )
        for parche, color in zip(caja["boxes"], ["#fca5a5", "#93c5fd"]):
            parche.set_facecolor(color)
        eje.set_xticklabels(["Antes", "Después"])
        eje.set_title(col, fontsize=10, fontweight="bold")
        eje.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Valores atípicos: distribución antes y después de la limpieza (los puntos rojos son atípicos)",
                 fontsize=12.5, fontweight="bold", color=AZUL_OSCURO)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)


# GRÁFICO: MATRICES DE CONFUSIÓN
def graficar_matrices(tabla, ruta_png):
    fig, ejes = plt.subplots(2, 3, figsize=(15, 9.5))
    ejes = ejes.ravel()

    for eje, (_, fila) in zip(ejes, tabla.iterrows()):
        matriz = np.array([
            [fila["verdaderos_negativos"], fila["falsos_positivos"]],
            [fila["falsos_negativos"], fila["verdaderos_positivos"]],
        ])
        porcentaje_fila = matriz / matriz.sum(axis=1, keepdims=True) * 100

        eje.imshow(matriz, cmap="Blues", vmin=0, vmax=matriz.max() * 1.15)
        eje.set_title(
            f"{int(fila['ranking'])}. {fila['modelo']}\n"
            f"F1 = {fila['f1_moroso']:.3f}  ·  Recall = {fila['recall_moroso']:.3f}",
            fontsize=11,
        )
        eje.set_xticks([0, 1])
        eje.set_yticks([0, 1])
        eje.set_xticklabels(["No moroso", "Moroso"])
        eje.set_yticklabels(["No moroso", "Moroso"])
        eje.set_xlabel("Predicción")
        eje.set_ylabel("Real")

        for i in range(2):
            for j in range(2):
                color = "white" if matriz[i, j] > matriz.max() * 0.6 else "#1f2937"
                eje.text(
                    j, i, f"{matriz[i, j]}\n({porcentaje_fila[i, j]:.1f}% de la fila)",
                    ha="center", va="center", color=color, fontsize=10, fontweight="bold",
                )

    for eje in ejes[len(tabla):]:
        eje.axis("off")

    fig.suptitle(
        f"Matrices de confusión (conjunto de prueba, umbral {UMBRAL:.0%}) · "
        "ordenadas por F1 de la clase moroso",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)


# GRÁFICO: COMPARACIÓN DE MÉTRICAS POR MODELO
def graficar_comparacion_metricas(tabla, ruta_png):
    metricas = [
        ("accuracy", "Accuracy"), ("precision_moroso", "Precisión"),
        ("recall_moroso", "Recall"), ("f1_moroso", "F1"), ("roc_auc", "ROC-AUC"),
    ]
    colores = ["#1d4ed8", "#0f766e", "#f59e0b", "#dc2626", "#7c3aed"]
    modelos = tabla["modelo"].tolist()
    x = np.arange(len(modelos))
    ancho = 0.16

    fig, eje = plt.subplots(figsize=(14, 6.5))
    for i, ((col, nombre), color) in enumerate(zip(metricas, colores)):
        valores = tabla[col].values
        barras = eje.bar(x + (i - 2) * ancho, valores, ancho, label=nombre, color=color)
        for barra, v in zip(barras, valores):
            eje.text(barra.get_x() + barra.get_width() / 2, v + 0.005, f"{v * 100:.0f}",
                     ha="center", va="bottom", fontsize=7.5)

    eje.set_xticks(x)
    eje.set_xticklabels([f"{r}. {m}" for r, m in zip(tabla["ranking"], modelos)])
    eje.set_ylim(0, 1.1)
    eje.set_ylabel("Valor de la métrica (0 a 1)")
    eje.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    eje.spines[["top", "right"]].set_visible(False)
    eje.set_title("Comparación de métricas por modelo (conjunto de prueba)",
                  fontsize=13, fontweight="bold", color=AZUL_OSCURO)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)


# GRÁFICO: VARIABILIDAD DE LA VALIDACIÓN CRUZADA (BOXPLOT)
def graficar_boxplot_vc(detalle_vc, tabla, ruta_png):
    detalle = pd.DataFrame(detalle_vc)
    orden = tabla["modelo"].tolist()

    fig, ejes = plt.subplots(1, 2, figsize=(15, 6))
    for eje, (col, titulo) in zip(ejes, [("accuracy", "Accuracy"), ("f1_moroso", "F1 de la clase moroso")]):
        datos = [detalle.loc[detalle["modelo"] == m, col].values for m in orden]
        caja = eje.boxplot(datos, patch_artist=True, showmeans=True)
        for parche in caja["boxes"]:
            parche.set_facecolor("#bfdbfe")
        eje.set_xticklabels(orden, rotation=20, ha="right")
        eje.set_ylabel(titulo)
        eje.set_title(f"{titulo} en {K_FOLDS * N_REPETICIONES} evaluaciones", fontsize=11)
        eje.spines[["top", "right"]].set_visible(False)
        eje.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Validación cruzada K-fold (K = {K_FOLDS}, repetida {N_REPETICIONES} veces): "
                 "variabilidad de cada modelo",
                 fontsize=13, fontweight="bold", color=AZUL_OSCURO)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)


# GRÁFICO: CURVAS ROC
def graficar_roc(curvas, tabla, ruta_png):
    fig, eje = plt.subplots(figsize=(8, 7))
    for _, fila in tabla.iterrows():
        nombre = fila["modelo"]
        if nombre not in curvas:
            continue
        fpr, tpr = curvas[nombre]
        eje.plot(fpr, tpr, lw=2, label=f"{nombre} (AUC = {fila['roc_auc']:.3f})")
    eje.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Azar")
    eje.set_xlabel("Tasa de falsos positivos")
    eje.set_ylabel("Tasa de verdaderos positivos (Recall)")
    eje.set_title("Curvas ROC (conjunto de prueba)", fontsize=13, fontweight="bold", color=AZUL_OSCURO)
    eje.legend(loc="lower right")
    eje.grid(alpha=0.3)
    eje.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=150)
    plt.close(fig)


# GRÁFICO: TABLA DE MÉTRICAS (IMAGEN)
def graficar_tabla_metricas(tabla, ruta_png):
    columnas = [
        "#", "Modelo", "Accuracy", "Precisión", "Recall", "F1", "ROC-AUC",
        "Accuracy (VC)", "Precisión (VC)", "Recall (VC)", "F1 (VC)",
    ]
    filas = []
    for _, f in tabla.iterrows():
        filas.append([
            int(f["ranking"]), f["modelo"],
            f"{f['accuracy'] * 100:.2f}%", f"{f['precision_moroso'] * 100:.2f}%",
            f"{f['recall_moroso'] * 100:.2f}%", f"{f['f1_moroso'] * 100:.2f}%",
            f"{f['roc_auc'] * 100:.2f}%",
            texto_media_std(f["cv_accuracy_media"], f["cv_accuracy_std"]),
            texto_media_std(f["cv_precision_media"], f["cv_precision_std"]),
            texto_media_std(f["cv_recall_media"], f["cv_recall_std"]),
            texto_media_std(f["cv_f1_media"], f["cv_f1_std"]),
        ])

    fig, eje = plt.subplots(figsize=(21, 1.6 + 0.6 * len(filas)))
    eje.axis("off")
    tb = eje.table(cellText=filas, colLabels=columnas, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 2.0)
    try:
        tb.auto_set_column_width(list(range(len(columnas))))
    except Exception:
        pass

    for (r, c), celda in tb.get_celld().items():
        celda.set_edgecolor("#d9e1ec")
        if r == 0:
            celda.set_facecolor(AZUL_OSCURO)
            celda.set_text_props(color="white", fontweight="bold")
        elif r == 1:
            celda.set_facecolor("#bfdbfe")
            celda.set_text_props(fontweight="bold")
        else:
            celda.set_facecolor("#f8fafc" if r % 2 == 0 else "white")

    fig.suptitle("Tabla de métricas de los modelos", fontsize=14, fontweight="bold", color=AZUL_OSCURO)
    fig.text(
        0.5, 0.02,
        f"Accuracy, Precisión, Recall, F1 y ROC-AUC: conjunto de prueba (umbral {UMBRAL:.0%}). "
        f"(VC) = media ± desviación estándar de la validación cruzada K = {K_FOLDS} × {N_REPETICIONES} repeticiones. "
        "Precisión, Recall y F1 se refieren a la clase moroso.",
        ha="center", fontsize=9, color="#6b7280",
    )
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# GENERACIÓN DE TODOS LOS GRÁFICOS EN LA CARPETA «Graficos»
def generar_graficos(tabla, detalle_vc, curvas, res_limpieza, datos_previos, X):
    os.makedirs(CARPETA_GRAFICOS, exist_ok=True)
    g = lambda nombre: os.path.join(CARPETA_GRAFICOS, nombre)  # noqa: E731

    graficar_diagrama_proceso(g("01_diagrama_proceso.png"))
    graficar_esquema_validacion_cruzada(g("02_esquema_validacion_cruzada.png"))
    graficar_limpieza(res_limpieza, g("03_limpieza_datos.png"))
    graficar_boxplots_atipicos(datos_previos, X, g("04_boxplots_atipicos.png"))
    graficar_tabla_metricas(tabla, g("05_tabla_metricas.png"))
    graficar_comparacion_metricas(tabla, g("06_comparacion_metricas.png"))
    graficar_matrices(tabla, g("07_matrices_confusion.png"))
    graficar_boxplot_vc(detalle_vc, tabla, g("08_validacion_cruzada_boxplot.png"))
    graficar_roc(curvas, tabla, g("09_curvas_roc.png"))


# CONFIGURACION PARA LA INTERFAZ

def guardar_config(tabla, X, y_train, y_test, res_limpieza):
    categorias = {
        col: sorted(X[col].dropna().unique().tolist()) for col in COLUMNAS_CATEGORICAS
    }
    config = {
        "version": 2,
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "umbral": UMBRAL,
        "criterio_ranking": CRITERIO,
        "mejor_modelo": tabla.loc[0, "modelo"],
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
            "filas_entrenamiento": int(len(y_train)),
            "filas_prueba": int(len(y_test)),
            "morosos_prueba": int(y_test.sum()),
        },
        "validacion_cruzada": {
            "tipo": "RepeatedStratifiedKFold",
            "k_folds": K_FOLDS,
            "repeticiones": N_REPETICIONES,
            "evaluaciones_por_modelo": K_FOLDS * N_REPETICIONES,
        },
        "limpieza": res_limpieza,
        "columnas_categoricas": COLUMNAS_CATEGORICAS,
        "columnas_numericas": COLUMNAS_NUMERICAS,
        "categorias": categorias,
        "entorno": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "modelos": json.loads(tabla.to_json(orient="records")),
    }
    with open(RUTA_CONFIG, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# PROGRAMA PRINCIPAL

def main():
    ruta_manual = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 66)
    print(" ENTRENAMIENTO COMPARATIVO DE SEIS ALGORITMOS - MOROSIDAD")
    print("=" * 66)
    print(f" Python {platform.python_version()} | scikit-learn {sklearn.__version__} | "
          f"pandas {pd.__version__}")

    # PASO 1: LECTURA DE DATOS
    print("\n[1/6] Localizando y leyendo el conjunto de datos...")
    ruta, origen = localizar_csv(ruta_manual)
    print(f"  Origen: {origen}\n  Archivo: {ruta}")
    df = leer_csv(ruta)

    # PASO 2: LIMPIEZA DE DATOS (DUPLICADOS Y ATÍPICOS)
    print("\n[2/6] Limpiando datos (duplicados y valores atipicos, comun a los seis modelos)...")
    X, y, res_limpieza, datos_previos = limpiar_datos(df)
    imprimir_resumen_limpieza(res_limpieza)
    print("\n  Datos despues de la limpieza:")
    informe_datos(X, y)

    # PASO 3: PARTICIÓN ENTRENAMIENTO / PRUEBA
    print("\n[3/6] Dividiendo en entrenamiento y prueba "
          f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=y)...")
    # PARTICIÓN ESTRATIFICADA 80 % / 20 %
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Entrenamiento: {len(X_train)} filas  |  Prueba: {len(X_test)} filas "
          f"({int((y_test == 0).sum())} no morosos, {int(y_test.sum())} morosos)")

    # PASO 4: VALIDACIÓN CRUZADA Y ENTRENAMIENTO DE LOS SEIS MODELOS
    print(f"\n[4/6] Validando (K-fold estratificado, K={K_FOLDS}, {N_REPETICIONES} repeticiones = "
          f"{K_FOLDS * N_REPETICIONES} evaluaciones por modelo) y entrenando los seis algoritmos...")
    print("  (el SVM es el mas lento; puede tardar varios minutos)")
    os.makedirs(CARPETA_MODELOS, exist_ok=True)

    metricas_modelos = []
    detalle_vc = []
    curvas = {}
    modelos = definir_modelos()
    for i, cfg in enumerate(modelos, start=1):
        print(f"  ({i}/{len(modelos)}) {cfg['nombre']}: validacion cruzada...", end=" ", flush=True)
        # VALIDACIÓN CRUZADA DEL MODELO
        filas_vc = validacion_cruzada(cfg, X_train, y_train)
        detalle_vc.extend(filas_vc)
        resumen_vc = resumir_validacion_cruzada(filas_vc)
        print(f"F1 = {resumen_vc['cv_f1_media'] * 100:.2f}% ± {resumen_vc['cv_f1_std'] * 100:.2f}% | "
              "entrenamiento final...", end=" ", flush=True)

        # ENTRENAMIENTO FINAL Y EVALUACIÓN EN PRUEBA
        pipeline, metricas, probabilidad = entrenar_y_evaluar(cfg, X_train, X_test, y_train, y_test)
        metricas.update(resumen_vc)
        joblib.dump(pipeline, os.path.join(CARPETA_MODELOS, cfg["archivo"]))
        metricas_modelos.append(metricas)

        # CURVA ROC DEL MODELO
        fpr, tpr, _ = roc_curve(y_test, probabilidad)
        curvas[cfg["nombre"]] = (fpr, tpr)
        print(f"listo en {metricas['tiempo_entrenamiento_s']} s "
              f"(F1 prueba = {metricas['f1_moroso'] * 100:.2f}%)")

    # PASO 5: GUARDADO DE TABLAS Y CONFIGURACIÓN
    print("\n[5/6] Guardando tablas y configuracion...")
    tabla = construir_tabla(metricas_modelos)
    tabla.to_csv(RUTA_CSV_RESULTADOS, index=False, encoding="utf-8-sig")
    guardar_csv_vc(detalle_vc, tabla)
    guardar_config(tabla, X, y_train, y_test, res_limpieza)
    print(f"  Modelos:           {CARPETA_MODELOS}")
    print(f"  Tabla (CSV):       {RUTA_CSV_RESULTADOS}")
    print(f"  Validacion cruzada:{RUTA_CSV_VC_RESUMEN}")
    print(f"  VC por iteracion:  {RUTA_CSV_VC_DETALLE}")
    print(f"  Configuracion:     {RUTA_CONFIG}")

    # PASO 6: GENERACIÓN DE GRÁFICOS
    print("\n[6/6] Generando graficos en la carpeta 'Graficos'...")
    generar_graficos(tabla, detalle_vc, curvas, res_limpieza, datos_previos, X)
    print(f"  Graficos (PNG):    {CARPETA_GRAFICOS}")
    for archivo in sorted(os.listdir(CARPETA_GRAFICOS)):
        print(f"    - {archivo}")

    print("\n" + "=" * 66)
    print(f" RESULTADOS EN EL CONJUNTO DE PRUEBA (umbral {UMBRAL:.0%})")
    print(f" Orden: {CRITERIO}")
    print("=" * 66)
    imprimir_tabla(tabla)

    print("\n" + "=" * 66)
    print(f" VALIDACION CRUZADA (K={K_FOLDS}, {N_REPETICIONES} repeticiones): media ± desviacion estandar")
    print("=" * 66)
    imprimir_tabla_vc(tabla)

    print(f"\n Modelo mejor evaluado segun el criterio: {tabla.loc[0, 'modelo']}")
    print(" Recuerda: con clases desbalanceadas no basta el Accuracy. Si el costo de no")
    print(" detectar a un moroso (falso negativo) es alto, prioriza el Recall; si el costo")
    print(" de rechazar a un buen cliente (falso positivo) es alto, prioriza la Precision.")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"\nERROR: {error}")
        sys.exit(1)