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
)
from sklearn.model_selection import train_test_split
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
RUTA_CSV_RESULTADOS = os.path.join(CARPETA, "resultados_comparacion.csv")
RUTA_PNG_MATRICES = os.path.join(CARPETA, "matrices_confusion.png")
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

TEST_SIZE = 0.20
RANDOM_STATE = 42
UMBRAL = 0.50  # Un cliente se clasifica como moroso si P(moroso) >= UMBRAL

CRITERIO = "F1-score de la clase moroso (desempate: Recall moroso, luego ROC-AUC)"

# Referencia registrada del modelo ORIGINAL (solo para comprobar la limpieza).
# Matriz [[TN, FP], [FN, TP]] sobre el conjunto de prueba.
REFERENCIA_ORIGINAL = {"matriz": [[1656, 45], [75, 250]], "accuracy": 0.9408}


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


def limpiar_datos(df):
    df = df.copy()

    faltan = [c for c in COLUMNAS_MODELO + [TARGET] if c not in df.columns]
    if faltan:
        raise ValueError(
            f"Faltan columnas en el CSV: {faltan}\nColumnas encontradas: {list(df.columns)}"
        )

    # Variable objetivo
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    antes = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    if len(df) < antes:
        print(f"  Aviso: {antes - len(df)} filas sin valor en '{TARGET}' fueron descartadas.")
    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"'{TARGET}' debe contener solo 0 y 1: {sorted(df[TARGET].unique())}")

    # Categoricas: solo se quitan espacios (se conservan los valores originales, p. ej. 'na')
    for col in COLUMNAS_CATEGORICAS:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)

    # Numericas: conversion robusta (el identificador 'id' NO se usa como predictor)
    for col in COLUMNAS_NUMERICAS:
        df[col] = df[col].map(convertir_numero).astype(float)

    X = df[COLUMNAS_MODELO].copy()
    y = df[TARGET].copy()
    return X, y


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


def definir_modelos():
    return [
        {
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
            "nombre": "Regresión logística",
            "archivo": "regresion_logistica.pkl",
            "escalar": True,
            "pesos": False,
            "clasificador": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
            ),
        },
        {
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
            "nombre": "KNN",
            "archivo": "knn.pkl",
            "escalar": True,
            "pesos": False,  # KNN no admite pesos de clase
            "clasificador": KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1,
            ),
        },
    ]


def entrenar_y_evaluar(cfg, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ("preprocesamiento", crear_preprocesador(cfg["escalar"])),
        ("modelo", cfg["clasificador"]),
    ])

    inicio = time.perf_counter()
    if cfg["pesos"]:
        pesos = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, modelo__sample_weight=pesos)
    else:
        pipeline.fit(X_train, y_train)
    tiempo = time.perf_counter() - inicio

    # Misma regla de decision para todos: P(moroso) >= UMBRAL
    probabilidad = pipeline.predict_proba(X_test)[:, 1]
    prediccion = (probabilidad >= UMBRAL).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, prediccion, labels=[0, 1]).ravel()

    try:
        auc = float(roc_auc_score(y_test, probabilidad))
    except ValueError:
        auc = float("nan")

    metricas = {
        "modelo": cfg["nombre"],
        "archivo": cfg["archivo"],
        "accuracy": float(accuracy_score(y_test, prediccion)),
        "precision_moroso": float(precision_score(y_test, prediccion, pos_label=1, zero_division=0)),
        "recall_moroso": float(recall_score(y_test, prediccion, pos_label=1, zero_division=0)),
        "f1_moroso": float(f1_score(y_test, prediccion, pos_label=1, zero_division=0)),
        "roc_auc": auc,
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp),
        "tiempo_entrenamiento_s": round(float(tiempo), 2),
    }
    return pipeline, metricas


# SALIDAS: TABLA, GRAFICAS Y CONFIGURACION

def construir_tabla(lista_metricas):
    tabla = pd.DataFrame(lista_metricas)
    tabla = tabla.sort_values(
        by=["f1_moroso", "recall_moroso", "roc_auc"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    tabla.insert(0, "ranking", np.arange(1, len(tabla) + 1))
    return tabla


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


def guardar_config(tabla, X, y_train, y_test):
    categorias = {
        col: sorted(X[col].dropna().unique().tolist()) for col in COLUMNAS_CATEGORICAS
    }
    config = {
        "version": 1,
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


def comprobar_contra_original(tabla):
    fila = tabla[tabla["modelo"] == "Random Forest"].iloc[0]
    matriz = [
        [int(fila["verdaderos_negativos"]), int(fila["falsos_positivos"])],
        [int(fila["falsos_negativos"]), int(fila["verdaderos_positivos"])],
    ]
    print(f"  Random Forest (este script): {matriz}  Accuracy {fila['accuracy'] * 100:.2f}%")
    print(f"  Modelo original (registrado): {REFERENCIA_ORIGINAL['matriz']}  "
          f"Accuracy {REFERENCIA_ORIGINAL['accuracy'] * 100:.2f}%")
    if matriz == REFERENCIA_ORIGINAL["matriz"]:
        print("  -> Coinciden: la limpieza y el split reproducen el entrenamiento original.")
    else:
        print("  -> NO coinciden exactamente. Puede deberse a diferencias de limpieza, de "
              "version de scikit-learn o de datos.\n"
              "     Revisa los resumenes de arriba y comparte esta salida para ajustarlo. "
              "Los resultados siguen siendo validos para comparar los seis algoritmos "
              "entre si.")


# PROGRAMA PRINCIPAL

def main():
    ruta_manual = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 66)
    print(" ENTRENAMIENTO COMPARATIVO DE SEIS ALGORITMOS - MOROSIDAD")
    print("=" * 66)
    print(f" Python {platform.python_version()} | scikit-learn {sklearn.__version__} | "
          f"pandas {pd.__version__}")

    print("\n[1/5] Localizando y leyendo el conjunto de datos...")
    ruta, origen = localizar_csv(ruta_manual)
    print(f"  Origen: {origen}\n  Archivo: {ruta}")
    df = leer_csv(ruta)

    print("\n[2/5] Limpiando datos (comun a los seis modelos)...")
    X, y = limpiar_datos(df)
    informe_datos(X, y)

    print("\n[3/5] Dividiendo en entrenamiento y prueba "
          f"(test_size={TEST_SIZE}, random_state={RANDOM_STATE}, stratify=y)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Entrenamiento: {len(X_train)} filas  |  Prueba: {len(X_test)} filas "
          f"({int((y_test == 0).sum())} no morosos, {int(y_test.sum())} morosos)")

    print("\n[4/5] Entrenando y evaluando los seis algoritmos...")
    os.makedirs(CARPETA_MODELOS, exist_ok=True)

    metricas_modelos = []
    modelos = definir_modelos()
    for i, cfg in enumerate(modelos, start=1):
        print(f"  ({i}/{len(modelos)}) {cfg['nombre']}...", end=" ", flush=True)
        pipeline, metricas = entrenar_y_evaluar(cfg, X_train, X_test, y_train, y_test)
        joblib.dump(pipeline, os.path.join(CARPETA_MODELOS, cfg["archivo"]))
        metricas_modelos.append(metricas)
        print(f"listo en {metricas['tiempo_entrenamiento_s']} s "
              f"(F1 moroso = {metricas['f1_moroso'] * 100:.2f}%)")

    print("\n[5/5] Guardando resultados...")
    tabla = construir_tabla(metricas_modelos)
    tabla.to_csv(RUTA_CSV_RESULTADOS, index=False, encoding="utf-8-sig")
    graficar_matrices(tabla, RUTA_PNG_MATRICES)
    guardar_config(tabla, X, y_train, y_test)
    print(f"  Modelos:        {CARPETA_MODELOS}")
    print(f"  Tabla (CSV):    {RUTA_CSV_RESULTADOS}")
    print(f"  Matrices (PNG): {RUTA_PNG_MATRICES}")
    print(f"  Configuracion:  {RUTA_CONFIG}")

    print("\n" + "=" * 66)
    print(f" RESULTADOS (conjunto de prueba, umbral {UMBRAL:.0%})")
    print(f" Orden: {CRITERIO}")
    print("=" * 66)
    imprimir_tabla(tabla)

    print(f"\n Modelo mejor evaluado segun el criterio: {tabla.loc[0, 'modelo']}")
    print(" Recuerda: con clases desbalanceadas no basta el Accuracy. Si el costo de no")
    print(" detectar a un moroso (falso negativo) es alto, prioriza el Recall; si el costo")
    print(" de rechazar a un buen cliente (falso positivo) es alto, prioriza la Precision.")

    print("\n" + "-" * 66)
    print(" Comprobacion contra el modelo original (solo referencia)")
    print("-" * 66)
    comprobar_contra_original(tabla)
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"\nERROR: {error}")
        sys.exit(1)