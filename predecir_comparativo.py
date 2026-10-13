import json
import math
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd


# CONFIGURACIÓN

def resource_path(relative_path):
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


CARPETA_MODELOS = resource_path("modelos_comparativos")
RUTA_CONFIG = os.path.join(CARPETA_MODELOS, "config_comparativo.json")

# Modelos esperados (se usan si no se encuentra config_comparativo.json)
MODELOS_POR_DEFECTO = [
    ("Random Forest", "random_forest.pkl"),
    ("Regresión logística", "regresion_logistica.pkl"),
    ("Árbol de decisión", "arbol_decision.pkl"),
    ("Gradient Boosting", "gradient_boosting.pkl"),
    ("SVM", "svm.pkl"),
    ("KNN", "knn.pkl"),
]

# Paleta de colores
NAVY = "#0f2a4a"
ACCENT = "#2563eb"
BG = "#eef2f7"
CARD = "#ffffff"
BORDER = "#d9e1ec"
TEXT = "#1f2937"
MUTED = "#6b7280"
SUCCESS = "#16a34a"
WARNING = "#f59e0b"
DANGER = "#dc2626"

COLUMNAS_CATEGORICAS = [
    "sexo", "escolaridade", "estado_civil", "salario_anual", "tipo_cartao"
]

COLUMNAS_MODELO = [
    "idade", "sexo", "dependentes", "escolaridade", "estado_civil",
    "salario_anual", "tipo_cartao", "meses_de_relacionamento",
    "qtd_produtos", "iteracoes_12m", "meses_inativo_12m",
    "limite_credito", "valor_transacoes_12m", "qtd_transacoes_12m"
]

NOMBRES_AMIGABLES = {
    "idade": "Edad",
    "sexo": "Sexo",
    "dependentes": "Dependientes",
    "escolaridade": "Escolaridad",
    "estado_civil": "Estado civil",
    "salario_anual": "Salario anual",
    "tipo_cartao": "Tipo de tarjeta",
    "meses_de_relacionamento": "Meses de relación",
    "qtd_produtos": "Cantidad de productos",
    "iteracoes_12m": "Interacciones (12 m)",
    "meses_inativo_12m": "Meses inactivo (12 m)",
    "limite_credito": "Límite de crédito",
    "valor_transacoes_12m": "Valor de transacciones",
    "qtd_transacoes_12m": "Cantidad de transacciones"
}

# Categorías reales con las que se entrenaron los modelos (valores del CSV).
# Si existe config_comparativo.json se usan las que quedaron registradas allí.
CATEGORIAS_POR_DEFECTO = {
    "sexo": ["M", "F"],
    "escolaridade": [
        "graduacao", "ensino medio", "mestrado", "doutorado",
        "sem educacao formal", "na"
    ],
    "estado_civil": ["casado", "solteiro", "divorciado", "na"],
    "salario_anual": [
        "menos que $40K", "$40K - $60K", "$60K - $80K",
        "$80K - $120K", "$120K +", "na"
    ],
    "tipo_cartao": ["blue", "silver", "gold", "platinum"],
}

# Texto que ve el usuario para cada valor real del CSV
ETIQUETAS = {
    "sexo": {"M": "Masculino", "F": "Femenino"},
    "escolaridade": {
        "graduacao": "Universitaria (graduación)",
        "ensino medio": "Secundaria",
        "mestrado": "Maestría",
        "doutorado": "Doctorado",
        "sem educacao formal": "Sin educación formal",
        "na": "No informado",
    },
    "estado_civil": {
        "casado": "Casado/a",
        "solteiro": "Soltero/a",
        "divorciado": "Divorciado/a",
        "na": "No informado",
    },
    "salario_anual": {
        "menos que $40K": "Menos de $40K",
        "$40K - $60K": "$40K - $60K",
        "$60K - $80K": "$60K - $80K",
        "$80K - $120K": "$80K - $120K",
        "$120K +": "Más de $120K",
        "na": "No informado",
    },
    "tipo_cartao": {
        "blue": "Blue", "silver": "Silver", "gold": "Gold", "platinum": "Platinum"
    },
}


# CARGAR CONFIGURACIÓN Y MODELOS

CONFIG = None
ERROR_CONFIG = None

try:
    with open(RUTA_CONFIG, encoding="utf-8") as archivo_config:
        CONFIG = json.load(archivo_config)
except Exception as e:
    ERROR_CONFIG = str(e)

UMBRAL = float(CONFIG.get("umbral", 0.50)) if CONFIG else 0.50
MEJOR_MODELO = CONFIG.get("mejor_modelo") if CONFIG else None

# Lista de modelos en el orden del ranking (si hay configuración)
if CONFIG and CONFIG.get("modelos"):
    LISTA_MODELOS = [
        {"nombre": m["modelo"], "archivo": m["archivo"], "metricas": m}
        for m in CONFIG["modelos"]
    ]
else:
    LISTA_MODELOS = [
        {"nombre": n, "archivo": a, "metricas": None}
        for n, a in MODELOS_POR_DEFECTO
    ]

MODELOS = {}          # nombre -> pipeline cargado
ERRORES_CARGA = {}    # nombre -> motivo del fallo

for modelo_info in LISTA_MODELOS:

    ruta_modelo = os.path.join(CARPETA_MODELOS, modelo_info["archivo"])

    try:
        pipeline = joblib.load(ruta_modelo)

        # Para predecir una sola fila no hace falta paralelizar (es más rápido)
        estimador = pipeline.named_steps.get("modelo")
        if type(estimador).__name__ in ("RandomForestClassifier", "KNeighborsClassifier"):
            try:
                pipeline.set_params(modelo__n_jobs=1)
            except Exception:
                pass

        MODELOS[modelo_info["nombre"]] = pipeline

    except Exception as e:
        ERRORES_CARGA[modelo_info["nombre"]] = str(e)

NOMBRES_CARGADOS = [m["nombre"] for m in LISTA_MODELOS if m["nombre"] in MODELOS]

if MEJOR_MODELO not in MODELOS:
    MEJOR_MODELO_VISIBLE = None
else:
    MEJOR_MODELO_VISIBLE = MEJOR_MODELO


def ordenar_categorias(columna, valores):
    base = CATEGORIAS_POR_DEFECTO[columna]
    return [v for v in base if v in valores] + [v for v in valores if v not in base]


CATEGORIAS = {}

for _col in COLUMNAS_CATEGORICAS:
    _desde_config = []
    if CONFIG:
        _desde_config = CONFIG.get("categorias", {}).get(_col, [])
    CATEGORIAS[_col] = (
        ordenar_categorias(_col, _desde_config)
        if _desde_config else list(CATEGORIAS_POR_DEFECTO[_col])
    )


def etiqueta(columna, valor):
    return ETIQUETAS.get(columna, {}).get(valor, valor)


OPCIONES = {
    col: [etiqueta(col, v) for v in CATEGORIAS[col]] for col in COLUMNAS_CATEGORICAS
}
VALOR_REAL = {
    col: {etiqueta(col, v): v for v in CATEGORIAS[col]} for col in COLUMNAS_CATEGORICAS
}


def valor_por_defecto(columna, preferido):
    valor = preferido if preferido in CATEGORIAS[columna] else CATEGORIAS[columna][0]
    return etiqueta(columna, valor)


# Estado de la última predicción
ULTIMO = {"probs": {}}
_CACHE_IMPORTANCIAS = {}


def calcular_importancias(nombre):
    if nombre in _CACHE_IMPORTANCIAS:
        return _CACHE_IMPORTANCIAS[nombre]

    resultado = (None, "")

    try:
        pipeline = MODELOS[nombre]
        preprocesador = pipeline.named_steps["preprocesamiento"]
        estimador = pipeline.named_steps["modelo"]

        valores = None
        nota = ""

        if hasattr(estimador, "feature_importances_"):
            valores = [float(v) for v in estimador.feature_importances_]
            nota = f"Importancia global de {nombre} (no específica de este cliente)"

        elif hasattr(estimador, "coef_"):
            valores = [abs(float(v)) for v in estimador.coef_[0]]
            nota = "Peso absoluto de los coeficientes (global, no específico de este cliente)"

        if valores is not None:

            total = sum(valores) or 1.0
            nombres = preprocesador.get_feature_names_out()
            acumulado = {}

            for nombre_col, valor in zip(nombres, valores):
                limpio = nombre_col.split("__", 1)[-1]

                candidatas = [
                    c for c in COLUMNAS_MODELO
                    if limpio == c or limpio.startswith(c + "_")
                ]

                columna = max(candidatas, key=len) if candidatas else limpio
                acumulado[columna] = acumulado.get(columna, 0) + valor / total

            ordenado = sorted(
                acumulado.items(),
                key=lambda item: item[1],
                reverse=True
            )

            resultado = (ordenado[:8], nota)

    except Exception as e:
        print(f"No se pudo calcular la importancia de variables ({nombre}):", e)

    _CACHE_IMPORTANCIAS[nombre] = resultado
    return resultado


# VENTANA PRINCIPAL

root = tk.Tk()
root.title("Sistema de Predicción de Morosidad - Comparación de modelos")
root.configure(bg=BG)

pantalla_w = root.winfo_screenwidth()
pantalla_h = root.winfo_screenheight()

ancho = min(1120, pantalla_w - 60)
alto = min(780, pantalla_h - 120)
pos_x = (pantalla_w - ancho) // 2
pos_y = max((pantalla_h - alto) // 2 - 20, 0)

root.geometry(f"{ancho}x{alto}+{pos_x}+{pos_y}")
root.minsize(940, 520)


# ESTILOS

style = ttk.Style()

try:
    style.theme_use("clam")
except Exception:
    pass

root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))
root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
root.option_add("*TCombobox*Listbox.selectForeground", "white")

style.configure(
    "TEntry",
    padding=7,
    font=("Segoe UI", 10),
    fieldbackground="white",
    foreground=TEXT,
    bordercolor=BORDER,
    lightcolor=BORDER,
    darkcolor=BORDER
)

style.map(
    "TEntry",
    bordercolor=[("focus", ACCENT)],
    lightcolor=[("focus", ACCENT)],
    darkcolor=[("focus", ACCENT)]
)

style.configure(
    "TCombobox",
    padding=7,
    font=("Segoe UI", 10),
    fieldbackground="white",
    background="#e2e8f0",
    foreground=TEXT,
    arrowcolor=TEXT,
    bordercolor=BORDER,
    lightcolor=BORDER,
    darkcolor=BORDER
)

style.map(
    "TCombobox",
    fieldbackground=[("readonly", "white")],
    foreground=[("readonly", TEXT)],
    selectbackground=[("readonly", "white")],
    selectforeground=[("readonly", TEXT)],
    bordercolor=[("focus", ACCENT)]
)

style.configure(
    "Vertical.TScrollbar",
    background="#cbd5e1",
    troughcolor=BG,
    bordercolor=BG,
    lightcolor="#cbd5e1",
    darkcolor="#cbd5e1",
    arrowcolor="#475569",
    relief="flat"
)

style.map(
    "Vertical.TScrollbar",
    background=[("active", "#94a3b8")]
)

# Tabla comparativa
style.configure(
    "Treeview",
    font=("Segoe UI", 10),
    rowheight=32,
    background="white",
    fieldbackground="white",
    foreground=TEXT,
    bordercolor=BORDER,
    lightcolor=BORDER,
    darkcolor=BORDER
)

style.configure(
    "Treeview.Heading",
    font=("Segoe UI", 10, "bold"),
    background=NAVY,
    foreground="white",
    padding=(6, 9),
    relief="flat",
    borderwidth=0
)

style.map(
    "Treeview.Heading",
    background=[("active", "#1e3a5f")]
)

style.map(
    "Treeview",
    background=[("selected", "#bfdbfe")],
    foreground=[("selected", TEXT)]
)


def estilo_boton(nombre, base, hover, presionado):

    style.configure(
        nombre,
        font=("Segoe UI", 10, "bold"),
        padding=(24, 11),
        background=base,
        foreground="white",
        borderwidth=0,
        bordercolor=base,
        lightcolor=base,
        darkcolor=base,
        focuscolor=base,
        focusthickness=0,
        relief="flat"
    )

    colores = [("pressed", presionado), ("active", hover)]

    style.map(
        nombre,
        background=colores,
        bordercolor=colores,
        lightcolor=colores,
        darkcolor=colores,
        foreground=[("!disabled", "white")]
    )


estilo_boton("Primario.TButton", "#1d4ed8", "#1e40af", "#1e3a8a")
estilo_boton("Verde.TButton", "#0f766e", "#115e59", "#134e4a")
estilo_boton("Gris.TButton", "#475569", "#334155", "#1e293b")


# VARIABLES

idade_var = tk.StringVar()
sexo_var = tk.StringVar(value=valor_por_defecto("sexo", "M"))
dependentes_var = tk.StringVar()

escolaridade_var = tk.StringVar(value=valor_por_defecto("escolaridade", "graduacao"))
estado_civil_var = tk.StringVar(value=valor_por_defecto("estado_civil", "casado"))
salario_var = tk.StringVar(value=valor_por_defecto("salario_anual", "$60K - $80K"))
tipo_cartao_var = tk.StringVar(value=valor_por_defecto("tipo_cartao", "blue"))

meses_relacionamiento_var = tk.StringVar()
qtd_produtos_var = tk.StringVar()
iteracoes_var = tk.StringVar()
meses_inativo_var = tk.StringVar()

limite_credito_var = tk.StringVar()
valor_transacoes_var = tk.StringVar()
qtd_transacoes_var = tk.StringVar()

# Modelo cuyo detalle se muestra (por defecto, el mejor evaluado)
if MEJOR_MODELO_VISIBLE:
    _modelo_inicial = MEJOR_MODELO_VISIBLE
elif NOMBRES_CARGADOS:
    _modelo_inicial = NOMBRES_CARGADOS[0]
else:
    _modelo_inicial = ""

modelo_var = tk.StringVar(value=_modelo_inicial)


# ENCABEZADO

header = tk.Frame(root, bg=NAVY)
header.pack(side="top", fill="x")

header_texto = tk.Frame(header, bg=NAVY)
header_texto.pack(side="left", padx=30, pady=18)

tk.Label(
    header_texto,
    text="Sistema de Predicción de Morosidad",
    bg=NAVY,
    fg="white",
    font=("Segoe UI", 20, "bold")
).pack(anchor="w")

tk.Label(
    header_texto,
    text="Comparación de algoritmos de Machine Learning sobre un mismo cliente",
    bg=NAVY,
    fg="#bfdbfe",
    font=("Segoe UI", 10)
).pack(anchor="w", pady=(3, 0))

if len(MODELOS) == len(LISTA_MODELOS) and MODELOS:
    texto_estado, color_estado = f"●  {len(MODELOS)} de {len(LISTA_MODELOS)} modelos cargados", "#4ade80"
elif MODELOS:
    texto_estado, color_estado = f"●  {len(MODELOS)} de {len(LISTA_MODELOS)} modelos cargados", "#fcd34d"
else:
    texto_estado, color_estado = "●  Modelos no encontrados", "#fca5a5"

estado_modelo = tk.Label(
    header,
    text=texto_estado,
    bg=NAVY,
    fg=color_estado,
    font=("Segoe UI", 10, "bold")
)
estado_modelo.pack(side="right", padx=30)


# BARRA INFERIOR FIJA (botones + información de los modelos)

barra_inferior = tk.Frame(root, bg="white")
barra_inferior.pack(side="bottom", fill="x")

tk.Frame(barra_inferior, bg=BORDER, height=1).pack(side="top", fill="x")

botones_frame = tk.Frame(barra_inferior, bg="white")
botones_frame.pack(side="left", padx=25, pady=12)

_texto_footer = f"Umbral: {UMBRAL * 100:.0f}%"

if MEJOR_MODELO_VISIBLE and CONFIG:
    _mejor = next(
        (m for m in CONFIG["modelos"] if m["modelo"] == MEJOR_MODELO_VISIBLE), None
    )
    if _mejor:
        _texto_footer = (
            f"Mejor modelo: {MEJOR_MODELO_VISIBLE} (F1 {_mejor['f1_moroso'] * 100:.1f}%)"
            f"   |   {_texto_footer}"
        )

footer = tk.Label(
    barra_inferior,
    text=_texto_footer,
    bg="white",
    fg=MUTED,
    font=("Segoe UI", 9)
)
footer.pack(side="right", padx=25)


# ÁREA CON SCROLL

area = tk.Frame(root, bg=BG)
area.pack(side="top", fill="both", expand=True)

lienzo = tk.Canvas(
    area,
    bg=BG,
    highlightthickness=0,
    yscrollincrement=24
)

scrollbar = ttk.Scrollbar(
    area,
    orient="vertical",
    command=lienzo.yview
)

lienzo.configure(yscrollcommand=scrollbar.set)

scrollbar.pack(side="right", fill="y")
lienzo.pack(side="left", fill="both", expand=True)

contenido = tk.Frame(lienzo, bg=BG)

ventana_contenido = lienzo.create_window(
    (0, 0),
    window=contenido,
    anchor="nw"
)


def actualizar_scrollregion(event=None):
    lienzo.configure(scrollregion=lienzo.bbox("all"))


def ajustar_ancho(event):
    lienzo.itemconfig(ventana_contenido, width=event.width)


contenido.bind("<Configure>", actualizar_scrollregion)
lienzo.bind("<Configure>", ajustar_ancho)


def on_mousewheel(event):

    if contenido.winfo_reqheight() <= lienzo.winfo_height():
        return

    if event.num == 4:
        lienzo.yview_scroll(-3, "units")
    elif event.num == 5:
        lienzo.yview_scroll(3, "units")
    else:
        lienzo.yview_scroll(int(-event.delta / 120) * 3, "units")


def on_mousewheel_combo(event):
    # Evita que la rueda cambie el valor de un Combobox y desplaza la página
    on_mousewheel(event)
    return "break"


root.bind_all("<MouseWheel>", on_mousewheel)
root.bind_all("<Button-4>", on_mousewheel)
root.bind_all("<Button-5>", on_mousewheel)
root.bind_class("TCombobox", "<MouseWheel>", on_mousewheel_combo)

# Margen interno del contenido
cuerpo_pagina = tk.Frame(contenido, bg=BG)
cuerpo_pagina.pack(fill="x", padx=28, pady=22)


# COMPONENTES REUTILIZABLES

def crear_tarjeta(parent, titulo, subtitulo, columnas=3, mostrar=True):

    tarjeta = tk.Frame(
        parent,
        bg=CARD,
        highlightbackground=BORDER,
        highlightthickness=1
    )

    if mostrar:
        tarjeta.pack(fill="x", pady=(0, 16))

    encabezado = tk.Frame(tarjeta, bg=CARD)
    encabezado.pack(fill="x", padx=20, pady=(16, 4))

    tk.Frame(encabezado, bg=ACCENT, width=4).pack(
        side="left", fill="y", padx=(0, 12)
    )

    textos = tk.Frame(encabezado, bg=CARD)
    textos.pack(side="left")

    tk.Label(
        textos,
        text=titulo,
        bg=CARD,
        fg=TEXT,
        font=("Segoe UI", 13, "bold")
    ).pack(anchor="w")

    tk.Label(
        textos,
        text=subtitulo,
        bg=CARD,
        fg=MUTED,
        font=("Segoe UI", 9)
    ).pack(anchor="w")

    cuerpo = tk.Frame(tarjeta, bg=CARD)
    cuerpo.pack(fill="x", padx=10, pady=(6, 12))

    for i in range(columnas):
        cuerpo.columnconfigure(i, weight=1, uniform=f"col_{titulo}")

    return tarjeta, cuerpo


def crear_campo(parent, fila, columna, texto, variable, valores=None):

    contenedor = tk.Frame(parent, bg=CARD)

    contenedor.grid(
        row=fila,
        column=columna,
        padx=12,
        pady=(4, 8),
        sticky="ew"
    )

    tk.Label(
        contenedor,
        text=texto,
        bg=CARD,
        fg="#374151",
        font=("Segoe UI", 10)
    ).pack(anchor="w", pady=(0, 3))

    if valores is not None:

        widget = ttk.Combobox(
            contenedor,
            textvariable=variable,
            values=valores,
            state="readonly"
        )

    else:

        widget = ttk.Entry(
            contenedor,
            textvariable=variable
        )

    widget.pack(fill="x")

    return widget


# TARJETA: INFORMACIÓN PERSONAL

tarjeta_personal, personal_body = crear_tarjeta(
    cuerpo_pagina,
    "Información personal",
    "Datos demográficos del cliente",
    columnas=3
)

crear_campo(personal_body, 0, 0, "Edad", idade_var)
crear_campo(personal_body, 0, 1, "Sexo", sexo_var, OPCIONES["sexo"])
crear_campo(personal_body, 0, 2, "Dependientes", dependentes_var)

crear_campo(
    personal_body, 1, 0, "Escolaridad", escolaridade_var,
    OPCIONES["escolaridade"]
)

crear_campo(
    personal_body, 1, 1, "Estado civil", estado_civil_var,
    OPCIONES["estado_civil"]
)

crear_campo(
    personal_body, 1, 2, "Salario anual", salario_var,
    OPCIONES["salario_anual"]
)

crear_campo(
    personal_body, 2, 0, "Tipo de tarjeta", tipo_cartao_var,
    OPCIONES["tipo_cartao"]
)


# TARJETA: COMPORTAMIENTO DEL CLIENTE

tarjeta_comportamiento, comportamiento_body = crear_tarjeta(cuerpo_pagina,"Comportamiento del cliente","Relación con la entidad y actividad reciente",columnas=2)

crear_campo(comportamiento_body, 0, 0, "Meses de relación", meses_relacionamiento_var)
crear_campo(comportamiento_body, 0, 1, "Cantidad de productos", qtd_produtos_var)
crear_campo(comportamiento_body, 1, 0, "Interacciones últimos 12 meses", iteracoes_var)
crear_campo(comportamiento_body, 1, 1, "Meses inactivo últimos 12 meses", meses_inativo_var)


# TARJETA: INFORMACIÓN FINANCIERA

tarjeta_financiera, financiero_body = crear_tarjeta(
    cuerpo_pagina,
    "Información financiera",
    "Línea de crédito y movimientos del último año",
    columnas=3
)

crear_campo(financiero_body, 0, 0, "Límite de crédito", limite_credito_var)
crear_campo(financiero_body, 0, 1, "Valor de transacciones (12 meses)", valor_transacoes_var)
crear_campo(financiero_body, 0, 2, "Cantidad de transacciones (12 meses)", qtd_transacoes_var)


# TARJETA: RESULTADO (oculta hasta predecir)

resultado_card, resultado_body = crear_tarjeta(
    cuerpo_pagina,
    "Resultado de la evaluación",
    "Detalle del modelo elegido y comparación de todos los algoritmos",
    columnas=1,
    mostrar=False
)

# --- Selector del modelo mostrado en detalle ---
fila_selector = tk.Frame(resultado_body, bg=CARD)
fila_selector.pack(fill="x", padx=10, pady=(4, 8))

tk.Label(
    fila_selector,
    text="Modelo mostrado en detalle:",
    bg=CARD,
    fg="#374151",
    font=("Segoe UI", 10, "bold")
).pack(side="left")

selector_modelo = ttk.Combobox(
    fila_selector,
    textvariable=modelo_var,
    values=NOMBRES_CARGADOS,
    state="readonly",
    width=28
)
selector_modelo.pack(side="left", padx=(10, 0))

tk.Label(
    fila_selector,
    text="(también puedes elegirlo haciendo clic en una fila de la tabla)",
    bg=CARD,
    fg=MUTED,
    font=("Segoe UI", 9)
).pack(side="left", padx=12)

# --- Veredicto del modelo elegido ---
fila_resultado = tk.Frame(resultado_body, bg=CARD)
fila_resultado.pack(fill="x", padx=10, pady=(4, 6))

resultado_badge = tk.Label(
    fila_resultado,
    text="",
    font=("Segoe UI", 15, "bold"),
    fg="white",
    bg=SUCCESS,
    padx=20,
    pady=7
)
resultado_badge.pack(side="left")

resultado_descripcion = tk.Label(
    fila_resultado,
    text="",
    bg=CARD,
    fg=TEXT,
    font=("Segoe UI", 10),
    justify="left",
    anchor="w"
)
resultado_descripcion.pack(side="left", padx=18)

# --- Gauge + variables influyentes ---
paneles = tk.Frame(resultado_body, bg=CARD)
paneles.pack(fill="x", padx=10, pady=(10, 4))

gauge_canvas = tk.Canvas(
    paneles,
    width=380,
    height=300,
    bg=CARD,
    highlightthickness=0
)
gauge_canvas.pack(side="left", padx=(0, 10))

tk.Frame(paneles, bg=BORDER, width=1).pack(side="left", fill="y", pady=10)

factores_canvas = tk.Canvas(
    paneles,
    height=300,
    bg=CARD,
    highlightthickness=0
)
factores_canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))

# --- Tabla comparativa (al final) ---
tk.Frame(resultado_body, bg=BORDER, height=1).pack(fill="x", padx=10, pady=(12, 10))

tk.Label(
    resultado_body,
    text="Comparación de modelos",
    bg=CARD,
    fg=TEXT,
    font=("Segoe UI", 12, "bold")
).pack(anchor="w", padx=12)

tk.Label(
    resultado_body,
    text="Predicción para este cliente y desempeño de cada algoritmo en el conjunto de prueba",
    bg=CARD,
    fg=MUTED,
    font=("Segoe UI", 9)
).pack(anchor="w", padx=12, pady=(0, 8))

resumen_label = tk.Label(
    resultado_body,
    text="",
    bg=CARD,
    fg=TEXT,
    font=("Segoe UI", 10, "bold"),
    justify="left",
    anchor="w"
)
resumen_label.pack(fill="x", padx=12, pady=(0, 8))

COLUMNAS_TABLA = [
    ("rank", "#", 44, "center"),
    ("modelo", "Modelo", 210, "w"),
    ("clase", "Clasificación", 140, "center"),
    ("prob", "Prob. morosidad", 130, "center"),
    ("acc", "Accuracy", 90, "center"),
    ("prec", "Precisión", 90, "center"),
    ("rec", "Recall", 90, "center"),
    ("f1", "F1", 90, "center"),
    ("auc", "ROC-AUC", 90, "center"),
]

tabla = ttk.Treeview(
    resultado_body,
    columns=[c[0] for c in COLUMNAS_TABLA],
    show="headings",
    height=max(len(LISTA_MODELOS), 1),
    selectmode="browse"
)

for clave, titulo, ancho_col, alineacion in COLUMNAS_TABLA:
    tabla.heading(clave, text=titulo)
    tabla.column(clave, width=ancho_col, minwidth=60, anchor=alineacion, stretch=True)

tabla.tag_configure("moroso", background="#fee2e2")
tabla.tag_configure("nomoroso", background="#dcfce7")
tabla.tag_configure("nodisp", background="#f3f4f6", foreground=MUTED)
tabla.tag_configure("mejor", font=("Segoe UI", 10, "bold"))

tabla.pack(fill="x", padx=12, pady=(0, 8))

nota_label = tk.Label(
    resultado_body,
    text=(
        "Las probabilidades son estimaciones de cada modelo; no garantizan que el cliente "
        "incumplirá sus pagos ni sustituyen el criterio profesional.\n"
        "Precisión, Recall y F1 se refieren a la clase moroso y provienen del conjunto de "
        "prueba del entrenamiento. ★ = modelo mejor evaluado según " +
        (CONFIG.get("criterio_ranking", "F1-score de la clase moroso") if CONFIG
         else "F1-score de la clase moroso") + "."
    ),
    bg=CARD,
    fg=MUTED,
    font=("Segoe UI", 9),
    justify="left",
    anchor="w",
    wraplength=900
)
nota_label.pack(fill="x", padx=12, pady=(0, 6))


def ajustar_notas(event):
    nota_label.config(wraplength=max(event.width - 40, 300))


resultado_body.bind("<Configure>", ajustar_notas)


# GRÁFICOS

def color_nivel(prob):

    if prob < 0.30:
        return SUCCESS, "Riesgo bajo"

    if prob < UMBRAL:
        return WARNING, "Riesgo moderado"

    return DANGER, "Riesgo alto"


def dibujar_gauge(prob):

    c = gauge_canvas
    c.delete("all")

    cx, cy, r = 190, 185, 125
    bbox = (cx - r, cy - r, cx + r, cy + r)

    zonas = [
        (0.00, min(0.30, UMBRAL), "#22c55e"),
        (min(0.30, UMBRAL), UMBRAL, "#f59e0b"),
        (UMBRAL, 1.00, "#ef4444")
    ]

    for ini, fin, color in zonas:

        if fin <= ini:
            continue

        c.create_arc(
            *bbox,
            start=180 - ini * 180,
            extent=-(fin - ini) * 180,
            style="arc",
            width=24,
            outline=color
        )

    # Etiquetas de los extremos
    c.create_text(cx - r, cy + 20, text="0%", fill=MUTED, font=("Segoe UI", 9))
    c.create_text(cx + r, cy + 20, text="100%", fill=MUTED, font=("Segoe UI", 9))

    # Marca del umbral
    ang_u = math.pi * (1 - UMBRAL)
    x1 = cx + (r - 20) * math.cos(ang_u)
    y1 = cy - (r - 20) * math.sin(ang_u)
    x2 = cx + (r + 20) * math.cos(ang_u)
    y2 = cy - (r + 20) * math.sin(ang_u)
    xt = cx + (r + 34) * math.cos(ang_u)
    yt = cy - (r + 34) * math.sin(ang_u)

    c.create_line(x1, y1, x2, y2, fill=TEXT, width=2)
    c.create_text(
        xt, yt,
        text=f"Umbral {UMBRAL * 100:.0f}%",
        fill=MUTED,
        font=("Segoe UI", 9, "bold")
    )

    # Aguja
    angulo = math.pi * (1 - prob)
    largo = r - 22

    x = cx + largo * math.cos(angulo)
    y = cy - largo * math.sin(angulo)

    c.create_line(cx, cy, x, y, fill=NAVY, width=4, capstyle="round")

    c.create_oval(
        cx - 10, cy - 10, cx + 10, cy + 10,
        fill=NAVY, outline="white", width=2
    )

    # Texto central
    color, nivel = color_nivel(prob)

    c.create_text(
        cx, cy + 50,
        text=f"{prob * 100:.1f}%",
        fill=color,
        font=("Segoe UI", 26, "bold")
    )

    c.create_text(
        cx, cy + 82,
        text=f"Probabilidad de morosidad · {nivel}",
        fill=MUTED,
        font=("Segoe UI", 9)
    )


def dibujar_factores(event=None):

    c = factores_canvas
    c.delete("all")

    ancho_c = max(c.winfo_width(), 420)
    nombre = modelo_var.get()

    c.create_text(
        12, 16,
        text="Variables más influyentes del modelo",
        anchor="w",
        fill=TEXT,
        font=("Segoe UI", 11, "bold")
    )

    if nombre not in MODELOS:
        return

    importancias, nota = calcular_importancias(nombre)

    if not importancias:

        c.create_text(
            12, 44,
            text=f"{nombre}",
            anchor="w",
            fill=TEXT,
            font=("Segoe UI", 10, "bold")
        )

        c.create_text(
            12, 70,
            text="Este algoritmo no ofrece una importancia de variables directa.",
            anchor="w",
            fill=MUTED,
            font=("Segoe UI", 10)
        )

        c.create_text(
            12, 92,
            text="Elige Random Forest, Gradient Boosting, Árbol de decisión\n"
                 "o Regresión logística para ver este análisis.",
            anchor="nw",
            fill=MUTED,
            font=("Segoe UI", 9)
        )

        return

    c.create_text(
        12, 38,
        text=nota,
        anchor="w",
        fill=MUTED,
        font=("Segoe UI", 9)
    )

    x_etiqueta = 12
    x_barra = 215
    x_maximo = ancho_c - 75
    maximo = importancias[0][1] or 1.0

    y = 68

    for i, (columna, valor) in enumerate(importancias):

        nombre_var = NOMBRES_AMIGABLES.get(columna, columna)

        c.create_text(
            x_etiqueta, y + 11,
            text=nombre_var,
            anchor="w",
            fill=TEXT,
            font=("Segoe UI", 9)
        )

        largo = max((x_maximo - x_barra) * valor / maximo, 2)

        c.create_rectangle(
            x_barra, y, x_barra + largo, y + 22,
            fill=ACCENT if i < 3 else "#93c5fd",
            outline=""
        )

        c.create_text(
            x_barra + largo + 8, y + 11,
            text=f"{valor * 100:.1f}%",
            anchor="w",
            fill=TEXT,
            font=("Segoe UI", 9, "bold")
        )

        y += 29


factores_canvas.bind("<Configure>", dibujar_factores)


# TABLA COMPARATIVA Y DETALLE DEL MODELO ELEGIDO

def pct(metricas, clave):

    if not metricas or metricas.get(clave) is None:
        return "—"

    try:
        return f"{float(metricas[clave]) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def llenar_tabla():

    for fila in tabla.get_children():
        tabla.delete(fila)

    probs = ULTIMO["probs"]

    for modelo_info in LISTA_MODELOS:

        nombre = modelo_info["nombre"]
        met = modelo_info["metricas"]
        es_mejor = nombre == MEJOR_MODELO_VISIBLE

        if nombre in probs:
            p = probs[nombre]
            es_moroso = p >= UMBRAL
            clase = "⚠  Moroso" if es_moroso else "✓  No moroso"
            prob_txt = f"{p * 100:.1f}%"
            tags = ["moroso" if es_moroso else "nomoroso"]
        else:
            clase = "No disponible"
            prob_txt = "—"
            tags = ["nodisp"]

        if es_mejor:
            tags.append("mejor")

        tabla.insert(
            "", "end",
            iid=nombre,
            values=(
                met.get("ranking", "—") if met else "—",
                ("★ " if es_mejor else "") + nombre,
                clase,
                prob_txt,
                pct(met, "accuracy"),
                pct(met, "precision_moroso"),
                pct(met, "recall_moroso"),
                pct(met, "f1_moroso"),
                pct(met, "roc_auc"),
            ),
            tags=tuple(tags)
        )


def actualizar_resumen():

    probs = ULTIMO["probs"]
    total = len(probs)
    n_morosos = sum(1 for p in probs.values() if p >= UMBRAL)

    if n_morosos == 0:
        color = SUCCESS
    elif n_morosos * 2 > total:
        color = DANGER
    else:
        color = WARNING

    texto = f"{n_morosos} de {total} modelos clasifican al cliente como MOROSO."

    if MEJOR_MODELO_VISIBLE and MEJOR_MODELO_VISIBLE in probs:
        p = probs[MEJOR_MODELO_VISIBLE]
        veredicto = "MOROSO" if p >= UMBRAL else "NO MOROSO"
        texto += (
            f"\nModelo mejor evaluado (★ {MEJOR_MODELO_VISIBLE}): "
            f"{veredicto} · probabilidad {p * 100:.1f}%"
        )

    resumen_label.config(text=texto, fg=color)


def actualizar_detalle(event=None):

    nombre = modelo_var.get()
    probs = ULTIMO["probs"]

    if nombre not in probs:
        return

    probabilidad = probs[nombre]
    porcentaje = probabilidad * 100
    es_moroso = probabilidad >= UMBRAL
    _, nivel = color_nivel(probabilidad)

    metricas = next(
        (m["metricas"] for m in LISTA_MODELOS if m["nombre"] == nombre), None
    )

    extra = ""
    if metricas:
        extra = (
            f"\nDesempeño en prueba:  F1 {pct(metricas, 'f1_moroso')}  ·  "
            f"Recall {pct(metricas, 'recall_moroso')}  ·  "
            f"Precisión {pct(metricas, 'precision_moroso')}"
        )

    if es_moroso:

        resultado_badge.config(text="⚠  MOROSO", bg=DANGER)

        resultado_descripcion.config(
            text=(
                f"{nombre} clasifica al cliente como MOROSO.\n"
                f"Probabilidad estimada: {porcentaje:.2f}%  ·  {nivel}"
                f"{extra}"
            )
        )

    else:

        resultado_badge.config(text="✓  NO MOROSO", bg=SUCCESS)

        resultado_descripcion.config(
            text=(
                f"{nombre} clasifica al cliente como NO MOROSO.\n"
                f"Probabilidad estimada de morosidad: {porcentaje:.2f}%  ·  {nivel}"
                f"{extra}"
            )
        )

    dibujar_gauge(probabilidad)
    dibujar_factores()

    # Resalta en la tabla el modelo mostrado
    if tabla.selection() != (nombre,) and tabla.exists(nombre):
        tabla.selection_set(nombre)


def al_seleccionar_fila(event=None):

    seleccion = tabla.selection()

    if not seleccion:
        return

    nombre = seleccion[0]

    if nombre == modelo_var.get():
        return

    if nombre not in ULTIMO["probs"]:
        return

    modelo_var.set(nombre)
    actualizar_detalle()


selector_modelo.bind("<<ComboboxSelected>>", actualizar_detalle)
tabla.bind("<<TreeviewSelect>>", al_seleccionar_fila)


# OBTENER DATOS

def obtener_datos():

    try:

        datos = {
            "idade": int(idade_var.get()),
            "sexo": VALOR_REAL["sexo"].get(sexo_var.get(), sexo_var.get()),
            "dependentes": int(dependentes_var.get()),
            "escolaridade": VALOR_REAL["escolaridade"].get(
                escolaridade_var.get(), escolaridade_var.get()
            ),
            "estado_civil": VALOR_REAL["estado_civil"].get(
                estado_civil_var.get(), estado_civil_var.get()
            ),
            "salario_anual": VALOR_REAL["salario_anual"].get(
                salario_var.get(), salario_var.get()
            ),
            "tipo_cartao": VALOR_REAL["tipo_cartao"].get(
                tipo_cartao_var.get(), tipo_cartao_var.get()
            ),

            "meses_de_relacionamento": int(meses_relacionamiento_var.get()),
            "qtd_produtos": int(qtd_produtos_var.get()),
            "iteracoes_12m": int(iteracoes_var.get()),
            "meses_inativo_12m": int(meses_inativo_var.get()),

            "limite_credito": float(
                limite_credito_var.get().replace(",", ".")
            ),

            "valor_transacoes_12m": float(
                valor_transacoes_var.get().replace(",", ".")
            ),

            "qtd_transacoes_12m": int(qtd_transacoes_var.get())
        }

        numericos = [
            valor for valor in datos.values()
            if isinstance(valor, (int, float))
        ]

        if any(valor < 0 for valor in numericos):
            raise ValueError("Valores negativos")

        return datos

    except ValueError:

        messagebox.showerror(
            "Datos inválidos",
            "Revisa los campos numéricos.\n\n"
            "Debes ingresar valores válidos y no negativos para edad, "
            "dependientes, meses, productos, crédito y transacciones."
        )

        return None


# REALIZAR PREDICCIÓN (con todos los modelos)

def predecir():

    if not MODELOS:

        messagebox.showerror(
            "Modelos no encontrados",
            "No se pudo cargar ningún modelo desde la carpeta:\n\n"
            "modelos_comparativos\n\n"
            "Ejecuta primero entrenamiento_comparativo.py."
        )

        return

    datos = obtener_datos()

    if datos is None:
        return

    df = pd.DataFrame([datos], columns=COLUMNAS_MODELO)

    probabilidades = {}
    fallos = {}

    for nombre, pipeline in MODELOS.items():

        try:
            probabilidades[nombre] = float(pipeline.predict_proba(df)[0][1])
        except Exception as e:
            fallos[nombre] = str(e)

    if not probabilidades:

        messagebox.showerror(
            "Error en la predicción",
            "Ningún modelo pudo realizar la predicción:\n\n"
            + "\n".join(f"• {n}: {m}" for n, m in fallos.items())
        )

        return

    ULTIMO["probs"] = probabilidades

    # Si el modelo elegido falló, se muestra el primero disponible
    if modelo_var.get() not in probabilidades:
        modelo_var.set(next(iter(probabilidades)))

    if not resultado_card.winfo_ismapped():
        resultado_card.pack(fill="x", pady=(0, 16))

    llenar_tabla()
    actualizar_resumen()
    actualizar_detalle()

    root.update_idletasks()

    # Desplaza la página hasta el inicio de la tarjeta de resultados
    total_alto = max(contenido.winfo_reqheight(), 1)
    lienzo.yview_moveto(min(resultado_card.winfo_y() / total_alto, 1.0))

    if fallos:

        messagebox.showwarning(
            "Algunos modelos fallaron",
            "No se pudo predecir con:\n\n"
            + "\n".join(f"• {n}: {m}" for n, m in fallos.items())
        )


# LIMPIAR FORMULARIO

def limpiar():

    idade_var.set("")
    sexo_var.set(valor_por_defecto("sexo", "M"))
    dependentes_var.set("")

    escolaridade_var.set(valor_por_defecto("escolaridade", "graduacao"))
    estado_civil_var.set(valor_por_defecto("estado_civil", "casado"))
    salario_var.set(valor_por_defecto("salario_anual", "$60K - $80K"))
    tipo_cartao_var.set(valor_por_defecto("tipo_cartao", "blue"))

    meses_relacionamiento_var.set("")
    qtd_produtos_var.set("")
    iteracoes_var.set("")
    meses_inativo_var.set("")

    limite_credito_var.set("")
    valor_transacoes_var.set("")
    qtd_transacoes_var.set("")

    ULTIMO["probs"] = {}
    resultado_card.pack_forget()

    lienzo.yview_moveto(0)


# CARGAR DATOS DE EJEMPLO

def cargar_ejemplo():

    idade_var.set("45")
    sexo_var.set(valor_por_defecto("sexo", "M"))
    dependentes_var.set("3")

    escolaridade_var.set(valor_por_defecto("escolaridade", "graduacao"))
    estado_civil_var.set(valor_por_defecto("estado_civil", "casado"))
    salario_var.set(valor_por_defecto("salario_anual", "$60K - $80K"))
    tipo_cartao_var.set(valor_por_defecto("tipo_cartao", "blue"))

    meses_relacionamiento_var.set("36")
    qtd_produtos_var.set("4")
    iteracoes_var.set("2")
    meses_inativo_var.set("2")

    limite_credito_var.set("8000")
    valor_transacoes_var.set("4000")
    qtd_transacoes_var.set("50")


# BOTONES

ttk.Button(
    botones_frame,
    text="PREDECIR",
    command=predecir,
    style="Primario.TButton",
    cursor="hand2"
).pack(side="left", padx=(0, 10))

ttk.Button(
    botones_frame,
    text="CARGAR EJEMPLO",
    command=cargar_ejemplo,
    style="Verde.TButton",
    cursor="hand2"
).pack(side="left", padx=5)

ttk.Button(
    botones_frame,
    text="LIMPIAR",
    command=limpiar,
    style="Gris.TButton",
    cursor="hand2"
).pack(side="left", padx=5)

root.bind("<Return>", lambda event: predecir())


# VERIFICAR MODELOS (avisos al iniciar)

def avisos_iniciales():

    mensajes = []

    if ERROR_CONFIG:
        mensajes.append(
            "No se encontró config_comparativo.json, por lo que no se mostrarán las "
            "métricas de entrenamiento ni el modelo mejor evaluado."
        )

    if ERRORES_CARGA:
        detalle = "\n".join(
            f"• {nombre}: {motivo[:120]}" for nombre, motivo in ERRORES_CARGA.items()
        )
        mensajes.append(
            "No se pudieron cargar estos modelos:\n" + detalle
        )

    if not MODELOS:
        mensajes.append("Ejecuta primero entrenamiento_comparativo.py.")

    if mensajes:
        messagebox.showwarning("Aviso al iniciar", "\n\n".join(mensajes))


root.after(300, avisos_iniciales)


# INICIAR APLICACIÓN

root.mainloop()