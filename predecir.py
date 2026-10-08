import os
import math
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import joblib
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


MODELO_PATH = resource_path("modelo_morosidad.pkl")

UMBRAL = 0.50

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


# ============================================================
# CARGAR MODELO
# ============================================================

modelo = None

try:
    modelo = joblib.load(MODELO_PATH)
except Exception as e:
    messagebox.showerror(
        "Error al cargar el modelo",
        f"No se pudo cargar el modelo:\n\n{e}"
    )
    modelo = None


def calcular_importancias():
    """Importancia global por variable original (suma de las columnas one-hot)."""
    if modelo is None:
        return []

    try:
        preprocesador = modelo.named_steps["preprocesamiento"]
        bosque = modelo.named_steps["modelo"]

        nombres = preprocesador.get_feature_names_out()
        importancias = bosque.feature_importances_

        acumulado = {}

        for nombre, valor in zip(nombres, importancias):
            limpio = nombre.split("__", 1)[-1]

            candidatas = [
                c for c in COLUMNAS_MODELO
                if limpio == c or limpio.startswith(c + "_")
            ]

            columna = max(candidatas, key=len) if candidatas else limpio
            acumulado[columna] = acumulado.get(columna, 0) + valor

        ordenado = sorted(
            acumulado.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return ordenado[:8]

    except Exception as e:
        print("No se pudo calcular la importancia de variables:", e)
        return []


IMPORTANCIAS = calcular_importancias()


# ============================================================
# VENTANA PRINCIPAL
# ============================================================

root = tk.Tk()
root.title("Sistema de Predicción de Morosidad")
root.configure(bg=BG)

pantalla_w = root.winfo_screenwidth()
pantalla_h = root.winfo_screenheight()

ancho = min(1120, pantalla_w - 60)
alto = min(780, pantalla_h - 120)
pos_x = (pantalla_w - ancho) // 2
pos_y = max((pantalla_h - alto) // 2 - 20, 0)

root.geometry(f"{ancho}x{alto}+{pos_x}+{pos_y}")
root.minsize(760, 520)


# ============================================================
# ESTILOS
# ============================================================

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


# ============================================================
# VARIABLES
# ============================================================

idade_var = tk.StringVar()
sexo_var = tk.StringVar(value="M")
dependentes_var = tk.StringVar()

escolaridade_var = tk.StringVar(value="Graduate")
estado_civil_var = tk.StringVar(value="Married")
salario_var = tk.StringVar(value="$60K - $80K")
tipo_cartao_var = tk.StringVar(value="Blue")

meses_relacionamiento_var = tk.StringVar()
qtd_produtos_var = tk.StringVar()
iteracoes_var = tk.StringVar()
meses_inativo_var = tk.StringVar()

limite_credito_var = tk.StringVar()
valor_transacoes_var = tk.StringVar()
qtd_transacoes_var = tk.StringVar()


# ============================================================
# ENCABEZADO
# ============================================================

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
    text="Evaluación automática del riesgo de incumplimiento mediante Machine Learning",
    bg=NAVY,
    fg="#bfdbfe",
    font=("Segoe UI", 10)
).pack(anchor="w", pady=(3, 0))

estado_modelo = tk.Label(
    header,
    text="●  Modelo cargado" if modelo is not None else "●  Modelo no encontrado",
    bg=NAVY,
    fg="#4ade80" if modelo is not None else "#fca5a5",
    font=("Segoe UI", 10, "bold")
)
estado_modelo.pack(side="right", padx=30)


# ============================================================
# BARRA INFERIOR FIJA (botones + información del modelo)
# ============================================================

barra_inferior = tk.Frame(root, bg="white")
barra_inferior.pack(side="bottom", fill="x")

tk.Frame(barra_inferior, bg=BORDER, height=1).pack(side="top", fill="x")

botones_frame = tk.Frame(barra_inferior, bg="white")
botones_frame.pack(side="left", padx=25, pady=12)

footer = tk.Label(
    barra_inferior,
    text="Modelo: Random Forest   |   Umbral: 50%   |   Accuracy: 94.08%",
    bg="white",
    fg=MUTED,
    font=("Segoe UI", 9)
)
footer.pack(side="right", padx=25)


# ============================================================
# ÁREA CON SCROLL
# ============================================================

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


# ============================================================
# COMPONENTES REUTILIZABLES
# ============================================================

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


# ============================================================
# TARJETA: INFORMACIÓN PERSONAL
# ============================================================

tarjeta_personal, personal_body = crear_tarjeta(
    cuerpo_pagina,
    "Información personal",
    "Datos demográficos del cliente",
    columnas=3
)

crear_campo(personal_body, 0, 0, "Edad", idade_var)
crear_campo(personal_body, 0, 1, "Sexo", sexo_var, ["M", "F"])
crear_campo(personal_body, 0, 2, "Dependientes", dependentes_var)

crear_campo(
    personal_body, 1, 0, "Escolaridad", escolaridade_var,
    [
        "Graduate", "High School", "College",
        "Post-Graduate", "Doctorate", "Uneducated", "Unknown"
    ]
)

crear_campo(
    personal_body, 1, 1, "Estado civil", estado_civil_var,
    ["Married", "Single", "Divorced", "Unknown"]
)

crear_campo(
    personal_body, 1, 2, "Salario anual", salario_var,
    [
        "menos que $40K", "$40K - $60K", "$60K - $80K",
        "$80K - $120K", "$120K +", "na"
    ]
)

crear_campo(
    personal_body, 2, 0, "Tipo de tarjeta", tipo_cartao_var,
    ["Blue", "Silver", "Gold", "Platinum"]
)


# ============================================================
# TARJETA: COMPORTAMIENTO DEL CLIENTE
# ============================================================

tarjeta_comportamiento, comportamiento_body = crear_tarjeta(
    cuerpo_pagina,
    "Comportamiento del cliente",
    "Relación con la entidad y actividad reciente",
    columnas=2
)

crear_campo(comportamiento_body, 0, 0, "Meses de relación", meses_relacionamiento_var)
crear_campo(comportamiento_body, 0, 1, "Cantidad de productos", qtd_produtos_var)
crear_campo(comportamiento_body, 1, 0, "Interacciones últimos 12 meses", iteracoes_var)
crear_campo(comportamiento_body, 1, 1, "Meses inactivo últimos 12 meses", meses_inativo_var)


# ============================================================
# TARJETA: INFORMACIÓN FINANCIERA
# ============================================================

tarjeta_financiera, financiero_body = crear_tarjeta(
    cuerpo_pagina,
    "Información financiera",
    "Línea de crédito y movimientos del último año",
    columnas=3
)

crear_campo(financiero_body, 0, 0, "Límite de crédito", limite_credito_var)
crear_campo(financiero_body, 0, 1, "Valor de transacciones (12 meses)", valor_transacoes_var)
crear_campo(financiero_body, 0, 2, "Cantidad de transacciones (12 meses)", qtd_transacoes_var)


# ============================================================
# TARJETA: RESULTADO (oculta hasta predecir)
# ============================================================

resultado_card, resultado_body = crear_tarjeta(
    cuerpo_pagina,
    "Resultado de la evaluación",
    "Clasificación del modelo y análisis visual",
    columnas=1,
    mostrar=False
)

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


# ============================================================
# GRÁFICOS
# ============================================================

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
        (0.00, 0.30, "#22c55e"),
        (0.30, UMBRAL, "#f59e0b"),
        (UMBRAL, 1.00, "#ef4444")
    ]

    for ini, fin, color in zonas:

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

    # Marca del umbral (50%)
    c.create_line(cx, cy - r - 20, cx, cy - r + 20, fill=TEXT, width=2)
    c.create_text(
        cx, cy - r - 30,
        text="Umbral 50%",
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

    c.create_text(
        12, 16,
        text="Variables más influyentes del modelo",
        anchor="w",
        fill=TEXT,
        font=("Segoe UI", 11, "bold")
    )

    c.create_text(
        12, 38,
        text="Importancia global del Random Forest (no específica de este cliente)",
        anchor="w",
        fill=MUTED,
        font=("Segoe UI", 9)
    )

    if not IMPORTANCIAS:

        c.create_text(
            12, 80,
            text="Importancia de variables no disponible.",
            anchor="w",
            fill=MUTED,
            font=("Segoe UI", 10)
        )

        return

    x_etiqueta = 12
    x_barra = 215
    x_maximo = ancho_c - 75
    maximo = IMPORTANCIAS[0][1]

    y = 68

    for i, (columna, valor) in enumerate(IMPORTANCIAS):

        nombre = NOMBRES_AMIGABLES.get(columna, columna)

        c.create_text(
            x_etiqueta, y + 11,
            text=nombre,
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


# ============================================================
# OBTENER DATOS
# ============================================================

def obtener_datos():

    try:

        datos = {
            "idade": int(idade_var.get()),
            "sexo": sexo_var.get(),
            "dependentes": int(dependentes_var.get()),
            "escolaridade": escolaridade_var.get(),
            "estado_civil": estado_civil_var.get(),
            "salario_anual": salario_var.get(),
            "tipo_cartao": tipo_cartao_var.get(),

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


# ============================================================
# REALIZAR PREDICCIÓN
# ============================================================

def predecir():

    if modelo is None:

        messagebox.showerror(
            "Modelo no encontrado",
            "No se encontró el archivo:\n\n"
            "modelo_morosidad.pkl\n\n"
            "Ejecuta primero entrenamiento.py."
        )

        return

    datos = obtener_datos()

    if datos is None:
        return

    try:

        df = pd.DataFrame([datos])

        probabilidad = modelo.predict_proba(df)[0][1]

        porcentaje = probabilidad * 100

        es_moroso = probabilidad >= UMBRAL

        _, nivel = color_nivel(probabilidad)

        if not resultado_card.winfo_ismapped():
            resultado_card.pack(fill="x", pady=(0, 16))

        if es_moroso:

            resultado_badge.config(text="⚠  MOROSO", bg=DANGER)

            resultado_descripcion.config(
                text=(
                    "El modelo clasifica al cliente como MOROSO.\n"
                    f"Probabilidad estimada: {porcentaje:.2f}%  ·  {nivel}"
                )
            )

        else:

            resultado_badge.config(text="✓  NO MOROSO", bg=SUCCESS)

            resultado_descripcion.config(
                text=(
                    "El modelo clasifica al cliente como NO MOROSO.\n"
                    f"Probabilidad estimada de morosidad: {porcentaje:.2f}%  ·  {nivel}"
                )
            )

        dibujar_gauge(probabilidad)
        dibujar_factores()

        root.update_idletasks()
        lienzo.yview_moveto(1.0)

    except Exception as e:

        messagebox.showerror(
            "Error en la predicción",
            f"Ocurrió un error al realizar la predicción:\n\n{e}"
        )


# ============================================================
# LIMPIAR FORMULARIO
# ============================================================

def limpiar():

    idade_var.set("")
    sexo_var.set("M")
    dependentes_var.set("")

    escolaridade_var.set("Graduate")
    estado_civil_var.set("Married")
    salario_var.set("$60K - $80K")
    tipo_cartao_var.set("Blue")

    meses_relacionamiento_var.set("")
    qtd_produtos_var.set("")
    iteracoes_var.set("")
    meses_inativo_var.set("")

    limite_credito_var.set("")
    valor_transacoes_var.set("")
    qtd_transacoes_var.set("")

    resultado_card.pack_forget()

    lienzo.yview_moveto(0)


# ============================================================
# CARGAR DATOS DE EJEMPLO
# ============================================================

def cargar_ejemplo():

    idade_var.set("45")
    sexo_var.set("M")
    dependentes_var.set("3")

    escolaridade_var.set("Graduate")
    estado_civil_var.set("Married")
    salario_var.set("$60K - $80K")
    tipo_cartao_var.set("Blue")

    meses_relacionamiento_var.set("36")
    qtd_produtos_var.set("4")
    iteracoes_var.set("2")
    meses_inativo_var.set("2")

    limite_credito_var.set("8000")
    valor_transacoes_var.set("4000")
    qtd_transacoes_var.set("50")


# ============================================================
# BOTONES
# ============================================================

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


# ============================================================
# VERIFICAR MODELO
# ============================================================

if modelo is None:

    root.after(
        300,
        lambda: messagebox.showwarning(
            "Modelo no encontrado",
            "No se encontró modelo_morosidad.pkl.\n\n"
            "Ejecuta primero entrenamiento.py."
        )
    )


# ============================================================
# INICIAR APLICACIÓN
# ============================================================

root.mainloop()