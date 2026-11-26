# Situacion problema
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL, NAR = "#257de0", "#f47230"
TXT, TXT2, GRID = "#222222", "#666666", "#e3e3e3"
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.edgecolor": TXT2,
                     "axes.labelcolor": TXT, "xtick.color": TXT2, "ytick.color": TXT2})


# O(filas)
def leer(n):
    return list(csv.DictReader(open("res/%s.csv" % n)))


# O(1)
def base(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


esc = leer("exp_escala")

fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
base(ax)
for mod, col, et in (("suave", AZUL, "Sombra suave"), ("aleatoria", NAR, "Sombra aleatoria")):
    f = [r for r in esc if r["modelo"] == mod]
    x = [float(r["lmax_prom"]) * float(r["E"]) / 1e6 for r in f]
    y = [float(r["ms_tabla_prom"]) for r in f]
    ax.plot(x, y, color=col, linewidth=2, marker="o", markersize=5, label=et,
            markeredgecolor="white", markeredgewidth=1)
ax.set_xlabel("Lmax promedio × E (millones de relajaciones)")
ax.set_ylabel("Tiempo promedio de la tabla (ms)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("res/g1_tabla_lineal.png")

fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
base(ax)
f = [r for r in esc if r["modelo"] == "aleatoria"]
v = [int(r["V"]) for r in f]
ax.plot(v, [float(r["nodos_exhaustivo_prom"]) for r in f], color=NAR, linewidth=2, marker="o", markersize=5,
        markeredgecolor="white", label="Busqueda exhaustiva simple")
ax.plot(v, [float(r["nodos_acotado_prom"]) for r in f], color=AZUL, linewidth=2, marker="o", markersize=5,
        markeredgecolor="white", label="Respaldo acotado con la tabla G")
ax.axhline(1e7, color=TXT2, linewidth=1, linestyle="--")
ax.text(v[0], 1.2e7, "limite de 10 millones de estados", color=TXT2, fontsize=8)
ax.set_yscale("log")
ax.set_xlabel("Puntos de la cuadricula (V)")
ax.set_ylabel("Estados visitados por consulta")
ax.legend(frameon=False, loc="center right")
fig.tight_layout()
fig.savefig("res/g2_respaldo.png")

hora = leer("exp_hora")
fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
base(ax)
hs = [int(r["hora"]) for r in hora]
ax.bar(hs, [float(r["ahorro_pct_prom"]) for r in hora], color=AZUL, width=0.7)
ax.set_xticks(hs)
ax.set_xticklabels(["%d h" % h for h in hs], fontsize=8)
ax.set_ylabel("Sol evitado frente a la ruta corta (%)")
ax.set_xlabel("Hora de salida (alfa 1.3, 400 pares del campus)")
fig.tight_layout()
fig.savefig("res/g3_hora.png")

alf = leer("exp_alfa")
fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=200)
base(ax)
a = [float(r["alfa"]) for r in alf]
y = [float(r["ahorro_pct_prom"]) for r in alf]
ax.plot(a, y, color=AZUL, linewidth=2, marker="o", markersize=5, markeredgecolor="white")
for xi, yi, r in zip(a, y, alf):
    ax.annotate("+%.0f m" % float(r["extra_m_prom"]), (xi, yi), textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=7.5, color=TXT2)
ax.set_xlabel("Tolerancia alfa")
ax.set_ylabel("Sol evitado promedio (%)")
fig.tight_layout()
fig.savefig("res/g4_alfa.png")
print("ok")