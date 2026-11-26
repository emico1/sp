# Situacion problema
import heapq
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict

import numpy as np
import tifffile
from matplotlib.path import Path
from scipy.spatial import ConvexHull

LAT = 25.651
LON = -100.290
DIA = 227
UTC = -6
RADIO = 6378137.0
ALTURA_PISO = 3.5
PISOS_DEFECTO = 2
ALTURA_MIN_ARBOL = 3.0
PASO = 2.0
MARGEN = 150.0
UNION = 10.0
UNION_CAMPUS = 40.0
EXPLANADA_MAX = 60.0
MALLA = 30.0
ATAJO = 1.4
ZONA = (-100.294878, -100.282153, 25.646540, 25.655663)
FRANJAS = [7 * 3600 + k * 1800 for k in range(24)]
CAMINABLE = {"footway", "steps", "pedestrian", "service", "path", "corridor", "residential", "unclassified",
             "living_street", "tertiary", "tertiary_link", "secondary", "secondary_link", "track", "cycleway"}
CALLES_PROLONGAR = {"residential", "unclassified", "living_street", "tertiary", "tertiary_link", "secondary"}
CALLES_EXPLANADA = CALLES_PROLONGAR | {"secondary_link"}
NUCLEO = {"Arena Borregos", "Aulas I", "Aulas II", "Aulas III", "Aulas VI", "Aulas VII", "BiblioTEC", "CEDES", "Centrales",
          "Centro Internacional de Aprendizaje Avanzado (CIAP)", "Centro de Biotecnología FEMSA", "Centro de Congresos",
          "Centro de Manufactura", "Centro de Tecnología Avanzada para la Producción (CETEC)", "Domo Cultural",
          "Edificio de Estudios Creativos", "Edificio de Negocios", "Espacio de Reflexión (Punto Blanco)",
          "Expedition FEMSA", "Hub de Innovación y Emprendimiento Eduardo Garza T.", "Núcleo", "Oxxo - Aulas IV",
          "Oxxo Grab & Go", "Pabellón La Carreta", "Pabellón TEC", "Rectoría del Tecnológico de Monterrey",
          "Residencias I", "Residencias II", "Residencias III", "Residencias V", "Wellness Center",
          "Estacionamiento E1", "Casa Naranjos", "Starbucks"}
CORTOS = {
    "Rectoría del Tecnológico de Monterrey": "Rectoria",
    "Centro de Tecnología Avanzada para la Producción (CETEC)": "CETEC",
    "Centro Internacional de Aprendizaje Avanzado (CIAP)": "CIAP",
    "BiblioTEC": "Biblioteca",
    "Centro Cultural del Tecnológico de Monterrey": "CentroCultural",
    "Centro de Biotecnología FEMSA": "Biotecnologia",
    "Hub de Innovación y Emprendimiento Eduardo Garza T.": "HubInnovacion",
    "Edificio de Estudios Creativos": "EstudiosCreativos",
    "Edificio de Negocios": "EdificioNegocios",
    "Espacio de Reflexión (Punto Blanco)": "PuntoBlanco",
    "Auditorio Luis Elizondo (ALE)": "AuditorioLuisElizondo",
}
ROMANOS = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"}

KX = 111320.0 * math.cos(math.radians(LAT))
KY = 110540.0


# O(1)
def xy(lon, lat):
    return ((lon - LON) * KX, (lat - LAT) * KY)


# O(1)
def inverso(x, y):
    return (round(LON + x / KX, 8), round(LAT + y / KY, 8))


# O(1)
# Altura y direccion del sol con las ecuaciones de la NOAA
def posicion_sol(seg):
    g = 2 * math.pi / 365 * (DIA - 1 + (seg / 3600 - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    dec = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
           + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    tst = seg / 60 + eqt + 4 * LON - 60 * UTC
    ha = math.radians(tst / 4 - 180)
    phi = math.radians(LAT)
    cz = max(-1.0, min(1.0, math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(ha)))
    zen = math.acos(cz)
    elev = 90 - math.degrees(zen)
    ca = max(-1.0, min(1.0, (math.sin(phi) * cz - math.sin(dec)) / (math.cos(phi) * math.sin(zen))))
    az = math.degrees(math.acos(ca))
    az = (az + 180) % 360 if ha > 0 else (540 - az) % 360
    return elev, az


# O(largo del nombre)
def slug(nombre):
    if nombre in CORTOS:
        return CORTOS[nombre]
    s = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    s = re.sub(r"\(.*?\)", "", s)
    partes = [w if w in ROMANOS else w.capitalize() for w in re.split(r"[^A-Za-z0-9]+", s) if w]
    return "".join(partes)[:24]


# O(k)
def recortar(coords):
    partes, actual = [], []
    for c in coords:
        if ZONA[0] <= c[0] <= ZONA[1] and ZONA[2] <= c[1] <= ZONA[3]:
            actual.append(tuple(c))
        else:
            if len(actual) > 1:
                partes.append(actual)
            actual = []
    if len(actual) > 1:
        partes.append(actual)
    return partes


# O(S)
# Indica si el segmento p1 p2 cruza alguno de los segmentos de M
def cruza(p1, p2, M):
    rx, ry = p2[0] - p1[0], p2[1] - p1[1]
    sx, sy = M[:, 2] - M[:, 0], M[:, 3] - M[:, 1]
    den = rx * sy - ry * sx
    qx, qy = M[:, 0] - p1[0], M[:, 1] - p1[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qx * sy - qy * sx) / den
        u = (qx * ry - qy * rx) / den
    return bool(np.any((np.abs(den) > 1e-9) & (t > 0.02) & (t < 0.98) & (u > 0.0) & (u < 1.0)))


# O(S)
def segmentos_calles(lineas, tipos):
    return np.array([list(xy(*a)) + list(xy(*b)) for coords, _, tipo in lineas if tipo in tipos
                     for a, b in zip(coords, coords[1:])])


# O(1)
def proyeccion(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return t, math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy), (a[0] + t * dx, a[1] + t * dy)


# O(F)
def leer(archivo):
    lineas, elevada, privadas, edificios = [], [], [], []
    for f in json.load(open(archivo))["features"]:
        p, g = f["properties"], f["geometry"]
        elev = any(k in p for k in ("bridge", "tunnel", "layer"))
        if p.get("highway") in CAMINABLE and g["type"] == "LineString" and p.get("access") in ("private", "no"):
            privadas.append(([tuple(c) for c in g["coordinates"]], p["highway"], elev))
        elif p.get("highway") in CAMINABLE and g["type"] == "LineString":
            techado = p.get("covered") == "yes" or p.get("tunnel") == "yes" or p.get("indoor") == "yes"
            for parte in recortar(g["coordinates"]):
                lineas.append((parte, techado, p["highway"]))
                elevada.append(elev)
        elif "building" in p and g["type"] == "Polygon":
            h = None
            try:
                if p.get("height") and float(p["height"]) > 0:
                    h = float(p["height"])
            except ValueError:
                pass
            if h is None:
                try:
                    h = float(p["building:levels"]) * ALTURA_PISO
                except (KeyError, ValueError):
                    h = PISOS_DEFECTO * ALTURA_PISO
            edificios.append(([xy(*c) for c in g["coordinates"][0]], h, p.get("name"), "building:levels" in p or "height" in p))
    return lineas, elevada, privadas, edificios


# O(N * S)
# Pega cada extremo suelto a la linea mas cercana si esta a menos de UNION metros
def pegar_extremos(lineas, conteo):
    pegados = 0
    for li, (coords, _, _) in enumerate(lineas):
        for extremo in (0, len(coords) - 1):
            c = coords[extremo]
            if conteo[c] > 1:
                continue
            p = xy(*c)
            mejor_d, mejor_p = UNION, None
            for lj, (otro, _, _) in enumerate(lineas):
                for k in range(len(otro) - 1):
                    if lj == li and (k == extremo or k + 1 == extremo or k == extremo - 1):
                        continue
                    t, d, _ = proyeccion(p, xy(*otro[k]), xy(*otro[k + 1]))
                    if d < mejor_d:
                        mejor_d, mejor_p = d, (lj, k, t)
            if mejor_p is None:
                continue
            lj, k, t = mejor_p
            otro = lineas[lj][0]
            if t < 0.05 or t * math.dist(xy(*otro[k]), xy(*otro[k + 1])) < 0.5:
                nuevo = otro[k]
            elif t > 0.95 or (1 - t) * math.dist(xy(*otro[k]), xy(*otro[k + 1])) < 0.5:
                nuevo = otro[k + 1]
            else:
                a, b = otro[k], otro[k + 1]
                nuevo = (round(a[0] + (b[0] - a[0]) * t, 8), round(a[1] + (b[1] - a[1]) * t, 8))
                otro.insert(k + 1, nuevo)
                conteo[nuevo] += 1
            conteo[c] -= 1
            coords[extremo] = nuevo
            conteo[nuevo] += 1
            pegados += 1
    return pegados


# O(N * S)
# Prolonga dentro del campus los andadores cortados sin cruzar edificios ni avenidas
def prolongar(lineas, conteo, casco, bordes, calles, edif_campus):
    prolongados = 0
    for li, (coords, _, _) in enumerate(lineas):
        for extremo in (0, len(coords) - 1):
            c = coords[extremo]
            if conteo[c] > 1 or not casco.contains_point(xy(*c)):
                continue
            p = xy(*c)
            opciones = []
            for lj, (otro, _, _) in enumerate(lineas):
                if lj == li:
                    continue
                for k in range(len(otro) - 1):
                    t, d, q = proyeccion(p, xy(*otro[k]), xy(*otro[k + 1]))
                    if UNION <= d < UNION_CAMPUS:
                        opciones.append((d, lj, k, t, q))
            opciones.sort()
            for d, lj, k, t, q in opciones:
                medio = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                if cruza(p, q, bordes) or cruza(p, q, calles) or any(cp.contains_point(medio) for cp in edif_campus):
                    continue
                otro = lineas[lj][0]
                if t < 0.02:
                    nuevo = otro[k]
                elif t > 0.98:
                    nuevo = otro[k + 1]
                else:
                    nuevo = inverso(*q)
                    otro.insert(k + 1, nuevo)
                if extremo == 0:
                    coords.insert(0, nuevo)
                else:
                    coords.append(nuevo)
                conteo[c] += 1
                conteo[nuevo] += 2
                prolongados += 1
                break
    return prolongados


# O(S^2)
# Agrega un punto donde dos lineas al mismo nivel se cruzan sin compartir nodo
def partir_cruces(lineas, elevada):
    segs = []
    for li, (coords, _, _) in enumerate(lineas):
        if elevada[li]:
            continue
        for k in range(len(coords) - 1):
            segs.append((li, k) + xy(*coords[k]) + xy(*coords[k + 1]))
    S = np.array([s[2:] for s in segs])
    ax, ay, bx, by = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    rx, ry = bx - ax, by - ay
    cortes = defaultdict(list)
    for i in range(len(segs)):
        qx, qy = ax[i + 1:] - ax[i], ay[i + 1:] - ay[i]
        den = rx[i] * ry[i + 1:] - ry[i] * rx[i + 1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qx * ry[i + 1:] - qy * rx[i + 1:]) / den
            u = (qx * ry[i] - qy * rx[i]) / den
        ok = (np.abs(den) > 1e-9) & (t > 0.001) & (t < 0.999) & (u > 0.001) & (u < 0.999)
        for j in np.nonzero(ok)[0]:
            jj = i + 1 + j
            if segs[i][0] == segs[jj][0] and abs(segs[i][1] - segs[jj][1]) <= 1:
                continue
            li, k = segs[i][:2]
            a, b = lineas[li][0][k], lineas[li][0][k + 1]
            tt = float(t[j])
            punto = (round(a[0] + (b[0] - a[0]) * tt, 8), round(a[1] + (b[1] - a[1]) * tt, 8))
            cortes[(li, k)].append((tt, punto))
            cortes[segs[jj][:2]].append((float(u[j]), punto))
    for (li, k) in sorted(cortes, key=lambda x: (x[0], -x[1])):
        lineas[li][0][k + 1:k + 1] = [p for _, p in sorted(cortes[(li, k)])]
    return sum(len(v) for v in cortes.values()) // 2


# O(N)
def partir_en_tramos(lineas):
    grado = defaultdict(int)
    for coords, _, _ in lineas:
        for i, c in enumerate(coords):
            grado[c] += 1 if i in (0, len(coords) - 1) else 2
    tramos = []
    for coords, techado, tipo in lineas:
        actual = [coords[0]]
        for c in coords[1:]:
            actual.append(c)
            if grado[c] != 2 or c == coords[-1]:
                if actual[0] != actual[-1]:
                    tramos.append((actual, techado, tipo))
                actual = [c]
    return tramos


# O(E log V)
def distancia_red(red, a, b, tope):
    vistos = {a: 0.0}
    cola = [(0.0, a)]
    while cola:
        d, u = heapq.heappop(cola)
        if u == b:
            return d
        if d > vistos.get(u, 1e18) or d > tope:
            continue
        for w, l in red[u].items():
            nd = d + l
            if nd <= tope and nd < vistos.get(w, 1e18):
                vistos[w] = nd
                heapq.heappush(cola, (nd, w))
    return 1e18


# O(C^2 * (S + E log V))
# Agrega entradas, puntos de malla y tramos de explanada en los espacios abiertos del campus
def agregar_explanadas(tramos, lineas, edificios, casco, vert):
    edif = [(Path(e[0]), e) for e in edificios if casco.contains_point(np.mean(e[0], axis=0)) or e[2] in NUCLEO]
    bordes = np.array([[a[0], a[1], b[0], b[1]] for _, e in edif for a, b in zip(e[0], e[0][1:])])
    calles = segmentos_calles(lineas, CALLES_EXPLANADA)
    cA, cB = calles[:, :2], calles[:, 2:]
    puntos = set()
    for coords, _, _ in tramos:
        puntos.add(coords[0])
        puntos.add(coords[-1])
    entradas = 0
    for camino, e in edif:
        if e[2] not in NUCLEO:
            continue
        anillo = np.array(e[0])
        cerca = [c for c in puntos if camino.contains_point(xy(*c), radius=-40) or
                 np.min(np.hypot(anillo[:, 0] - xy(*c)[0], anillo[:, 1] - xy(*c)[1])) < 15]
        if cerca:
            continue
        cx, cy = anillo[:-1].mean(axis=0)
        for k in range(len(anillo) - 1):
            a, b = anillo[k], anillo[k + 1]
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            if math.dist(a, b) < 12:
                continue
            dx, dy = mx - cx, my - cy
            n = math.hypot(dx, dy)
            p = (mx + 2.0 * dx / n, my + 2.0 * dy / n)
            if any(cp.contains_point(p) for cp, _ in edif):
                continue
            puntos.add(inverso(*p))
            entradas += 1
    candidatos = [c for c in puntos if casco.contains_point(xy(*c))]
    gx0, gy0 = vert.min(axis=0)
    gx1, gy1 = vert.max(axis=0)
    malla = 0
    for gx in np.arange(gx0, gx1, MALLA):
        for gy in np.arange(gy0, gy1, MALLA):
            p = (float(gx), float(gy))
            if not casco.contains_point(p) or any(cp.contains_point(p, radius=-3) or cp.contains_point(p, radius=3) for cp, _ in edif):
                continue
            dx, dy = cB[:, 0] - cA[:, 0], cB[:, 1] - cA[:, 1]
            L2 = dx * dx + dy * dy
            tt = np.clip(((p[0] - cA[:, 0]) * dx + (p[1] - cA[:, 1]) * dy) / np.where(L2 == 0, 1, L2), 0, 1)
            if np.min(np.hypot(p[0] - cA[:, 0] - tt * dx, p[1] - cA[:, 1] - tt * dy)) < 4:
                continue
            candidatos.append(inverso(*p))
            malla += 1
    red = defaultdict(dict)
    for coords, _, _ in tramos:
        L = sum(math.dist(xy(*a), xy(*b)) for a, b in zip(coords, coords[1:]))
        u, w = coords[0], coords[-1]
        red[u][w] = min(red[u].get(w, 1e18), L)
        red[w][u] = min(red[w].get(u, 1e18), L)
    cxy = {c: xy(*c) for c in candidatos}
    pares = []
    for i, a in enumerate(candidatos):
        for b in candidatos[i + 1:]:
            d = math.dist(cxy[a], cxy[b])
            if 3.0 < d <= EXPLANADA_MAX:
                pares.append((d, a, b))
    pares.sort()
    explanadas = 0
    for d, a, b in pares:
        if distancia_red(red, a, b, ATAJO * d) <= ATAJO * d:
            continue
        p1, p2 = cxy[a], cxy[b]
        medio = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        if not casco.contains_point(medio):
            continue
        if cruza(p1, p2, bordes) or any(cp.contains_point(medio) for cp, _ in edif) or cruza(p1, p2, calles):
            continue
        tramos.append(([a, b], False, "explanada"))
        red[a][b] = d
        red[b][a] = d
        explanadas += 1
    return entradas, malla, explanadas


# O(V + E)
def componente_mayor(tramos):
    ady = defaultdict(set)
    for t in tramos:
        ady[t[0][0]].add(t[0][-1])
        ady[t[0][-1]].add(t[0][0])
    visto, mejor = set(), []
    for s in ady:
        if s in visto:
            continue
        comp, pila = [], [s]
        visto.add(s)
        while pila:
            u = pila.pop()
            comp.append(u)
            for w in ady[u]:
                if w not in visto:
                    visto.add(w)
                    pila.append(w)
        if len(comp) > len(mejor):
            mejor = comp
    return set(mejor)


# O(B * V)
def nombrar(nodos, pos, edificios):
    nombre = {c: "P%03d" % i for i, c in enumerate(nodos)}
    usados = set()
    for anillo, _, nom, _ in edificios:
        if not nom:
            continue
        cx = sum(p[0] for p in anillo) / len(anillo)
        cy = sum(p[1] for p in anillo) / len(anillo)
        c = min(nodos, key=lambda n: (pos[n][0] - cx) ** 2 + (pos[n][1] - cy) ** 2)
        s = slug(nom)
        if math.hypot(pos[c][0] - cx, pos[c][1] - cy) < 120 and c not in usados and s and s not in nombre.values():
            nombre[c] = s
            usados.add(c)
    return nombre


# O(M * B)
def dentro_de_edificio(px, py, edificios):
    res = np.zeros(len(px), dtype=bool)
    for anillo, _, _, _ in edificios:
        a = np.array(anillo)
        xa, ya = a[:-1, 0], a[:-1, 1]
        xb, yb = a[1:, 0], a[1:, 1]
        corta = ((ya[None, :] > py[:, None]) != (yb[None, :] > py[:, None]))
        xi = xa[None, :] + (py[:, None] - ya[None, :]) * (xb - xa)[None, :] / np.where(yb - ya == 0, 1e-12, yb - ya)[None, :]
        res ^= (np.sum(corta & (px[:, None] < xi), axis=1) % 2 == 1)
    return res


# O(pixeles * B)
# Lee la altura de arboles y descarta los pixeles que caen dentro de edificios
def leer_arboles(archivo, edificios):
    with tifffile.TiffFile(archivo) as tf:
        pg = tf.pages[0]
        arbol = pg.asarray().astype(float)
        escala = pg.tags["ModelPixelScaleTag"].value[0]
        X0, Y0 = pg.tags["ModelTiepointTag"].value[3:5]
    filas, cols = arbol.shape
    cc, rr = np.meshgrid(np.arange(cols) + 0.5, np.arange(filas) + 0.5)
    plon = np.degrees((X0 + cc * escala) / RADIO)
    plat = np.degrees(2 * np.arctan(np.exp((Y0 - rr * escala) / RADIO)) - math.pi / 2)
    puntos = np.column_stack([((plon - LON) * KX).ravel(), ((plat - LAT) * KY).ravel()])
    en_edificio = np.zeros(len(puntos), dtype=bool)
    for anillo, _, _, _ in edificios:
        en_edificio |= Path(anillo).contains_points(puntos)
    arbol = arbol * (arbol >= ALTURA_MIN_ARBOL) * (~en_edificio.reshape(arbol.shape))
    return arbol, escala, X0, Y0


# O(M)
def altura_arbol(x, y, arboles):
    arbol, escala, X0, Y0 = arboles
    mx = RADIO * np.radians(LON + x / KX)
    my = RADIO * np.log(np.tan(math.pi / 4 + np.radians(LAT + y / KY) / 2))
    c = ((mx - X0) / escala).astype(int)
    r = ((Y0 - my) / escala).astype(int)
    ok = (c >= 0) & (c < arbol.shape[1]) & (r >= 0) & (r < arbol.shape[0])
    h = np.zeros(len(x))
    h[ok] = arbol[r[ok], c[ok]]
    return h


# O(H * M * (B + alcance))
# Fraccion de cada tramo a la sombra en cada franja, proyectando edificios y arboles hacia el sol
def calcular_sombra(tramos, edificios, arboles):
    muestras, dueno, largos = [], [], []
    for i, (coords, _, _) in enumerate(tramos):
        pts = [xy(*c) for c in coords]
        largo = 0.0
        for a, b in zip(pts, pts[1:]):
            seg = math.dist(a, b)
            n = max(1, int(round(seg / PASO)))
            for k in range(n):
                t = (k + 0.5) / n
                muestras.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, seg / n))
                dueno.append(i)
            largo += seg
        largos.append(largo)
    M = np.array(muestras)
    dueno = np.array(dueno)
    A = np.array([a for anillo, _, _, _ in edificios for a in anillo[:-1]])
    B = np.array([b for anillo, _, _, _ in edificios for b in anillo[1:]])
    H = np.array([h for anillo, h, _, _ in edificios for _ in anillo[1:]])
    bajo_edificio = dentro_de_edificio(M[:, 0], M[:, 1], edificios)
    bajo_copa = altura_arbol(M[:, 0], M[:, 1], arboles) > 0
    sombra = np.zeros((len(tramos), len(FRANJAS)))
    for k, h0 in enumerate(FRANJAS):
        elev, az = posicion_sol(h0 + 900)
        if elev <= 0:
            sombra[:, k] = 1.0
            continue
        dx, dy = math.sin(math.radians(az)), math.cos(math.radians(az))
        tg = math.tan(math.radians(elev))
        alcance = H.max() / tg
        sombreado = bajo_edificio | bajo_copa
        for paso in np.arange(1.0, arboles[0].max() / tg + 1.0, 1.0):
            sombreado |= altura_arbol(M[:, 0] + paso * dx, M[:, 1] + paso * dy, arboles) > paso * tg
        ex, ey = B[:, 0] - A[:, 0], B[:, 1] - A[:, 1]
        den = dx * ey - dy * ex
        for ini in range(0, len(M), 2000):
            P = M[ini:ini + 2000, :2]
            wx = A[None, :, 0] - P[:, None, 0]
            wy = A[None, :, 1] - P[:, None, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (wx * ey[None, :] - wy * ex[None, :]) / den[None, :]
                u = (wx * dy - wy * dx) / den[None, :]
            golpe = (t > 0) & (t < alcance) & (u >= 0) & (u <= 1) & (t * tg < H[None, :])
            sombreado[ini:ini + 2000] |= golpe.any(axis=1)
        w = M[:, 2]
        sombra[:, k] = np.bincount(dueno, weights=w * sombreado, minlength=len(tramos)) / np.bincount(dueno, weights=w, minlength=len(tramos))
    for i, (_, techado, _) in enumerate(tramos):
        if techado:
            sombra[i, :] = 1.0
    return sombra, largos, bajo_copa


# O(V + E * H)
def escribir(archivo, nodos, pos, nombre, tramos, sombra, largos):
    with open(archivo, "w") as f:
        f.write("franjas %d\n" % len(FRANJAS))
        f.write(" ".join("%02d%02d" % (h // 3600, h % 3600 // 60) for h in FRANJAS) + "\n")
        f.write("puntos %d\n" % len(nodos))
        for c in nodos:
            f.write("%s %.1f %.1f\n" % (nombre[c], pos[c][0], pos[c][1]))
        f.write("tramos %d\n" % len(tramos))
        for i, (coords, techado, tipo) in enumerate(tramos):
            f.write("%s %s %d %s %s\n" % (nombre[coords[0]], nombre[coords[-1]], math.ceil(largos[i]),
                                          "techado" if techado else tipo, " ".join("%.2f" % v for v in sombra[i])))


# O(todo el proceso)
def main():
    archivo = sys.argv[1] if len(sys.argv) > 1 else "export.geojson"
    archivo_arboles = sys.argv[2] if len(sys.argv) > 2 else "arboles_campus.tif"
    lineas, elevada, privadas, edificios = leer(archivo)
    vert = np.array([v for e in edificios if e[2] in NUCLEO for v in e[0]])
    casco = Path(vert[ConvexHull(vert).vertices])
    internas = 0
    for coords, tipo, elev in privadas:
        for parte in recortar(coords):
            if casco.contains_point(xy(*parte[len(parte) // 2])):
                lineas.append((parte, False, tipo))
                elevada.append(elev)
                internas += 1
    conteo = defaultdict(int)
    for coords, _, _ in lineas:
        for c in coords:
            conteo[c] += 1
    pegados = pegar_extremos(lineas, conteo)
    campus = [e for e in edificios if casco.contains_point(e[0][0]) or e[2] in NUCLEO]
    bordes = np.array([[a[0], a[1], b[0], b[1]] for e in campus for a, b in zip(e[0], e[0][1:])])
    calles = segmentos_calles(lineas, CALLES_PROLONGAR)
    prolongados = prolongar(lineas, conteo, casco, bordes, calles, [Path(e[0]) for e in campus])
    cruces = partir_cruces(lineas, elevada)
    tramos = partir_en_tramos(lineas)
    entradas, malla, explanadas = agregar_explanadas(tramos, lineas, edificios, casco, vert)
    componente = componente_mayor(tramos)
    tramos = [t for t in tramos if t[0][0] in componente]
    nodos = sorted(componente)
    pos = {c: xy(*c) for c in nodos}
    xs = [pos[c][0] for c in nodos]
    ys = [pos[c][1] for c in nodos]
    x0, x1, y0, y1 = min(xs) - MARGEN, max(xs) + MARGEN, min(ys) - MARGEN, max(ys) + MARGEN
    edificios = [e for e in edificios if any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in e[0])]
    nombre = nombrar(nodos, pos, edificios)
    arboles = leer_arboles(archivo_arboles, edificios)
    sombra, largos, bajo_copa = calcular_sombra(tramos, edificios, arboles)
    escribir("zona_campus.txt", nodos, pos, nombre, tramos, sombra, largos)
    print("extremos unidos", pegados, "cruces agregados", cruces, "calles internas", internas, "prolongados", prolongados,
          "entradas", entradas, "puntos de malla", malla, "explanadas", explanadas)
    print("puntos", len(nodos), "tramos", len(tramos), "largo total %.0f m" % sum(largos))
    print("edificios", len(edificios), "con altura registrada", sum(1 for e in edificios if e[3]))
    print("pixeles de arbol", int((arboles[0] > 0).sum()), "muestras bajo copa %.1f %%" % (100 * bajo_copa.mean()))


if __name__ == "__main__":
    main()