// Situacion problema
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <string>
#include <vector>

using namespace std;

const double INF = numeric_limits<double>::infinity();
const double EPS = 1e-9;
const double V = 1.25;
const long long LIMITE_NODOS = 10000000;

struct Tramo {
    int destino;
    int largo;
    vector<double> sombra;
};

struct Mapa {
    vector<string> nombres;
    map<string, int> indice;
    vector<vector<Tramo>> ady;
    vector<int> inicios;

    // O(log V)
    int agregarPunto(const string& nombre) {
        indice[nombre] = (int)nombres.size();
        nombres.push_back(nombre);
        ady.push_back({});
        return (int)nombres.size() - 1;
    }

    // O(H)
    void agregarTramo(int u, int w, int largo, const vector<double>& sombra) {
        ady[u].push_back({w, largo, sombra});
        ady[w].push_back({u, largo, sombra});
    }

    // O(V)
    int tramos() const {
        int e = 0;
        for (const auto& l : ady) e += (int)l.size();
        return e / 2;
    }
};

struct Resultado {
    bool hayRuta = false;
    double sol = INF;
    int largo = 0;
    vector<int> ruta;
    bool usoRespaldo = false;
    long long nodos = 0;
    bool excedido = false;
    double cota = 0;
};

// O(1)
int segundos(const string& hhmm) {
    int v = atoi(hhmm.c_str());
    return (v / 100) * 3600 + (v % 100) * 60;
}

// O(V + E * H)
// Lee el mapa desde un archivo de texto con franjas, puntos y tramos
Mapa leerMapa(const string& archivo) {
    Mapa m;
    ifstream f(archivo);
    if (!f) {
        fprintf(stderr, "no se pudo abrir %s\n", archivo.c_str());
        exit(1);
    }
    string etiqueta, nombre, a, b, tipo;
    int h, n, e, largo;
    double x, y;
    f >> etiqueta >> h;
    for (int k = 0; k < h; k++) {
        f >> a;
        m.inicios.push_back(segundos(a));
    }
    f >> etiqueta >> n;
    for (int i = 0; i < n; i++) {
        f >> nombre >> x >> y;
        m.agregarPunto(nombre);
    }
    f >> etiqueta >> e;
    for (int j = 0; j < e; j++) {
        f >> a >> b >> largo >> tipo;
        vector<double> s(h);
        for (double& v : s) f >> v;
        m.agregarTramo(m.indice.at(a), m.indice.at(b), largo, s);
    }
    return m;
}

// O(E log V)
// Distancia minima de cada punto hacia d, solo por longitud
vector<double> dijkstra(const Mapa& m, int d) {
    int n = (int)m.ady.size();
    vector<double> dist(n, INF);
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
    dist[d] = 0;
    pq.push({0, d});
    while (!pq.empty()) {
        auto [du, u] = pq.top();
        pq.pop();
        if (du > dist[u]) continue;
        for (const Tramo& t : m.ady[u]) {
            if (du + t.largo < dist[t.destino]) {
                dist[t.destino] = du + t.largo;
                pq.push({dist[t.destino], t.destino});
            }
        }
    }
    return dist;
}

// O(log H + k)
// Metros al sol de un tramo que empieza despues de caminar x metros, partido por franjas
double solTramo(const Mapa& m, const Tramo& t, double x, int t0) {
    int h = (int)m.inicios.size();
    double instante = t0 + x / V;
    int k = (int)(upper_bound(m.inicios.begin(), m.inicios.end(), (int)floor(instante)) - m.inicios.begin()) - 1;
    if (k < 0) k = 0;
    double total = 0, pos = x, fin = x + t.largo;
    while (pos < fin - EPS) {
        double limite = (k + 1 < h) ? V * (m.inicios[k + 1] - t0) : INF;
        double hasta = min(fin, limite);
        if (hasta > pos) total += (hasta - pos) * (1 - t.sombra[k]);
        pos = max(pos, hasta);
        k++;
    }
    return total;
}

// O(1)
bool mejor(double s1, int l1, double s2, int l2) {
    return s1 < s2 - EPS || (fabs(s1 - s2) <= EPS && l1 < l2);
}

// O(V)
bool esSimple(const vector<int>& ruta, int n) {
    vector<bool> visto(n, false);
    for (int u : ruta) {
        if (visto[u]) return false;
        visto[u] = true;
    }
    return true;
}

// O(k log H)
double solDeRuta(const Mapa& m, const vector<int>& ruta, int t0) {
    double s = 0;
    int x = 0;
    for (size_t i = 0; i + 1 < ruta.size(); i++) {
        for (const Tramo& t : m.ady[ruta[i]]) {
            if (t.destino == ruta[i + 1]) {
                s += solTramo(m, t, x, t0);
                x += t.largo;
                break;
            }
        }
    }
    return s;
}

// O(Lmax * E * log H)
// Tabla hacia adelante, F[x][u] es el menor sol para llegar a u tras caminar exactamente x metros
Resultado programacionDinamica(const Mapa& m, int o, int d, double lmax, int t0, const vector<double>& r) {
    int n = (int)m.ady.size();
    int X = (int)floor(lmax + EPS);
    vector<vector<double>> F(X + 1, vector<double>(n, INF));
    vector<vector<pair<int, int>>> P(X + 1, vector<pair<int, int>>(n, {-1, -1}));
    F[0][o] = 0;
    for (int x = 0; x <= X; x++) {
        for (int u = 0; u < n; u++) {
            if (F[x][u] == INF) continue;
            for (const Tramo& t : m.ady[u]) {
                int nx = x + t.largo;
                if (nx + r[t.destino] > lmax + EPS) continue;
                double c = F[x][u] + solTramo(m, t, x, t0);
                if (c < F[nx][t.destino] - EPS) {
                    F[nx][t.destino] = c;
                    P[nx][t.destino] = {x, u};
                }
            }
        }
    }
    Resultado res;
    int bx = -1;
    for (int x = 0; x <= X; x++)
        if (F[x][d] < INF && (bx == -1 || F[x][d] < F[bx][d] - EPS)) bx = x;
    if (bx == -1) return res;
    res.hayRuta = true;
    res.sol = F[bx][d];
    res.largo = bx;
    for (int x = bx, u = d; x != -1;) {
        res.ruta.push_back(u);
        auto [px, pu] = P[x][u];
        x = px;
        u = pu;
    }
    reverse(res.ruta.begin(), res.ruta.end());
    return res;
}

// O(Lmax * E * log H)
// Tabla hacia atras, G[x][u] es el menor sol para ir de u a d si ya se caminaron x metros
vector<vector<double>> cotaInferior(const Mapa& m, int d, double lmax, int t0, const vector<double>& r) {
    int n = (int)m.ady.size();
    int X = (int)floor(lmax + EPS);
    vector<vector<double>> G(X + 1, vector<double>(n, INF));
    for (int x = X; x >= 0; x--) {
        G[x][d] = 0;
        for (int u = 0; u < n; u++) {
            if (x + r[u] > lmax + EPS) continue;
            for (const Tramo& t : m.ady[u]) {
                int nx = x + t.largo;
                if (nx > X || G[nx][t.destino] == INF) continue;
                G[x][u] = min(G[x][u], G[nx][t.destino] + solTramo(m, t, x, t0));
            }
        }
    }
    return G;
}

// O(k)
// Quita los ciclos de un recorrido y deja una ruta simple que sirve como primera cota superior
vector<int> quitarCiclos(const vector<int>& ruta, int n) {
    vector<int> pos(n, -1), res;
    for (int u : ruta) {
        if (pos[u] != -1) {
            while ((int)res.size() > pos[u] + 1) {
                pos[res.back()] = -1;
                res.pop_back();
            }
            continue;
        }
        pos[u] = (int)res.size();
        res.push_back(u);
    }
    return res;
}

// O(k)
int largoDeRuta(const Mapa& m, const vector<int>& ruta) {
    int L = 0;
    for (size_t i = 0; i + 1 < ruta.size(); i++)
        for (const Tramo& t : m.ady[ruta[i]])
            if (t.destino == ruta[i + 1]) {
                L += t.largo;
                break;
            }
    return L;
}

// Exponencial en el peor caso
// Busqueda de rutas simples, empieza con la ruta sin ciclos y la cota G descarta ramas que no pueden mejorar
Resultado respaldoAcotado(const Mapa& m, int o, int d, double lmax, int t0, const vector<double>& r,
                          const vector<int>& recorrido) {
    vector<vector<double>> G = cotaInferior(m, d, lmax, t0, r);
    int X = (int)G.size() - 1;
    Resultado res;
    if (!recorrido.empty()) {
        vector<int> simple = quitarCiclos(recorrido, (int)m.ady.size());
        int largo = largoDeRuta(m, simple);
        if (largo <= lmax + EPS) {
            res.ruta = simple;
            res.hayRuta = true;
            res.largo = largo;
            res.sol = solDeRuta(m, simple, t0);
        }
    }
    res.cota = G[0][o];
    bool listo = false;
    vector<int> ruta = {o};
    vector<bool> enRuta(m.ady.size(), false);
    enRuta[o] = true;
    function<void(int, int, double)> dfs = [&](int u, int L, double S) {
        if (listo) return;
        if (res.hayRuta && res.sol <= res.cota + EPS) {
            listo = true;
            return;
        }
        res.nodos++;
        if (res.nodos > LIMITE_NODOS) {
            res.excedido = true;
            listo = true;
            return;
        }
        if (L + r[u] > lmax + EPS) return;
        double cota = S + G[L][u];
        if (cota > res.sol + EPS) return;
        if (cota > res.sol - EPS && L + r[u] >= res.largo) return;
        if (u == d) {
            if (mejor(S, L, res.sol, res.largo)) {
                res.hayRuta = true;
                res.sol = S;
                res.largo = L;
                res.ruta = ruta;
            }
            return;
        }
        vector<pair<double, const Tramo*>> hijos;
        for (const Tramo& t : m.ady[u]) {
            if (enRuta[t.destino]) continue;
            int nx = L + t.largo;
            if (nx > X || G[nx][t.destino] == INF) continue;
            hijos.push_back({solTramo(m, t, L, t0) + G[nx][t.destino], &t});
        }
        sort(hijos.begin(), hijos.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& h : hijos) {
            const Tramo& t = *h.second;
            ruta.push_back(t.destino);
            enRuta[t.destino] = true;
            dfs(t.destino, L + t.largo, S + solTramo(m, t, L, t0));
            enRuta[t.destino] = false;
            ruta.pop_back();
        }
    };
    dfs(o, 0, 0.0);
    return res;
}

// Exponencial
// Busqueda exhaustiva sin la cota de la tabla, sirve para comparar
Resultado fuerzaBruta(const Mapa& m, int o, int d, double lmax, int t0, bool simples) {
    vector<double> r = dijkstra(m, d);
    Resultado res;
    vector<int> ruta = {o};
    vector<int> veces(m.ady.size(), 0);
    veces[o] = 1;
    function<void(int, int, double)> dfs = [&](int u, int L, double S) {
        res.nodos++;
        if (res.nodos > LIMITE_NODOS) {
            res.excedido = true;
            return;
        }
        if (L + r[u] > lmax + EPS) return;
        if (S > res.sol + EPS) return;
        if (u == d && mejor(S, L, res.sol, res.largo)) {
            res.hayRuta = true;
            res.sol = S;
            res.largo = L;
            res.ruta = ruta;
        }
        for (const Tramo& t : m.ady[u]) {
            if (simples && veces[t.destino]) continue;
            ruta.push_back(t.destino);
            veces[t.destino]++;
            dfs(t.destino, L + t.largo, S + solTramo(m, t, L, t0));
            veces[t.destino]--;
            ruta.pop_back();
        }
    };
    dfs(o, 0, 0.0);
    return res;
}

// O(Lmax * E * log H) si la tabla da ruta simple, exponencial acotado si no
// Solucion completa, cota de longitud, tabla y respaldo de ruta simple
Resultado resolver(const Mapa& m, int o, int d, double alfa, int t0) {
    vector<double> r = dijkstra(m, d);
    if (r[o] == INF) return Resultado();
    double lmax = alfa * r[o];
    Resultado res = programacionDinamica(m, o, d, lmax, t0, r);
    if (res.hayRuta && !esSimple(res.ruta, (int)m.ady.size())) {
        res = respaldoAcotado(m, o, d, lmax, t0, r, res.ruta);
        res.usoRespaldo = true;
    }
    return res;
}

// O(E log V)
// Ruta mas corta por longitud, la que tomaria alguien sin pensar en el sol
Resultado rutaMasCorta(const Mapa& m, int o, int d, int t0) {
    vector<double> r = dijkstra(m, d);
    Resultado res;
    if (r[o] == INF) return res;
    int u = o;
    res.ruta.push_back(u);
    while (u != d) {
        for (const Tramo& t : m.ady[u]) {
            if (fabs(r[u] - t.largo - r[t.destino]) < EPS) {
                u = t.destino;
                break;
            }
        }
        res.ruta.push_back(u);
    }
    res.hayRuta = true;
    res.largo = (int)r[o];
    res.sol = solDeRuta(m, res.ruta, t0);
    return res;
}

// O(E log V)
// Dijkstra con peso igual al sol de la franja de salida, solo para comparar
Resultado dijkstraIngenuo(const Mapa& m, int o, int d, double alfa, int t0) {
    int n = (int)m.ady.size();
    int k = 0;
    while (k + 1 < (int)m.inicios.size() && m.inicios[k + 1] <= t0) k++;
    vector<double> dist(n, INF);
    vector<int> largo(n, 0), pre(n, -1);
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<>> pq;
    dist[o] = 0;
    pq.push({0, o});
    while (!pq.empty()) {
        auto [du, u] = pq.top();
        pq.pop();
        if (du > dist[u]) continue;
        for (const Tramo& t : m.ady[u]) {
            double c = du + t.largo * (1 - t.sombra[k]);
            if (c < dist[t.destino] - EPS) {
                dist[t.destino] = c;
                largo[t.destino] = largo[u] + t.largo;
                pre[t.destino] = u;
                pq.push({c, t.destino});
            }
        }
    }
    Resultado res;
    double lmax = alfa * dijkstra(m, d)[o];
    if (dist[d] == INF || largo[d] > lmax + EPS) return res;
    res.hayRuta = true;
    res.sol = dist[d];
    res.largo = largo[d];
    for (int u = d; u != -1; u = pre[u]) res.ruta.push_back(u);
    reverse(res.ruta.begin(), res.ruta.end());
    return res;
}

// O(k)
string textoRuta(const Mapa& m, const vector<int>& ruta) {
    string s;
    for (size_t i = 0; i < ruta.size(); i++) s += (i ? " - " : "") + m.nombres[ruta[i]];
    return s;
}

// O(1)
string textoHora(int seg) {
    char b[32];
    snprintf(b, sizeof b, "%d h %02d min", seg / 3600, seg % 3600 / 60);
    return b;
}

// O(1)
// Instancia pequena calculada a mano en el documento
Mapa ejemplo() {
    Mapa m;
    m.inicios = {9 * 3600, 13 * 3600};
    int A = m.agregarPunto("A"), B = m.agregarPunto("B"), C = m.agregarPunto("C");
    int D = m.agregarPunto("D"), E = m.agregarPunto("E"), F = m.agregarPunto("F");
    m.agregarTramo(A, B, 80, {0.25, 0.25});
    m.agregarTramo(A, C, 60, {0.0, 0.0});
    m.agregarTramo(B, D, 70, {0.3, 0.9});
    m.agregarTramo(C, D, 50, {0.2, 0.2});
    m.agregarTramo(C, E, 90, {1.0, 1.0});
    m.agregarTramo(D, F, 60, {0.5, 0.5});
    m.agregarTramo(E, F, 100, {0.8, 0.8});
    return m;
}

// O((n + extra) * H)
Mapa aleatorio(mt19937& gen, int n, int extra, int franjas) {
    Mapa m;
    for (int k = 0; k < franjas; k++) m.inicios.push_back(9 * 3600 + k * 60);
    for (int i = 0; i < n; i++) m.agregarPunto(to_string(i));
    uniform_int_distribution<int> largo(5, 40), s10(0, 10);
    auto sombra = [&]() {
        vector<double> s(franjas);
        for (double& x : s) x = s10(gen) / 10.0;
        return s;
    };
    for (int i = 1; i < n; i++) m.agregarTramo(uniform_int_distribution<int>(0, i - 1)(gen), i, largo(gen), sombra());
    for (int j = 0; j < extra; j++) {
        int u = uniform_int_distribution<int>(0, n - 1)(gen);
        int w = uniform_int_distribution<int>(0, n - 1)(gen);
        if (u != w) m.agregarTramo(u, w, largo(gen), sombra());
    }
    return m;
}

// O(V * H)
// Cuadricula tipo campus con 24 franjas de 30 min, sombra suave por tipo de tramo o aleatoria por franja
Mapa cuadricula(mt19937& gen, int filas, int cols, bool suave) {
    Mapa m;
    for (int h = 7 * 3600; h < 19 * 3600; h += 1800) m.inicios.push_back(h);
    int H = (int)m.inicios.size();
    for (int i = 0; i < filas * cols; i++) m.agregarPunto(to_string(i));
    uniform_int_distribution<int> largo(20, 80);
    uniform_real_distribution<double> u01(0, 1);
    auto sombra = [&]() {
        vector<double> s(H, 0.0);
        if (!suave) {
            for (double& x : s) x = u01(gen);
            return s;
        }
        double tipo = u01(gen), cob = 0.5 + 0.4 * u01(gen);
        for (int k = 0; k < H; k++) {
            if (tipo < 0.15) s[k] = 1.0;
            else if (tipo < 0.40) s[k] = 0.65;
            else if (tipo < 0.60) s[k] = cob * min(1.0, max(0.0, 1.0 - k / 9.0));
            else if (tipo < 0.80) s[k] = cob * min(1.0, max(0.0, (k - 13) / 3.0));
            else s[k] = 0.0;
        }
        return s;
    };
    for (int f = 0; f < filas; f++)
        for (int c = 0; c < cols; c++) {
            int u = f * cols + c;
            if (c + 1 < cols) m.agregarTramo(u, u + 1, largo(gen), sombra());
            if (f + 1 < filas) m.agregarTramo(u, u + cols, largo(gen), sombra());
        }
    return m;
}

// O(1)
double msDesde(chrono::steady_clock::time_point ini) {
    return chrono::duration<double, milli>(chrono::steady_clock::now() - ini).count();
}

int pasadas = 0, fallidas = 0;

// O(1)
void revisar(const string& nombre, bool ok) {
    printf("  [%s] %s\n", ok ? "OK" : "FALLA", nombre.c_str());
    ok ? pasadas++ : fallidas++;
}

// O(k)
void mostrar(const Mapa& m, const Resultado& r) {
    if (!r.hayRuta) {
        printf("    sin ruta\n");
        return;
    }
    printf("    ruta %s, largo %d m, sol %.2f m%s\n", textoRuta(m, r.ruta).c_str(), r.largo, r.sol,
           r.usoRespaldo ? " (respaldo)" : "");
}

// O(Lmax * E * log H) mas el respaldo si se usa
// Consulta de un usuario sobre el mapa leido del archivo
int modoConsulta(const Mapa& m, const string& so, const string& sd, const string& hora, double alfa) {
    if (!m.indice.count(so) || !m.indice.count(sd)) {
        printf("punto desconocido\n");
        return 1;
    }
    int o = m.indice.at(so), d = m.indice.at(sd), t0 = segundos(hora);
    auto ini = chrono::steady_clock::now();
    Resultado r = resolver(m, o, d, alfa, t0);
    double ms = msDesde(ini);
    Resultado c = rutaMasCorta(m, o, d, t0);
    printf("Consulta %s a %s, salida %s, alfa %.2f\n", so.c_str(), sd.c_str(), textoHora(t0).c_str(), alfa);
    if (!r.hayRuta) {
        printf("  No hay ruta entre esos puntos\n");
        return 0;
    }
    printf("  Ruta mas corta    %s\n", textoRuta(m, c.ruta).c_str());
    printf("                    %d m, %.1f m al sol\n", c.largo, c.sol);
    printf("  Ruta recomendada  %s\n", textoRuta(m, r.ruta).c_str());
    printf("                    %d m (limite %.0f m), %.1f m al sol%s\n", r.largo, alfa * c.largo, r.sol,
           r.usoRespaldo ? ", se uso el respaldo de ruta simple" : "");
    if (r.excedido)
        printf("  Aviso, la busqueda llego a su limite y no certifico el optimo, a lo mas %.1f m de sol por encima\n",
               r.sol - r.cota);
    double ahorro = c.sol > EPS ? 100.0 * (c.sol - r.sol) / c.sol : 0.0;
    printf("  Caminas %d m extra (%.0f s mas) y evitas %.1f m de sol (%.0f %% menos)\n", r.largo - c.largo,
           (r.largo - c.largo) / V, c.sol - r.sol, ahorro);
    printf("  Tiempo de calculo %.2f ms\n", ms);
    return 0;
}

// Exponencial por las busquedas exhaustivas de comparacion
// Pruebas de correctitud con resultado esperado
int modoPruebas(const string& archivoZona) {
    Mapa m = ejemplo();
    int A = 0, F = 5;
    Resultado r;

    printf("Prueba 1, ejemplo a mano, salida 9 h, alfa 1.3\n");
    r = resolver(m, A, F, 1.3, 9 * 3600);
    mostrar(m, r);
    revisar("A - C - D - F con 130 m al sol", textoRuta(m, r.ruta) == "A - C - D - F" && fabs(r.sol - 130) < 1e-6);

    printf("Prueba 2, salida 13 h, alfa 1.3\n");
    r = resolver(m, A, F, 1.3, 13 * 3600);
    mostrar(m, r);
    revisar("A - B - D - F con 97 m al sol", textoRuta(m, r.ruta) == "A - B - D - F" && fabs(r.sol - 97) < 1e-6);

    printf("Prueba 3, salida 12 h 58 min 30 s, la ruta cruza de franja dentro del tramo B - D\n");
    r = resolver(m, A, F, 1.3, 12 * 3600 + 58 * 60 + 30);
    mostrar(m, r);
    revisar("A - B - D - F con 116.5 m al sol", textoRuta(m, r.ruta) == "A - B - D - F" && fabs(r.sol - 116.5) < 1e-6);

    printf("Prueba 4, salida 13 h, alfa 1.2 (Lmax 204 m), contra Dijkstra ingenuo\n");
    r = resolver(m, A, F, 1.2, 13 * 3600);
    mostrar(m, r);
    revisar("A - C - D - F con 130 m al sol", textoRuta(m, r.ruta) == "A - C - D - F" && fabs(r.sol - 130) < 1e-6);
    Resultado ing = dijkstraIngenuo(m, A, F, 1.2, 13 * 3600);
    mostrar(m, ing);
    revisar("el Dijkstra ingenuo no encuentra ruta aunque si existe", !ing.hayRuta);

    printf("Prueba 5, salida antes de la primera franja (8 h), usa la sombra de la primera\n");
    r = resolver(m, A, F, 1.3, 8 * 3600);
    mostrar(m, r);
    revisar("igual que a las 9 h", textoRuta(m, r.ruta) == "A - C - D - F" && fabs(r.sol - 130) < 1e-6);

    printf("Prueba 6, destino aislado\n");
    Mapa m2 = ejemplo();
    int G = m2.agregarPunto("G");
    r = resolver(m2, A, G, 1.3, 9 * 3600);
    mostrar(m2, r);
    revisar("responde sin ruta", !r.hayRuta);

    printf("Prueba 7, origen igual a destino\n");
    r = resolver(m, A, A, 1.3, 9 * 3600);
    mostrar(m, r);
    revisar("ruta A con 0 m", r.hayRuta && r.largo == 0 && r.sol == 0);

    printf("Prueba 8, alfa 1.0 solo admite la ruta mas corta\n");
    r = resolver(m, A, F, 1.0, 13 * 3600);
    mostrar(m, r);
    revisar("A - C - D - F con 170 m", textoRuta(m, r.ruta) == "A - C - D - F" && r.largo == 170);

    printf("Prueba 9, alfa muy grande (5.0), la cota deja de limitar\n");
    r = resolver(m, A, F, 5.0, 9 * 3600);
    Resultado b = fuerzaBruta(m, A, F, 5.0 * 170, 9 * 3600, true);
    mostrar(m, r);
    revisar("coincide con la busqueda exhaustiva", fabs(r.sol - b.sol) < 1e-6);

    mt19937 gen(2026);

    printf("Prueba 10, tabla contra busqueda exhaustiva de recorridos (500 grafos)\n");
    int iguales = 0;
    for (int it = 0; it < 500; it++) {
        int n = uniform_int_distribution<int>(4, 8)(gen);
        Mapa g = aleatorio(gen, n, n, 3);
        int t0 = 9 * 3600 + uniform_int_distribution<int>(-30, 150)(gen);
        double alfa = uniform_real_distribution<double>(1.0, 1.6)(gen);
        vector<double> rr = dijkstra(g, n - 1);
        double lmax = alfa * rr[0];
        Resultado a = programacionDinamica(g, 0, n - 1, lmax, t0, rr);
        Resultado c = fuerzaBruta(g, 0, n - 1, lmax, t0, false);
        if (a.hayRuta == c.hayRuta && fabs(a.sol - c.sol) < 1e-6) iguales++;
    }
    printf("    coinciden %d de 500\n", iguales);
    revisar("la tabla da el mismo sol que la busqueda exhaustiva", iguales == 500);

    printf("Prueba 11, solucion completa contra busqueda exhaustiva de rutas simples (3000 grafos)\n");
    iguales = 0;
    int respaldos = 0;
    for (int it = 0; it < 3000; it++) {
        int n = uniform_int_distribution<int>(4, 9)(gen);
        Mapa g = aleatorio(gen, n, n, 3);
        int t0 = 9 * 3600 + uniform_int_distribution<int>(-30, 150)(gen);
        double alfa = uniform_real_distribution<double>(1.0, 1.6)(gen);
        Resultado a = resolver(g, 0, n - 1, alfa, t0);
        Resultado c = fuerzaBruta(g, 0, n - 1, alfa * dijkstra(g, n - 1)[0], t0, true);
        if (a.usoRespaldo) respaldos++;
        if (a.hayRuta == c.hayRuta && fabs(a.sol - c.sol) < 1e-6 && esSimple(a.ruta, n)) iguales++;
    }
    printf("    coinciden %d de 3000, respaldo usado en %d casos (%.2f %%)\n", iguales, respaldos, 100.0 * respaldos / 3000);
    revisar("la solucion completa da el optimo entre rutas simples", iguales == 3000);

    printf("Prueba 12, zona del campus, solucion completa contra busqueda exhaustiva (300 consultas)\n");
    Mapa z = leerMapa(archivoZona);
    int nz = (int)z.ady.size();
    iguales = 0;
    int validas = 0;
    for (int q = 0; q < 300; q++) {
        int o = uniform_int_distribution<int>(0, nz - 1)(gen);
        int d = uniform_int_distribution<int>(0, nz - 1)(gen);
        int t0 = uniform_int_distribution<int>(7 * 3600, 18 * 3600)(gen);
        Resultado a = resolver(z, o, d, 1.3, t0);
        Resultado c = fuerzaBruta(z, o, d, 1.3 * dijkstra(z, d)[o], t0, true);
        if (c.excedido || a.excedido) continue;
        validas++;
        bool largoOk = a.largo <= 1.3 * dijkstra(z, d)[o] + EPS;
        if (a.hayRuta == c.hayRuta && fabs(a.sol - c.sol) < 1e-6 && esSimple(a.ruta, nz) && largoOk) iguales++;
    }
    printf("    coinciden %d de %d\n", iguales, validas);
    revisar("en la zona del campus el resultado es optimo y respeta Lmax", iguales == validas && validas > 0);

    printf("\nResumen, %d pruebas OK y %d fallidas\n", pasadas, fallidas);
    return fallidas == 0 ? 0 : 1;
}

// O(consultas * Lmax * E * log H) mas los respaldos
// Experimentos con semilla 2026, escriben archivos CSV en la carpeta indicada
int modoExperimentos(const string& archivoZona, const string& carpeta) {
    Mapa z = leerMapa(archivoZona);
    int nz = (int)z.ady.size();
    mt19937 gen(2026);

    vector<array<int, 3>> consultas;
    while ((int)consultas.size() < 400) {
        int o = uniform_int_distribution<int>(0, nz - 1)(gen);
        int d = uniform_int_distribution<int>(0, nz - 1)(gen);
        int t0 = uniform_int_distribution<int>(7 * 3600, 18 * 3600)(gen);
        if (o != d) consultas.push_back({o, d, t0});
    }

    FILE* fa = fopen((carpeta + "/exp_alfa.csv").c_str(), "w");
    fprintf(fa, "alfa,lmax_prom,ms_tabla_prom,ms_total_prom,ms_total_peor,respaldo_pct,extra_m_prom,ahorro_pct_prom,sin_certificar_pct,brecha_max_m\n");
    for (double alfa = 1.0; alfa <= 1.61; alfa += 0.1) {
        double sl = 0, st = 0, sto = 0, pto = 0, sx = 0, sa = 0, brecha = 0;
        int resp = 0, nA = 0, sinCert = 0;
        for (auto [o, d, t0] : consultas) {
            vector<double> r = dijkstra(z, d);
            auto ini = chrono::steady_clock::now();
            programacionDinamica(z, o, d, alfa * r[o], t0, r);
            st += msDesde(ini);
            ini = chrono::steady_clock::now();
            Resultado res = resolver(z, o, d, alfa, t0);
            double ms = msDesde(ini);
            sto += ms;
            pto = max(pto, ms);
            Resultado c = rutaMasCorta(z, o, d, t0);
            sl += alfa * r[o];
            sx += res.largo - c.largo;
            if (c.sol > EPS) {
                sa += 100.0 * (c.sol - res.sol) / c.sol;
                nA++;
            }
            if (res.usoRespaldo) resp++;
            if (res.excedido) {
                sinCert++;
                brecha = max(brecha, res.sol - res.cota);
            }
        }
        int q = (int)consultas.size();
        fprintf(fa, "%.1f,%.1f,%.3f,%.3f,%.3f,%.2f,%.1f,%.2f,%.2f,%.1f\n", alfa, sl / q, st / q, sto / q, pto, 100.0 * resp / q,
                sx / q, sa / max(1, nA), 100.0 * sinCert / q, brecha);
        printf("alfa %.1f listo\n", alfa);
    }
    fclose(fa);

    FILE* fh = fopen((carpeta + "/exp_hora.csv").c_str(), "w");
    fprintf(fh, "hora,sol_corta_prom,sol_opt_prom,ahorro_pct_prom,extra_m_prom\n");
    for (int hora = 7; hora <= 18; hora++) {
        double sc = 0, so = 0, sa = 0, sx = 0;
        int nA = 0, q = 0;
        for (const auto& cq : consultas) {
            int o = cq[0], d = cq[1];
            Resultado res = resolver(z, o, d, 1.3, hora * 3600);
            Resultado c = rutaMasCorta(z, o, d, hora * 3600);
            sc += c.sol;
            so += res.sol;
            sx += res.largo - c.largo;
            if (c.sol > EPS) {
                sa += 100.0 * (c.sol - res.sol) / c.sol;
                nA++;
            }
            q++;
        }
        fprintf(fh, "%d,%.1f,%.1f,%.2f,%.1f\n", hora, sc / q, so / q, sa / max(1, nA), sx / q);
        printf("hora %d lista\n", hora);
    }
    fclose(fh);

    FILE* fe = fopen((carpeta + "/exp_escala.csv").c_str(), "w");
    fprintf(fe, "modelo,V,E,lmax_prom,ms_tabla_prom,ms_tabla_peor,ns_por_estado,respaldo_pct,ms_acotado_prom,ms_acotado_peor,"
                "nodos_acotado_prom,ms_exhaustivo_prom,nodos_exhaustivo_prom,exhaustivo_excedido\n");
    for (int suave = 1; suave >= 0; suave--) {
        for (int lado = 10; lado <= 40; lado += 5) {
            Mapa c = cuadricula(gen, lado, lado, suave);
            int n = lado * lado, e = c.tramos(), Q = 100;
            double sl = 0, st = 0, pt = 0, sns = 0, sa = 0, pa = 0, sna = 0, sb = 0, snb = 0;
            int resp = 0, exc = 0, compar = 0;
            for (int q = 0; q < Q; q++) {
                int o = uniform_int_distribution<int>(0, n - 1)(gen);
                int d = uniform_int_distribution<int>(0, n - 1)(gen);
                int t0 = uniform_int_distribution<int>(7 * 3600, 18 * 3600)(gen);
                vector<double> r = dijkstra(c, d);
                double lmax = 1.3 * r[o];
                auto ini = chrono::steady_clock::now();
                Resultado res = programacionDinamica(c, o, d, lmax, t0, r);
                double ms = msDesde(ini);
                st += ms;
                pt = max(pt, ms);
                sl += lmax;
                sns += ms * 1e6 / max(1.0, (floor(lmax) + 1) * 2.0 * e);
                if (res.hayRuta && !esSimple(res.ruta, n)) {
                    resp++;
                    ini = chrono::steady_clock::now();
                    Resultado a = respaldoAcotado(c, o, d, lmax, t0, r, res.ruta);
                    double ma = msDesde(ini);
                    sa += ma;
                    pa = max(pa, ma);
                    sna += a.nodos;
                    {
                        ini = chrono::steady_clock::now();
                        Resultado b = fuerzaBruta(c, o, d, lmax, t0, true);
                        sb += msDesde(ini);
                        snb += b.nodos;
                        compar++;
                        if (b.excedido) exc++;
                    }
                }
            }
            fprintf(fe, "%s,%d,%d,%.1f,%.3f,%.3f,%.3f,%.1f,%.2f,%.2f,%.0f,%.2f,%.0f,%d\n", suave ? "suave" : "aleatoria", n, e,
                    sl / Q, st / Q, pt, sns / Q, 100.0 * resp / Q, resp ? sa / resp : 0.0, pa, resp ? sna / resp : 0.0,
                    compar ? sb / compar : -1.0, compar ? snb / compar : -1.0, exc);
            printf("%s %d listo\n", suave ? "suave" : "aleatoria", n);
            fflush(fe);
        }
    }
    fclose(fe);
    return 0;
}

// O(1) mas el modo elegido
int main(int argc, char** argv) {
    string modo = argc > 1 ? argv[1] : "";
    if (modo == "consulta" && argc == 7) return modoConsulta(leerMapa(argv[2]), argv[3], argv[4], argv[5], atof(argv[6]));
    if (modo == "pruebas" && argc == 3) return modoPruebas(argv[2]);
    if (modo == "experimentos" && argc == 4) return modoExperimentos(argv[2], argv[3]);
    printf("Uso\n");
    printf("  ruta_sombra consulta zona_campus.txt ORIGEN DESTINO HHMM ALFA\n");
    printf("  ruta_sombra pruebas zona_campus.txt\n");
    printf("  ruta_sombra experimentos zona_campus.txt CARPETA\n");
    return 1;
}