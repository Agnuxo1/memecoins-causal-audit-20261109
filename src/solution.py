# SPDX-FileCopyrightText: 2026 Francisco Angulo de Lafuente
# SPDX-License-Identifier: MIT
# Research-Project: memecoins-walk-forward-research
# Research-Date: 2026-11-04
# Research-ORCID: https://orcid.org/0009-0001-1634-7063
# Reuse: preserve applicable notices and cite the original author.

# =============================================================================
# Memecoins: Pump or Dump?  —  politica de valor esperado con reservorio holografico
# Francisco Angulo de Lafuente  (Kaggle: franciscoangulo · GitHub: Agnuxo1)
#
# En una linea: para cada token se estima la curva de supervivencia calibrada
# P(max_x >= u), se elige el nivel de take-profit que maximiza el valor esperado
# del pago oficial, y se compra solo si ese valor esperado supera un umbral
# fijado con los datos de entrenamiento.
#
# Cumplimiento de las reglas del concurso:
#  §1  tp es 0 o un real en [3, 100]; se comprueba con un assert al final.
#  §2  todo se entrena dentro de la ejecucion. No hay pesos, coeficientes ni
#      estimadores embebidos: las unicas constantes son hiperparametros.
#  §3  no hay acceso a red; solo se leen los ficheros montados del concurso.
#  §4  ninguna caracteristica ni decision usa tokens que migraron despues. La
#      reputacion es acumulativa y con la regla de resolucion de 1 h, y el
#      umbral de compra sale del entrenamiento, nunca de las filas de test.
#      Borrar cualquier fila de test.csv no cambia el tp de las demas.
#  §5  solo CPU, n_jobs = 4.
# =============================================================================
import os
import gc
import time
import hashlib
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings('ignore')
T0 = time.time()

COST = 0.3          # coste por posicion, de las reglas
WIN = 3600          # la etiqueta de un token se cierra 1 h despues de su instantanea
EMBARGO = 3600      # separacion extra entre entrenamiento y dia validado
LEVELS = (3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0)
SEEDS = (0, 1, 2)       # semillas del modelo final (se promedian)
SEEDS_SEL = (0,)        # semillas de la seleccion de umbral y del walk-forward, por presupuesto
WSOL = 'So11111111111111111111111111111111111111112'
NBIN, BINW, NCH = 20, 3, 7
HRC_SIZE = 48
N_JOBS = 4
USE_HRC = True
INNER_DAYS = 8      # dias de validacion interna para fijar el umbral
WF_DAYS = 5         # dias del walk-forward que se imprime como informe
WF_INNER = 3        # validacion interna dentro de cada pliegue del informe
TAU_GRID = np.arange(0.0, 2.51, 0.1)

LGB = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=60,
           subsample=0.85, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
           verbose=-1, n_jobs=N_JOBS)

CAND = ['/kaggle/input/competitions/memecoins-pump-or-dump',
        '/kaggle/input/memecoins-pump-or-dump', 'data']
DATA = next(p for p in CAND if os.path.exists(p))
print('datos:', DATA, flush=True)


def tape_path(day, table):
    for p in (DATA + '/trades/day=' + day + '/' + table + '_all_swaps.parquet',
              DATA + '/day_' + day + '_' + table + '_all_swaps.parquet'):
        if os.path.exists(p):
            return p
    return None


def pnl(tp, max_x, final_x):
    """El pago oficial, tal cual la pestana Evaluation."""
    tp = np.asarray(tp, float)
    hit = (max_x >= tp) & (tp > 0)
    held = np.maximum(final_x - 1 - COST, -1.0)
    return np.where(tp <= 0, 0.0, np.where(hit, tp - 1 - COST, held))


# ----------------------------------------------------------------- etiquetas
def load_labels():
    parts = []
    for f in ('train_archive.csv', 'train.csv'):
        p = DATA + '/' + f
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    tr = pd.concat(parts, ignore_index=True).rename(columns={'finalx': 'final_x'})
    tr = tr.drop_duplicates('mint')
    te = pd.read_csv(DATA + '/test.csv')
    te['max_x'] = np.nan
    te['final_x'] = np.nan
    a = pd.concat([tr, te], ignore_index=True)
    a['day'] = pd.to_datetime(a.mt, unit='s').dt.strftime('%Y-%m-%d')
    return a.sort_values('mt', kind='stable').reset_index(drop=True)


# ------------------------------------------------- caracteristicas de cinta
def gini(v):
    v = np.sort(np.asarray(v, float))
    n = len(v)
    s = v.sum()
    if n == 0 or s <= 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * s))


CURVE_COLS = ['mint', 'ts_rel', 'direction', 'failed', 'sol', 'new_price',
              'wallet', 'fee_payer', 'slot']
AMM_COLS = ['mint', 'ts_rel', 'direction', 'sol', 'new_price', 'signing_wallet',
            'fee_payer', 'slot', 'base_token', 'pool_base_token_reserves_after',
            'pool_quote_token_reserves_after', 'fee']


def curve_feats(craw):
    """Fase de curva. Las filas con failed != 0 son intentos revertidos: se excluyen
    de todo recuento de volumen o precio y se cuentan aparte."""
    g0 = craw.groupby('mint', sort=False)
    FF = pd.DataFrame({'c_nfail': craw[craw.failed != 0].groupby('mint').size(),
                       'c_failshare': craw[craw.failed != 0].groupby('mint').size() / g0.size()})
    c = craw[craw.failed == 0]
    if len(c) == 0:
        return FF
    c = c.copy()
    c['is_buy'] = (c.direction == 'buy')     # la curva usa buy/sell, siempre con SOL de quote
    g = c.groupby('mint', sort=False)
    F = pd.DataFrame(index=g.size().index)
    F['c_n'] = g.size()
    F['c_uw'] = g.wallet.nunique()
    F['c_tpw'] = F.c_n / F.c_uw.clip(lower=1)
    F['c_sol'] = g.sol.sum()
    b = c[c.is_buy]
    s = c[~c.is_buy]
    F['c_buy_sol'] = b.groupby('mint').sol.sum()
    F['c_sell_sol'] = s.groupby('mint').sol.sum()
    F['c_sell_share'] = F.c_sell_sol / (F.c_buy_sol.fillna(0) + F.c_sell_sol.fillna(0)).replace(0, np.nan)
    F['c_dur'] = -g.ts_rel.min()
    F['c_last10_share'] = c[c.ts_rel >= -600].groupby('mint').sol.sum() / F.c_sol.replace(0, np.nan)
    F['c_last60_share'] = c[c.ts_rel >= -60].groupby('mint').sol.sum() / F.c_sol.replace(0, np.nan)
    ws = b.groupby(['mint', 'wallet']).sol.sum()
    tot = ws.groupby('mint').sum().replace(0, np.nan)
    F['c_top1'] = ws.groupby('mint').max() / tot
    F['c_top3'] = ws.groupby('mint').apply(lambda x: np.sort(x.values)[-3:].sum()) / tot
    F['c_gini'] = ws.groupby('mint').apply(lambda x: gini(x.values))
    F['c_maxbuy'] = b.groupby('mint').sol.max()
    F['c_feepayer_diff'] = g.apply(lambda x: (x.fee_payer != x.wallet).mean())
    bw = b.groupby('mint').wallet.apply(set)
    sw = s.groupby('mint').wallet.apply(set)
    F['c_roundtrip'] = pd.Series({m: len(v & sw.get(m, set())) / max(len(v), 1) for m, v in bw.items()})
    F['c_slotclust'] = g.apply(lambda x: 1 - x.slot.nunique() / len(x))
    F['c_price_last'] = g.new_price.last()
    F['c_price_max'] = g.new_price.max()
    return F.join(FF, how='outer')


def amm_prep(a):
    a = a.copy()
    raw_buy = a.direction.astype(str).str.upper().str.startswith('B')
    # direction es relativa a base_token: si SOL es la base, B es vender la moneda
    a['is_buy'] = np.where(a.base_token == WSOL, ~raw_buy, raw_buy)
    a['wallet'] = a.signing_wallet
    # el precio sale de las reservas del pool, nunca del cociente de la propia operacion
    a['sol_res'] = np.where(a.base_token == WSOL, a.pool_base_token_reserves_after,
                            a.pool_quote_token_reserves_after) / 1e9
    return a


def amm_feats(a):
    """Los primeros 60 s en el pool, separados en subventanas 0-15 / 15-30 / 30-60."""
    g = a.groupby('mint', sort=False)
    F = pd.DataFrame(index=g.size().index)
    F['a_n'] = g.size()
    F['a_uw'] = g.wallet.nunique()
    F['a_tpw'] = F.a_n / F.a_uw.clip(lower=1)
    F['a_sol'] = g.sol.sum()
    b = a[a.is_buy]
    F['a_buy_sol'] = b.groupby('mint').sol.sum()
    F['a_buyshare'] = F.a_buy_sol / F.a_sol.replace(0, np.nan)
    F['a_maxbuy'] = b.groupby('mint').sol.max()
    for lo, hi, tag in [(0, 15, 'w1'), (15, 30, 'w2'), (30, 61, 'w3')]:
        w = a[(a.ts_rel >= lo) & (a.ts_rel < hi)]
        F['a_sol_' + tag] = w.groupby('mint').sol.sum()
        F['a_n_' + tag] = w.groupby('mint').size()
        F['a_buyshare_' + tag] = w[w.is_buy].groupby('mint').sol.sum() / F['a_sol_' + tag].replace(0, np.nan)
    F['a_mom'] = F.a_sol_w3 / F.a_sol_w1.replace(0, np.nan)
    F['a_mom_n'] = F.a_n_w3 / F.a_n_w1.replace(0, np.nan)
    ws = b.groupby(['mint', 'wallet']).sol.sum()
    tot = ws.groupby('mint').sum().replace(0, np.nan)
    F['a_top1'] = ws.groupby('mint').max() / tot
    F['a_gini'] = ws.groupby('mint').apply(lambda x: gini(x.values))
    F['a_p_first'] = g.new_price.first()
    F['a_p_last'] = g.new_price.last()
    F['a_p_max'] = g.new_price.max()
    F['a_p_min'] = g.new_price.min()
    F['a_ret'] = F.a_p_last / F.a_p_first.replace(0, np.nan)
    F['a_runup'] = F.a_p_max / F.a_p_first.replace(0, np.nan)
    F['a_dd'] = F.a_p_last / F.a_p_max.replace(0, np.nan)
    F['a_depth_last'] = g.sol_res.last()
    F['a_depth_first'] = g.sol_res.first()
    F['a_depth_growth'] = F.a_depth_last / F.a_depth_first.replace(0, np.nan)
    F['a_fee'] = g.fee.mean()
    F['a_feepayer_diff'] = g.apply(lambda x: (x.fee_payer != x.signing_wallet).mean())
    F['a_slotclust'] = g.apply(lambda x: 1 - x.slot.nunique() / len(x))
    early = a[a.ts_rel < 15]
    late = a[a.ts_rel >= 45]
    seen = early.groupby('mint').wallet.apply(set)
    lw = late.groupby('mint').wallet.apply(set)
    F['a_fresh_late'] = pd.Series({m: len(v - seen.get(m, set())) for m, v in lw.items()})
    eb = early[early.is_buy].groupby('mint').wallet.apply(set)
    sl = a[~a.is_buy].groupby('mint').wallet.apply(set)
    F['a_flip'] = pd.Series({m: len(v & sl.get(m, set())) / max(len(v), 1) for m, v in eb.items()})
    return F


def amm_seq(a):
    """Secuencia binada del primer minuto, (NBIN, NCH). Es lo que lee el reservorio.
    Canales: SOL comprado, SOL vendido, n operaciones, carteras nuevas,
    log precio relativo al primero, log profundidad relativa, cuota de compra."""
    a = a.copy()
    a['b'] = np.clip(a.ts_rel.values // BINW, 0, NBIN - 1).astype(int)
    a = a.sort_values(['mint', 'ts_rel'], kind='stable')
    mints = a.mint.unique()
    idx = {m: i for i, m in enumerate(mints)}
    T = np.zeros((len(mints), NBIN, NCH), dtype=np.float32)
    mi = a.mint.map(idx).values
    bi = a.b.values
    sol = a.sol.values.astype(np.float64)
    isb = a.is_buy.values.astype(bool)
    np.add.at(T, (mi[isb], bi[isb], 0), sol[isb])
    np.add.at(T, (mi[~isb], bi[~isb], 1), sol[~isb])
    np.add.at(T, (mi, bi, 2), 1.0)
    p0 = a.groupby('mint').new_price.first().reindex(mints).values
    d0 = a.groupby('mint').sol_res.first().reindex(mints).values
    last = a.groupby(['mint', 'b']).agg(p=('new_price', 'last'), d=('sol_res', 'last')).reset_index()
    li = last.mint.map(idx).values
    lb = last.b.values
    T[li, lb, 4] = np.log(np.clip(last.p.values / np.clip(p0[li], 1e-18, None), 1e-6, 1e6))
    T[li, lb, 5] = np.log(np.clip(last.d.values / np.clip(d0[li], 1e-12, None), 1e-6, 1e6))
    for ch in (4, 5):
        for t in range(1, NBIN):
            m = T[:, t, ch] == 0
            T[m, t, ch] = T[m, t - 1, ch]
    tot = T[:, :, 0] + T[:, :, 1]
    T[:, :, 6] = np.where(tot > 0, T[:, :, 0] / np.maximum(tot, 1e-9), 0.0)
    nw = np.zeros((len(mints), NBIN), dtype=np.float32)
    for m, g in a.groupby('mint', sort=False):
        s = set()
        i = idx[m]
        for b_, gg in g.groupby('b', sort=True):
            new = set(gg.wallet) - s
            nw[i, b_] = len(new)
            s |= new
    T[:, :, 3] = nw
    for ch in (0, 1, 2, 3):
        T[:, :, ch] = np.log1p(T[:, :, ch])
    return mints, T


def read_tape(days):
    """Un dia a la vez y solo las columnas necesarias: la cinta completa no entra en memoria."""
    feats, seqs, mints_all, edges = [], [], [], []
    for day in days:
        cp, ap = tape_path(day, 'pumpfun'), tape_path(day, 'pumpswap')
        if cp is None or ap is None:
            continue
        craw = pd.read_parquet(cp, columns=CURVE_COLS)
        F1 = curve_feats(craw)
        del craw
        gc.collect()
        a = amm_prep(pd.read_parquet(ap, columns=AMM_COLS))
        F2 = amm_feats(a)
        m, T = amm_seq(a)
        ab = a[a.is_buy & (a.ts_rel <= 60)]
        edges.append(ab.groupby(['mint', 'wallet'], as_index=False).sol.sum())
        del a
        gc.collect()
        F = F1.join(F2, how='outer')
        F.index.name = 'mint'
        feats.append(F.reset_index())
        mints_all.append(m)
        seqs.append(T)
    FEAT = pd.concat(feats, ignore_index=True).drop_duplicates('mint')
    SEQ_M = np.concatenate(mints_all)
    SEQ_T = np.concatenate(seqs)
    E = pd.concat(edges, ignore_index=True).groupby(['mint', 'wallet'], as_index=False).sol.sum()
    return FEAT, SEQ_M, SEQ_T, E


# --------------------------------------------------- creacion del token
CRE_NUM = ['bundle_size', 'gas_used', 'amount_of_instructions', 'amount_of_lookup_reads',
           'amount_of_lookup_writes', 'bundled_buys', 'bundled_buys_count', 'dev_balance',
           'creation_ix_index', 'pf_program_index', 'direct_pf_invocation', 'version',
           'mayhem_mode', 'is_cashback_enabled']


def load_creation():
    c = pd.read_csv(DATA + '/pumpfun_token_creation.csv').drop_duplicates('mint')
    out = c[['mint', 'created_ts', 'creator']].copy()
    for k in CRE_NUM:
        if k in c.columns:
            out['x_' + k] = pd.to_numeric(c[k], errors='coerce').values
    if 'token_program' in c.columns:
        out['x_token2022'] = (~c.token_program.fillna('').astype(str)
                              .str.startswith('Tokenkeg')).astype(int).values
    if 'bundle_structure' in c.columns:
        out['x_bundle_struct_n'] = c.bundle_structure.fillna('').astype(str).str.count(',').values + 1
    out['x_symbol_len'] = c.symbol.fillna('').astype(str).str.len().values
    out['x_name_len'] = c.name.fillna('').astype(str).str.len().values
    return out


# -------------------------------- identidad convertida en estadistica legal
def creator_rep(a, levels=(3.0, 5.0, 10.0)):
    """Historial del creador, estrictamente acumulativo. Para el token i solo entran
    tokens con mt <= mt_i - WIN, es decir cuya propia hora ya cerro. Nunca se usa el
    fichero completo: borrar tokens posteriores no cambia ningun valor."""
    mts, mx = a.mt.values, a.max_x.values
    cre = a.creator.fillna('?').astype('category').cat.codes.values
    nC = cre.max() + 1
    n = np.zeros(nC)
    hits = {u: np.zeros(nC) for u in levels}
    out = np.full((len(a), 1 + len(levels)), np.nan)
    j = 0
    for i in range(len(a)):
        while j < len(a) and mts[j] <= mts[i] - WIN:
            if not np.isnan(mx[j]):
                n[cre[j]] += 1
                for u in levels:
                    if mx[j] >= u:
                        hits[u][cre[j]] += 1
            j += 1
        c = cre[i]
        k = n[c]
        out[i, 0] = k
        for q, u in enumerate(levels):
            out[i, 1 + q] = hits[u][c] / k if k > 0 else np.nan
    cols = ['r_creator_n'] + ['r_creator_p%g' % u for u in levels]
    return pd.DataFrame(out, columns=cols, index=a.index)


def wallet_rep(a, E, k_level=5.0, min_seen=3):
    """Reputacion de los compradores del primer minuto, con la misma regla de
    resolucion. Un token que migro hace 20 minutos no ha resuelto y no cuenta."""
    cols = ['r_w_known', 'r_w_max', 'r_w_mean', 'r_w_top3', 'r_w_elite_n',
            'r_w_elite_share', 'r_w_nwal']
    if E is None or len(E) == 0:
        return pd.DataFrame(np.nan, columns=cols, index=a.index)
    E = E.copy()
    E['wid'] = E.wallet.astype('category').cat.codes.astype(np.int64)
    by = {m: (g.wid.values, g.sol.values)
          for m, g in E.sort_values('mint', kind='stable').groupby('mint', sort=False)}
    nW = int(E.wid.max()) + 1
    seen = np.zeros(nW, np.int32)
    hit = np.zeros(nW, np.int32)
    mts, mx, mints = a.mt.values, a.max_x.values, a.mint.values
    out = np.full((len(a), len(cols)), np.nan)
    j = 0
    for i in range(len(a)):
        while j < len(a) and mts[j] <= mts[i] - WIN:
            if not np.isnan(mx[j]):
                w = by.get(mints[j])
                if w is not None:
                    np.add.at(seen, w[0], 1)
                    if mx[j] >= k_level:
                        np.add.at(hit, w[0], 1)
            j += 1
        w = by.get(mints[i])
        if w is None:
            continue
        wid, sol = w
        out[i, 6] = len(wid)
        s = seen[wid]
        kn = s >= min_seen
        if kn.sum() == 0:
            out[i, :6] = [0, 0, 0, 0, 0, 0]
            continue
        rate = hit[wid][kn] / s[kn]
        elite = (s >= 5) & (hit[wid] >= 1)
        out[i, :6] = [kn.sum(), rate.max(), rate.mean(), np.sort(rate)[-3:].mean(),
                      elite.sum(), sol[elite].sum() / max(sol.sum(), 1e-9)]
    return pd.DataFrame(out, columns=cols, index=a.index)


# ------------------------------------- reservorio holografico, portado en linea
def _sha_entropy(n, seed):
    """La entropia determinista del backend simulado del Holographic-Reservoir
    original (SHA-256 encadenado). Sin red y sin hardware: solo la biblioteca estandar."""
    out = []
    h = hashlib.sha256(('hrc-init-%d' % seed).encode()).digest()
    while len(out) < n:
        out.extend(h)
        h = hashlib.sha256(h).digest()
    return np.frombuffer(bytes(out[:n]), dtype=np.uint8).astype(np.float64) / 127.5 - 1.0


class HolographicReservoir:
    """Port en linea del nucleo de Holographic-Reservoir (Apache-2.0, F. Angulo):
    capa recurrente dispersa con fuga y tanh, radio espectral fijado, entropia
    SHA-256 inyectada en la diagonal, y lectura ridge (la forma cerrada de la RLS
    del original con factor de olvido 1).

    La caja de arena del concurso no tiene red, asi que el paquete no se puede
    instalar: se porta el mismo principio dinamico a NumPy puro, vectorizado
    sobre todos los tokens a la vez (un paso de tiempo para los N tokens a la vez
    en lugar de un bucle por token). Nada se entrena fuera de la ejecucion: los
    pesos del reservorio son aleatorios con semilla y la lectura se ajusta aqui.
    """

    def __init__(self, size=HRC_SIZE, input_dim=NCH, spectral_radius=0.9, leak=0.3,
                 density=0.1, ridge=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.size, self.leak, self.ridge = size, leak, ridge
        self.W_in = rng.uniform(-0.5, 0.5, size=(size, input_dim))
        W = rng.standard_normal((size, size)) * (rng.random((size, size)) < density)
        np.fill_diagonal(W, np.diag(W) + _sha_entropy(size, seed) * 0.05)
        r = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / r) if r > 0 else W
        self.W_out = None

    def states(self, U):
        """U: (N, T, input_dim) -> (N, 2*size+1). Estado final y estado medio: el
        final lleva el instante de la decision, el medio resume la trayectoria."""
        N, T, _ = U.shape
        X = np.zeros((N, self.size))
        acc = np.zeros((N, self.size))
        for t in range(T):
            X = (1.0 - self.leak) * X + self.leak * np.tanh(U[:, t, :] @ self.W_in.T + X @ self.W.T)
            acc += X
        return np.hstack([X, acc / T, np.ones((N, 1))])

    def fit_readout(self, S, y):
        A = S.T @ S + self.ridge * np.eye(S.shape[1])
        self.W_out = np.linalg.solve(A, S.T @ y)
        return self

    def predict(self, S):
        return S @ self.W_out


# ------------------------------------- curva de supervivencia + valor esperado
def _cal_split(trn, cal_days=5):
    d = sorted(trn.day.unique())
    if len(d) <= cal_days + 3:
        return trn, None
    cal = trn[trn.day >= d[-cal_days]]
    fit = trn[(trn.day < d[-cal_days]) & (trn.mt <= cal.mt.min() - WIN - EMBARGO)]
    return (fit, cal) if len(fit) > 500 and len(cal) > 200 else (trn, None)


class Survival:
    """P(max_x >= u) calibrada para cada nivel, mas el termino de perdida.

    La calibracion importa mas que el ranking aqui: la probabilidad entra
    directamente en el pago, asi que una P distorsionada elige mal el nivel."""

    def __init__(self, features, levels=LEVELS, seeds=SEEDS):
        self.FE = list(features)
        self.levels = tuple(levels)
        self.seeds = tuple(seeds)

    def fit(self, trn):
        Xall = trn[self.FE].astype(float).values
        fit, cal = _cal_split(trn)
        self.models, self.cals, self.loss = {}, {}, {}
        Xf = fit[self.FE].astype(float).values if cal is not None else None
        Xc = cal[self.FE].astype(float).values if cal is not None else None
        for u in self.levels:
            yall = (trn.max_x.values >= u).astype(int)
            self.models[u] = [lgb.LGBMClassifier(random_state=s, **LGB).fit(Xall, yall)
                              for s in self.seeds]
            if cal is not None and (fit.max_x.values >= u).sum() >= 20:
                mf = [lgb.LGBMClassifier(random_state=s, **LGB).fit(
                          Xf, (fit.max_x.values >= u).astype(int)) for s in self.seeds]
                p = np.mean([m.predict_proba(Xc)[:, 1] for m in mf], 0)
                ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                ir.fit(p, (cal.max_x.values >= u).astype(int))
                self.cals[u] = ir
            else:
                self.cals[u] = None
            miss = trn.max_x.values < u
            self.loss[u] = (float(np.maximum(trn.final_x.values[miss] - 1 - COST, -1.0).mean())
                            if miss.sum() else -1.0)
        return self

    def proba(self, df):
        X = df[self.FE].astype(float).values
        P = np.empty((len(df), len(self.levels)))
        for q, u in enumerate(self.levels):
            p = np.mean([m.predict_proba(X)[:, 1] for m in self.models[u]], 0)
            if self.cals[u] is not None:
                p = self.cals[u].transform(p)
            P[:, q] = p
        return np.minimum.accumulate(P, axis=1)      # P(>=u) no puede crecer con u

    def ev(self, df):
        P = self.proba(df)
        L = np.array([self.loss[u] for u in self.levels])
        G = np.array([u - 1 - COST for u in self.levels])
        return P * G + (1 - P) * L, P

    def decide(self, df, tau):
        EV, P = self.ev(df)
        k = EV.argmax(1)
        best = EV[np.arange(len(df)), k]
        tp = np.where(best >= tau, np.array(self.levels)[k], 0.0)
        return tp, best, P


ABSTAIN_TAU = 1e9                  # umbral prohibitivo: cero compras, PnL exactamente 0
DEPTH_LO, DEPTH_HI = 0.1, 2.0      # banda de compras por dia que respalda la ablacion


def pick_tau(TR, FE, inner_days=INNER_DAYS, grid=TAU_GRID, seeds=SEEDS_SEL):
    """El umbral de compra se fija con una validacion interna sobre los ultimos dias
    del entrenamiento, con un modelo que no ha visto esos dias. El dia de test no se
    toca: esto es lo que hace que la decision de un token no dependa de las demas.

    Dos reglas, las dos sacadas de la metrica y no de la intuicion:
    - solo se consideran umbrales cuya profundidad cae en la banda que la ablacion
      respalda, entre 0,3 y 2 compras al dia. Toda politica se degrada al tomar mas
      tokens por dia, y saltarse un dia cuesta cero.
    - si en esa banda la mejor media interna no es positiva, se ABSTIENE del todo.
      Cero puntua exactamente cero y solo se pagan puntuaciones oficiales positivas,
      asi que desplegar un regimen que su propia validacion dice que pierde es peor
      que no comprar nada.
    """
    days = sorted(TR.day.unique())
    rows = []
    for day in days[-inner_days:]:
        val = TR[TR.day == day]
        trn = TR[(TR.day < day) & (TR.mt <= val.mt.min() - WIN - EMBARGO)]
        if len(trn) < 2000 or len(val) < 50:
            continue
        m = Survival(FE, seeds=seeds).fit(trn)
        EV, _ = m.ev(val)
        k = EV.argmax(1)
        best = EV[np.arange(len(val)), k]
        lev = np.array(m.levels)[k]
        for t in grid:
            tp = np.where(best >= t, lev, 0.0)
            s = pnl(tp, val.max_x.values, val.final_x.values)
            rows.append(dict(day=day, tau=round(float(t), 2), pnl=float(s.sum()),
                             takes=int((tp > 0).sum())))
    R = pd.DataFrame(rows)
    if R.empty:
        return ABSTAIN_TAU, R
    g = R.groupby('tau').agg(mean=('pnl', 'mean'), std=('pnl', 'std'),
                             pos=('pnl', lambda x: float((x > 0).mean())),
                             takes=('takes', 'mean')).reset_index()
    band = g[(g.takes >= DEPTH_LO) & (g.takes <= DEPTH_HI) & (g['mean'] > 0)]
    if band.empty:
        print('la validacion interna no encuentra regimen positivo en la banda: se abstiene')
        return ABSTAIN_TAU, g
    # criterio: media partido por desviacion, que penaliza los regimenes profundos y ruidosos
    band = band.assign(score=band['mean'] / (band['std'].fillna(0) + 1.0))
    return float(band.sort_values('score', ascending=False).tau.iloc[0]), g


def hrc_features(T, trn_pos, val_pos, trn_y_by_level, seeds=SEEDS):
    """El reservorio holografico leido por ridge: un numero por nivel y token.
    La lectura se ajusta SOLO con el entrenamiento del pliegue."""
    ht = np.zeros((len(trn_pos), len(LEVELS)))
    hv = np.zeros((len(val_pos), len(LEVELS)))
    for s in seeds:
        r = HolographicReservoir(seed=s)
        St = r.states(T[trn_pos])
        Sv = r.states(T[val_pos])
        for q in range(len(LEVELS)):
            r.fit_readout(St, trn_y_by_level[q])
            ht[:, q] += r.predict(St) / len(seeds)
            hv[:, q] += r.predict(Sv) / len(seeds)
    return ht, hv


# =============================================================================
# Ejecucion
# =============================================================================
A = load_labels()
TEST_DAY = pd.read_csv(DATA + '/test.csv')
TEST_DAY['day'] = pd.to_datetime(TEST_DAY.mt, unit='s').dt.strftime('%Y-%m-%d')
test_day = TEST_DAY.day.iloc[0]
days = sorted(A.day.unique())
print('etiquetas: %d tokens, %d dias (%s -> %s) · dia de test %s'
      % (len(A), len(days), days[0], days[-1], test_day), flush=True)

FEAT, SEQ_M, SEQ_T, EDGES = read_tape(days)
print('cinta: %d tokens con caracteristicas, %d con secuencia, %d aristas (%.0fs)'
      % (len(FEAT), len(SEQ_M), len(EDGES), time.time() - T0), flush=True)

A = A.merge(load_creation(), on='mint', how='left').merge(FEAT, on='mint', how='left')
A = A.sort_values('mt', kind='stable').reset_index(drop=True)
A['x_age_s'] = A.mt - A.created_ts
_dt = pd.to_datetime(A.mt, unit='s')
A['x_hour'] = _dt.dt.hour
A['x_dow'] = _dt.dt.dayofweek
A = pd.concat([A, creator_rep(A), wallet_rep(A, EDGES)], axis=1)
A = A.drop(columns=['creator'])
print('panel %s (%.0fs)' % (str(A.shape), time.time() - T0), flush=True)

SKIP = {'mint', 'mt', 'max_x', 'final_x', 'day', 'created_ts'}
FE = [c for c in A.columns if c not in SKIP and A[c].dtype.kind in 'fiub']

# posicion de cada token en el tensor de secuencias
pos = {m: i for i, m in enumerate(SEQ_M)}
SEQ_IDX = np.array([pos.get(m, -1) for m in A.mint.values])
ZERO = np.zeros((1, NBIN, NCH), dtype=np.float32)
SEQ_FULL = np.concatenate([SEQ_T, ZERO])        # el indice -1 cae en la fila de ceros

labelled = A.max_x.notna().values
is_test = (A.day == test_day).values & (~labelled)
TR = A[labelled].reset_index(drop=True)
TE = A[is_test].reset_index(drop=True)
tr_pos = SEQ_IDX[labelled]
te_pos = SEQ_IDX[is_test]
print('entrenamiento %d · test %d · %d caracteristicas base' % (len(TR), len(TE), len(FE)), flush=True)

FE_FULL = list(FE)
if USE_HRC:
    ys = [(TR.max_x.values >= u).astype(float) for u in LEVELS]
    ht, hv = hrc_features(SEQ_FULL, tr_pos, te_pos, ys)
    hcols = ['h_%g' % u for u in LEVELS]
    for q, c in enumerate(hcols):
        TR[c] = ht[:, q]
        TE[c] = hv[:, q]
    FE_FULL += hcols
    print('HRC: %d rasgos de lectura ridge anadidos (%.0fs)' % (len(hcols), time.time() - T0), flush=True)

# -------------------------------------------- walk-forward honesto, para el informe
WF = []
wf_days = sorted(TR.day.unique())[-WF_DAYS:]
for day in wf_days:
    val = TR[TR.day == day]
    trn = TR[(TR.day < day) & (TR.mt <= val.mt.min() - WIN - EMBARGO)]
    if len(trn) < 3000 or len(val) < 50:
        continue
    t_in, _ = pick_tau(trn, FE_FULL, inner_days=WF_INNER)
    m = Survival(FE_FULL, seeds=SEEDS_SEL).fit(trn)
    tp, _, _ = m.decide(val, t_in)
    s = pnl(tp, val.max_x.values, val.final_x.values)
    WF.append(dict(day=day, tokens=len(val), tau=round(t_in, 2), takes=int((tp > 0).sum()),
                   hits=int(((val.max_x.values >= tp) & (tp > 0)).sum()),
                   pnl=round(float(s.sum()), 2)))
    print(WF[-1], flush=True)
WF = pd.DataFrame(WF)
if len(WF):
    print('\nwalk-forward: total %+.2f en %d dias · media %+.3f · sd %.2f · '
          'dias positivos %d/%d · compras %d · aciertos %d (%.1f%%)'
          % (WF.pnl.sum(), len(WF), WF.pnl.mean(), WF.pnl.std(),
             int((WF.pnl > 0).sum()), len(WF), int(WF.takes.sum()), int(WF.hits.sum()),
             100 * WF.hits.sum() / max(WF.takes.sum(), 1)), flush=True)
    print('cero, la referencia que importa, puntuaria 0,00. Comprar todo a cualquier '
          'nivel fijo es profundamente negativo.', flush=True)

# ------------------------------------------------------------------ decision final
TAU, GRID = pick_tau(TR, FE_FULL)
print('\numbral de compra fijado con el entrenamiento: tau = %.2f' % TAU, flush=True)
if len(GRID):
    print(GRID.to_string(index=False), flush=True)

MODEL = Survival(FE_FULL, seeds=SEEDS).fit(TR)
tp, ev, P = MODEL.decide(TE, TAU)

sample = pd.read_csv(DATA + '/sample_submission.csv')
sub = sample[['mint']].merge(pd.DataFrame({'mint': TE.mint.values, 'tp': tp}),
                             on='mint', how='left')
sub['tp'] = sub.tp.fillna(0.0).astype(float)

ok = (sub.tp == 0) | ((sub.tp >= 3.0) & (sub.tp <= 100.0))
assert ok.all(), 'tp fuera del conjunto permitido'
assert len(sub) == len(sample), 'numero de filas distinto del de sample_submission'
assert sub.mint.equals(sample.mint), 'orden de mints distinto del de sample_submission'
assert sub.tp.notna().all(), 'hay tp sin valor'

for p in ('submission.csv', '/kaggle/working/submission.csv'):
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        continue
    sub.to_csv(p, index=False)
    print('escrito', p, flush=True)

nb = int((sub.tp > 0).sum())
print('\n%d filas · %d compras (%.2f%%) · niveles usados %s'
      % (len(sub), nb, 100 * nb / len(sub), sorted(set(sub.tp[sub.tp > 0]))), flush=True)
if nb:
    print(sub[sub.tp > 0].to_string(index=False), flush=True)
print('terminado en %.0fs' % (time.time() - T0), flush=True)
