# Feature and random-reservoir functions preserved from the prepared solution.
# Francisco Angulo de Lafuente. Holographic-Reservoir original: Apache-2.0.
# No pretrained parameters; fit_readout is retained for provenance but NEVER USED.
import hashlib
import numpy as np
import pandas as pd
WSOL='So11111111111111111111111111111111111111112'
WIN=3600
NBIN,BINW,NCH=20,3,7
HRC_SIZE=48


def gini(v):
    v = np.sort(np.asarray(v, float))
    n = len(v)
    s = v.sum()
    if n == 0 or s <= 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * s))


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
    if not a.ts_rel.between(0,60,inclusive='both').all():
        raise ValueError('AMM sequence contains a trade outside the visible first minute')
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
            m = T[:, t, 2] == 0
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
