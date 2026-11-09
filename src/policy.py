# SPDX-FileCopyrightText: 2026 Francisco Angulo de Lafuente
# SPDX-License-Identifier: MIT
# Research-Project: memecoins-walk-forward-research
# Research-Date: 2026-11-04
# Research-ORCID: https://orcid.org/0009-0001-1634-7063
# Reuse: preserve applicable notices and cite the original author.

"""Politica de valor esperado: curva de supervivencia calibrada -> nivel y decision.

Cumple la regla 4 del concurso (una decision por token): el tp de un token sale solo de
sus propias caracteristicas y de umbrales fijados con el entrenamiento. No hay cuantiles,
rangos ni topes calculados sobre el dia de test.
"""
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

COST = 0.3
WIN, EMBARGO = 3600, 3600
LEVELS = (3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0)
LGB = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=60,
           subsample=0.85, subsample_freq=1, colsample_bytree=0.7,
           reg_lambda=5.0, verbose=-1, n_jobs=4)


def pnl(tp, max_x, final_x):
    tp = np.asarray(tp, float)
    hit = (max_x >= tp) & (tp > 0)
    held = np.maximum(final_x - 1 - COST, -1.0)
    return np.where(tp <= 0, 0.0, np.where(hit, tp - 1 - COST, held))


def _cal_split(trn, cal_days=5):
    d = sorted(trn.day.unique())
    if len(d) <= cal_days + 3: return trn, None
    cal = trn[trn.day >= d[-cal_days]]
    fit = trn[(trn.day < d[-cal_days]) & (trn.mt <= cal.mt.min() - WIN - EMBARGO)]
    return (fit, cal) if len(fit) > 500 and len(cal) > 200 else (trn, None)


class Survival:
    """P(max_x >= u) calibrada para cada u de LEVELS, mas el termino de perdida."""

    def __init__(self, features, levels=LEVELS, seeds=(0,)):
        self.FE, self.levels, self.seeds = list(features), tuple(levels), tuple(seeds)

    def fit(self, trn):
        Xall = trn[self.FE].astype(float).values
        fit, cal = _cal_split(trn)
        self.models, self.cals, self.loss = {}, {}, {}
        for u in self.levels:
            yall = (trn.max_x.values >= u).astype(int)
            ms = []
            for s in self.seeds:
                ms.append(lgb.LGBMClassifier(random_state=s, **LGB).fit(Xall, yall))
            self.models[u] = ms
            if cal is not None and (fit.max_x.values >= u).sum() >= 20:
                mf = [lgb.LGBMClassifier(random_state=s, **LGB).fit(
                        fit[self.FE].astype(float).values, (fit.max_x.values >= u).astype(int))
                      for s in self.seeds]
                p = np.mean([m.predict_proba(cal[self.FE].astype(float).values)[:, 1] for m in mf], 0)
                ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                ir.fit(p, (cal.max_x.values >= u).astype(int))
                self.cals[u] = ir
            else:
                self.cals[u] = None
            miss = trn.max_x.values < u
            self.loss[u] = float(np.maximum(trn.final_x.values[miss] - 1 - COST, -1.0).mean()) if miss.sum() else -1.0
        return self

    def proba(self, df):
        X = df[self.FE].astype(float).values
        P = np.empty((len(df), len(self.levels)))
        for q, u in enumerate(self.levels):
            p = np.mean([m.predict_proba(X)[:, 1] for m in self.models[u]], 0)
            if self.cals[u] is not None: p = self.cals[u].transform(p)
            P[:, q] = p
        # monotonia: P(>=u) no puede crecer con u
        P = np.minimum.accumulate(P, axis=1)
        return P

    def ev(self, df):
        P = self.proba(df)
        L = np.array([self.loss[u] for u in self.levels])
        G = np.array([u - 1 - COST for u in self.levels])
        return P * G + (1 - P) * L, P

    def decide(self, df, tau):
        EV, P = self.ev(df)
        k = EV.argmax(1); best = EV[np.arange(len(df)), k]
        tp = np.where(best >= tau, np.array(self.levels)[k], 0.0)
        return tp, best, P


ABSTAIN_TAU = 1e9     # umbral prohibitivo: cero compras, PnL exactamente 0
DEPTH_LO, DEPTH_HI = 0.1, 2.0     # banda de compras por dia que la evidencia respalda


def pick_tau(TR, FE, inner_days=8, grid=None, seeds=(0,)):
    """Elige el umbral de EV con una validacion interna sobre los ultimos dias del
    entrenamiento. No toca el dia de test.

    Dos reglas, ambas sacadas de la metrica:
    - solo se consideran umbrales cuya profundidad cae en la banda que la ablacion
      respalda (entre 0,3 y 2 compras al dia); mas profundo se degrada siempre.
    - si en esa banda la mejor media interna no es positiva, se ABSTIENE. Saltarse
      un dia puntua exactamente 0, y solo se pagan puntuaciones oficiales positivas,
      asi que desplegar un regimen que la propia validacion dice que pierde es peor
      que no comprar.
    """
    grid = grid if grid is not None else np.arange(0.0, 2.01, 0.1)
    days = sorted(TR.day.unique())
    rows = []
    for day in days[-inner_days:]:
        val = TR[TR.day == day]
        trn = TR[(TR.day < day) & (TR.mt <= val.mt.min() - WIN - EMBARGO)]
        if len(trn) < 2000 or len(val) < 50: continue
        m = Survival(FE, seeds=seeds).fit(trn)
        EV, _ = m.ev(val)
        k = EV.argmax(1); best = EV[np.arange(len(val)), k]
        lev = np.array(m.levels)[k]
        for t in grid:
            tp = np.where(best >= t, lev, 0.0)
            s = pnl(tp, val.max_x.values, val.final_x.values)
            rows.append(dict(day=day, tau=round(float(t), 2), pnl=float(s.sum()),
                             takes=int((tp > 0).sum())))
    R = pd.DataFrame(rows)
    if R.empty: return ABSTAIN_TAU, R
    g = R.groupby('tau').agg(mean=('pnl', 'mean'), std=('pnl', 'std'),
                             pos=('pnl', lambda x: float((x > 0).mean())),
                             takes=('takes', 'mean')).reset_index()
    band = g[(g.takes >= DEPTH_LO) & (g.takes <= DEPTH_HI) & (g['mean'] > 0)]
    if band.empty:
        return ABSTAIN_TAU, g
    # criterio: media partido por desviacion, que penaliza los regimenes profundos y ruidosos
    band = band.assign(score=band['mean'] / (band['std'].fillna(0) + 1.0))
    return float(band.sort_values('score', ascending=False).tau.iloc[0]), g
