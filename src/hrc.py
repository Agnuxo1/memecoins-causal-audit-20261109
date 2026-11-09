# SPDX-FileCopyrightText: 2026 Francisco Angulo de Lafuente
# SPDX-License-Identifier: MIT
# Research-Project: memecoins-walk-forward-research
# Research-Date: 2026-11-04
# Research-ORCID: https://orcid.org/0009-0001-1634-7063
# Reuse: preserve applicable notices and cite the original author.

"""Reservorio holografico (HRC) portado en linea, sin dependencias externas.

Port del nucleo de D:/PROJECTS/3_Holographic-Reservoir (Apache-2.0, Francisco Angulo):
capa recurrente dispersa con fuga, tanh, radio espectral fijado y entropia determinista
derivada de SHA-256 inyectada en la diagonal (el backend simulado del original).
La caja de arena del concurso no tiene red, asi que no se puede instalar el paquete:
se porta el mismo principio a NumPy puro, vectorizado sobre todos los tokens a la vez.
Lectura: ridge (forma cerrada de la RLS del original con olvido 1).
"""
import hashlib
import numpy as np


def _sha_entropy(n, seed):
    out = []
    h = hashlib.sha256(f'hrc-init-{seed}'.encode()).digest()
    while len(out) < n:
        out.extend(h)
        h = hashlib.sha256(h).digest()
    return (np.frombuffer(bytes(out[:n]), dtype=np.uint8).astype(np.float64) / 127.5 - 1.0)


class HolographicReservoir:
    def __init__(self, size=48, input_dim=7, spectral_radius=0.9, leak=0.3,
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
        """U: (N, T, input_dim) -> rasgos del reservorio (N, 2*size+1).

        Estado final y estado medio: el final lleva el instante de la decision,
        el medio resume toda la trayectoria del primer minuto."""
        N, T, _ = U.shape
        X = np.zeros((N, self.size))
        acc = np.zeros((N, self.size))
        for t in range(T):
            pre = U[:, t, :] @ self.W_in.T + X @ self.W.T
            X = (1.0 - self.leak) * X + self.leak * np.tanh(pre)
            acc += X
        return np.hstack([X, acc / T, np.ones((N, 1))])

    def fit_readout(self, S, y):
        A = S.T @ S + self.ridge * np.eye(S.shape[1])
        self.W_out = np.linalg.solve(A, S.T @ y)
        return self

    def predict(self, S):
        return S @ self.W_out
