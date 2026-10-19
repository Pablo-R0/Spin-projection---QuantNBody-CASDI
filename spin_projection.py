"""
spin_projection
===============

Pipeline complet : fichier ESCASOUT (CAS(n,m)SCF) -> analyse de spin local de
TOUS les vecteurs propres (spin total + spin de chaque fragment, p.ex. metal et
ligand), via QuantNBody.

    ESCASOUT  ->  extraction des determinants  ->  reconstruction des Psi
              ->  <S^2>, <S_z>, <S^2_fragment> et poids des etats de spin locaux

Ce module regroupe deux parties :

1. EXTRACTION
   Lecture du format `&ess ic=...,v(k)=coeff /` (positif = alpha, negatif = beta,
   ranges « tous les alpha puis tous les beta »), conversion en bitstrings dans
   la convention orbital-major de QuantNBody (`MO0a MO0b MO1a MO1b ...`) avec le
   signe de permutation applique au coefficient. Detection automatique des
   orbitales actives, du nombre d'electrons et de Ms.

2. ANALYSE (classe `LocalSpinAnalyzer`)
   Construction de la base a N corps et des operateurs de spin (totaux + locaux),
   puis projection de chaque vecteur sur les sous-espaces de spin demandes.

Une seule fonction d'extraction gere n'importe quel Ms et n'importe quelle taille
de CAS : pas besoin de specifier Ms (il est detecte et seulement rapporte).

Exemple minimal
---------------
    from spin_projection import LocalSpinAnalyzer
    analyzer = LocalSpinAnalyzer(
        n_electrons=6,
        fragments={
            "metal":  {"orbitals": ['106','107','108','109'], "spins": ['singlet','triplet']},
            "ligand": {"orbitals": ['110','111'],             "spins": ['singlet','triplet']},
        },
    )
    results = analyzer.analyze_file("ESCASOUT3.ms0")   # DataFrame: une ligne par vecteur
"""

import re
import math
import numpy as np
import pandas as pd
import quantnbody as qnb


# --------------------------------------------------------------------------- #
#  Etats de spin : etiquette -> valeur de S^2 = S(S+1)                         #
# --------------------------------------------------------------------------- #
SPIN_LABEL_TO_S2 = {
    "singlet": 0.0,    # S = 0
    "doublet": 0.75,   # S = 1/2
    "triplet": 2.0,    # S = 1
    "quartet": 3.75,   # S = 3/2
    "quintet": 6.0,    # S = 2
    "sextet": 8.75,    # S = 5/2
    "septet": 12.0,    # S = 3
}


def spin_to_s2(spin):
    """Convertit une etiquette ('triplet') ou une valeur numerique en S(S+1)."""
    if isinstance(spin, str):
        return SPIN_LABEL_TO_S2[spin.lower()]
    return float(spin)


# ===========================================================================
#  1.  EXTRACTION DES DETERMINANTS
# ===========================================================================

# Capture : la liste d'orbitales (det), le numero de vecteur, le coefficient.
_LINE_RE = re.compile(r"ic=(?P<det>[-\d,]+),v\((?P<vec>\d+)\)=(?P<coeff>\S+)")


def _iter_records(file_name):
    """Itere sur les lignes valides : (tokens, vec_num, coeff)."""
    with open(file_name, "r") as handle:
        for line in handle:
            m = _LINE_RE.search(line)
            if m is not None:
                yield m.group("det").split(","), int(m.group("vec")), float(m.group("coeff"))


def detect_orbitals(file_name):
    """Liste triee des etiquettes d'orbitales actives presentes dans le fichier."""
    labels = set()
    for tokens, _, _ in _iter_records(file_name):
        for tok in tokens:
            labels.add(tok.lstrip("-"))
    return sorted(labels, key=int)


def detect_n_electrons(file_name):
    """Nombre d'electrons actifs (= nombre d'orbitales par determinant)."""
    for tokens, _, _ in _iter_records(file_name):
        return len(tokens)
    raise ValueError(f"Aucun determinant trouve dans {file_name}")


def detect_ms(file_name):
    """Ms = (n_alpha - n_beta) / 2, lu sur le premier determinant."""
    for tokens, _, _ in _iter_records(file_name):
        n_beta = sum(t.startswith("-") for t in tokens)
        n_alpha = len(tokens) - n_beta
        return (n_alpha - n_beta) / 2
    raise ValueError(f"Aucun determinant trouve dans {file_name}")


def _spin_orbital_index(token, orbitals):
    """Indice du bit QuantNBody (orbital-major) : ``2*idx_orbitale + (1 si beta)``."""
    beta = token.startswith("-")
    label = token.lstrip("-")
    return 2 * orbitals.index(label) + (1 if beta else 0)


def _permutation_sign(indices):
    """Parite (0 = '+'/pair, 1 = '-'/impair) du nombre d'inversions de ``indices``.

    C'est le signe a appliquer pour reordonner les operateurs de creation du
    fichier (tous-alpha-puis-tous-beta) vers l'ordre orbital-major de QuantNBody.
    """
    inversions = 0
    n = len(indices)
    for i in range(n):
        for j in range(i + 1, n):
            if indices[i] > indices[j]:
                inversions += 1
    return inversions % 2


def extract_vector(file_name, vec_num, orbitals):
    """DataFrame (det bitstring, coeff signe) du vecteur ``vec_num``, tout Ms.

    Colonnes : ``det`` (liste de bits QuantNBody) et ``coeff`` (avec signe de
    permutation deja applique).
    """
    n_spin_orb = 2 * len(orbitals)
    rows = []
    for tokens, vec, coeff in _iter_records(file_name):
        if vec != vec_num:
            continue
        indices = [_spin_orbital_index(t, orbitals) for t in tokens]
        bitstring = [0] * n_spin_orb
        for idx in indices:
            bitstring[idx] = 1
        if _permutation_sign(indices):
            coeff = -coeff
        rows.append({"det": bitstring, "coeff": coeff})
    return pd.DataFrame(rows, columns=["det", "coeff"])


def extract_all_vectors(file_name, orbitals):
    """{vec_num: DataFrame} pour tous les vecteurs du fichier."""
    vec_numbers = sorted({vec for _, vec, _ in _iter_records(file_name)})
    return {v: extract_vector(file_name, v, orbitals) for v in vec_numbers}


# ===========================================================================
#  2.  ANALYSE DE SPIN LOCAL
# ===========================================================================

class LocalSpinAnalyzer:
    """Analyse de spin local de tous les vecteurs d'un fichier ESCASOUT.

    Parameters
    ----------
    n_electrons : int
        Nombre d'electrons actifs du CAS.
    fragments : dict
        ``{nom: {"orbitals": [labels], "spins": [etiquettes ou valeurs S(S+1)]}}``
        decrivant chaque fragment (p.ex. ``"metal"`` et ``"ligand"``) et les
        etats de spin a projeter pour chacun.
    orbitals : list[str], optionnel
        Ordre des etiquettes d'orbitales actives. Par defaut : union triee des
        orbitales des fragments. Doit etre coherent avec celui de l'extraction.
    frozen_orbitals, virtual_orbitals : list[str], optionnel
        Orbitales toujours doublement occupees / jamais occupees. Si vides, le
        projecteur CAS vaut l'identite (rien n'est applique).
    """

    def __init__(self, n_electrons, fragments, orbitals=None,
                 frozen_orbitals=None, virtual_orbitals=None):
        # --- Orbitales et indices 0-based -------------------------------
        if orbitals is None:
            union = set()
            for frag in fragments.values():
                union.update(str(o) for o in frag["orbitals"])
            orbitals = sorted(union, key=int)
        self.orbitals = [str(o) for o in orbitals]
        self.n_mo = len(self.orbitals)
        self.n_electrons = n_electrons

        def to_idx(labels):
            return [self.orbitals.index(str(o)) for o in labels]

        self.frozen_indices = to_idx(frozen_orbitals or [])
        self.virtual_indices = to_idx(virtual_orbitals or [])

        # --- Base a N corps + operateur a_dagger_a ----------------------
        self.dim_H = math.comb(2 * self.n_mo, n_electrons)
        self.nbody_basis = qnb.fermionic.tools.build_nbody_basis(self.n_mo, n_electrons)
        self.a_dagger_a = qnb.fermionic.tools.build_operator_a_dagger_a(self.nbody_basis)

        # --- Operateurs de spin total -----------------------------------
        self.S2, self.Sz, _ = qnb.fermionic.tools.build_s2_sz_splus_operator(self.a_dagger_a)

        # --- Operateurs et projecteurs de spin local par fragment -------
        self.fragments = {}
        for name, frag in fragments.items():
            mo = to_idx(frag["orbitals"])
            S2_loc, _, _ = qnb.fermionic.tools.build_local_s2_sz_splus_operator(self.a_dagger_a, mo)
            proj = {
                str(spin): qnb.fermionic.tools.build_spin_subspaces(S2_loc, spin_to_s2(spin))
                for spin in frag["spins"]
            }
            self.fragments[name] = {"S2": S2_loc, "proj": proj}

        # --- Projecteur CAS (None = identite) ---------------------------
        self.Proj_CAS = self._build_proj_cas()

    # ------------------------------------------------------------------ #
    def _build_proj_cas(self):
        if not self.frozen_indices and not self.virtual_indices:
            return None
        state_frozen = np.zeros(2 * self.n_mo, dtype=np.int32)
        state_virtual = np.zeros(2 * self.n_mo, dtype=np.int32)
        for i in range(self.n_mo):
            if i in self.frozen_indices:
                state_frozen[2 * i] = state_frozen[2 * i + 1] = 1
            if i in self.virtual_indices:
                state_virtual[2 * i] = state_virtual[2 * i + 1] = 1
        proj = np.zeros((self.dim_H, self.dim_H))
        for state in self.nbody_basis:
            if (state_frozen @ state == 2 * len(self.frozen_indices)
                    and state_virtual @ state == 0):
                fstate = qnb.fermionic.tools.my_state(state, self.nbody_basis)
                proj += np.outer(fstate, fstate)
        return proj

    # ------------------------------------------------------------------ #
    def psi_from_df(self, df):
        """Reconstruit la fonction d'onde normalisee a partir d'un DataFrame."""
        coeffs = df["coeff"].to_numpy()
        norm = np.sqrt(np.sum(coeffs ** 2))
        psi = np.zeros(self.dim_H)
        for det, coeff in zip(df["det"], coeffs):
            psi += (coeff / norm) * qnb.fermionic.tools.my_state(det, self.nbody_basis)
        if self.Proj_CAS is not None:
            psi = self.Proj_CAS @ psi
        return psi

    def analyze_psi(self, psi):
        """Valeurs moyennes de spin total + spin local par fragment."""
        row = {
            "S2_total": float(psi.T @ self.S2 @ psi),
            "Sz": float(psi.T @ self.Sz @ psi),
        }
        for name, frag in self.fragments.items():
            row[f"S2_{name}"] = float(psi.T @ frag["S2"] @ psi)
            for spin, proj in frag["proj"].items():
                row[f"{name}_{spin}"] = float(psi.T @ proj @ psi)
        return row

    def analyze_file(self, file_name, round_to=3):
        """Analyse TOUS les vecteurs du fichier. Renvoie un DataFrame indexe par vecteur.

        Les determinants sont extraits avec l'ordre d'orbitales de l'analyseur,
        donc Ms est gere automatiquement.
        """
        vectors = extract_all_vectors(file_name, self.orbitals)
        results = {f"v{vec}": self.analyze_psi(self.psi_from_df(df))
                   for vec, df in vectors.items()}
        out = pd.DataFrame(results).T
        return out.round(round_to) if round_to is not None else out

    def visualize_vector(self, file_name, vec_num, cutoff=0.1, atomic_orbitals=True):
        """Affiche les determinants dominants d'un vecteur (debug/inspection)."""
        df = extract_vector(file_name, vec_num, self.orbitals)
        psi = self.psi_from_df(df)
        qnb.fermionic.tools.visualize_wft(
            psi, self.nbody_basis, cutoff=cutoff, atomic_orbitals=atomic_orbitals)
        return psi
