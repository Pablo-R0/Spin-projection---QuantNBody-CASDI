# Pipeline projection sur spins locaux

Version compilee et nettoyee de la pipeline. **Deux fichiers** :

| Fichier | Role |
|---|---|
| `spin_projection.py` | Tout le code utile : extraction des determinants + analyse de spin local (`LocalSpinAnalyzer`). |
| `Spin_projection_pipeline.ipynb` | Notebook de bout en bout : ESCASOUT en entree -> tableau d'analyse en sortie. |

## Pipeline

```
ESCASOUT  ->  extraction (tous les vecteurs, tout Ms)  ->  reconstruction des Psi
          ->  <S2>, <Sz>, <S2_fragment> + poids des spins locaux  ->  DataFrame
```

L'utilisateur fournit un fichier ESCASOUT et remplit la cellule *Configuration* :
- les **orbitales** de chaque fragment (metal, ligand...) ;
- les **etats de spin** a projeter pour chaque fragment ;
- le **nombre d'electrons** (ou `None` pour detection automatique).

`Ms`, le nombre d'electrons et la liste des orbitales sont **detectes
automatiquement** et rapportes. `Ms` n'a pas besoin d'etre fourni : l'extraction
le gere quel que soit sa valeur.

## Exemple (sans notebook)

```python
from spin_projection import LocalSpinAnalyzer

analyzer = LocalSpinAnalyzer(
    n_electrons=6,
    fragments={
        "metal":  {"orbitals": ['106','107','108','109'], "spins": ['singlet','triplet']},
        "ligand": {"orbitals": ['110','111'],             "spins": ['singlet','triplet']},
    },
)
results = analyzer.analyze_file("example_datas/ESCASOUT3.ms0")   # une ligne par vecteur
print(results)
```

## Adapter a un autre calcul (p.ex. CAS(7,7))

Ne changer que la configuration. Spins disponibles : `singlet` (0), `doublet`
(0.75), `triplet` (2), `quartet` (3.75), `quintet` (6), `sextet` (8.75),
`septet` (12) — ou une valeur numerique `S(S+1)` directe.

```python
fragments = {
    "metal":  {"orbitals": ['1','2','3','4','5'], "spins": ['doublet','quartet']},
    "ligand": {"orbitals": ['6','7'],             "spins": ['singlet','triplet']},
}
analyzer = LocalSpinAnalyzer(n_electrons=7, fragments=fragments)
```

---

# Comment installer QuantNBody
```
#Install conda
#https://conda.io/projects/conda/en/latest/user-guide/install/macos.html

#Create a conda environment
conda create -n quant
conda activate quant

conda install -c psi4 psi4
conda install pip numba notebook pandas

git clone https://github.com/SYalouz/QuantNBody.git
cd QuantNBody
python -m pip install -e .
```
