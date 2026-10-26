_Généré par `src/evaluate.py` le 2026-10-26 11:29 — GTSRB test, 12630 images, 9 dégradations x 5 niveaux. CE et mCE relatifs à `baseline` (100 %)._

#### Résumé

| Modèle | Acc. propre | Acc. moy. dégradée | mCE | mCE (vues) | mCE (non vues) | ECE propre | ECE dégradée |
|---|---|---|---|---|---|---|---|
| baseline | 99.1 % | 78.1 % | 100.0 % | 100.0 % | 100.0 % | 10.6 % | 10.6 % |
| robust | 98.9 % | 89.9 % | 61.7 % | 62.0 % | 60.4 % | 10.4 % | 9.4 % |

#### Par type de dégradation (précision moyenne sur les niveaux)

| Dégradation | Vue par robust | Acc baseline | Acc robust | CE robust |
|---|---|---|---|---|
| gaussian_noise | oui | 72.8 % | 88.3 % | 42.8 % |
| gaussian_blur | oui | 98.2 % | 98.3 % | 95.1 % |
| motion_blur | non | 93.9 % | 94.5 % | 91.4 % |
| fog | oui | 72.7 % | 89.9 % | 36.9 % |
| rain | oui | 52.7 % | 86.5 % | 28.6 % |
| snow | non | 61.9 % | 88.8 % | 29.4 % |
| dark | oui | 82.1 % | 82.0 % | 100.6 % |
| overexposure | oui | 88.3 % | 90.0 % | 85.3 % |
| occlusion | oui | 79.9 % | 91.0 % | 44.9 % |

#### Précision par niveau

**baseline** (propre : 99.1 %)

| Dégradation | niv. 1 | niv. 2 | niv. 3 | niv. 4 | niv. 5 |
|---|---|---|---|---|---|
| gaussian_noise | 97.6 | 85.6 | 72.7 | 60.2 | 47.7 |
| gaussian_blur | 99.0 | 98.9 | 98.8 | 98.1 | 96.3 |
| motion_blur | 99.0 | 98.7 | 97.2 | 92.4 | 82.4 |
| fog | 94.9 | 86.5 | 75.6 | 61.4 | 45.3 |
| rain | 87.2 | 68.9 | 49.5 | 35.4 | 22.7 |
| snow | 90.1 | 72.3 | 58.4 | 48.6 | 40.4 |
| dark | 98.6 | 92.8 | 83.0 | 73.0 | 62.9 |
| overexposure | 97.9 | 95.5 | 91.0 | 83.8 | 73.3 |
| occlusion | 96.9 | 93.7 | 85.3 | 71.1 | 52.3 |

**robust** (propre : 98.9 %)

| Dégradation | niv. 1 | niv. 2 | niv. 3 | niv. 4 | niv. 5 |
|---|---|---|---|---|---|
| gaussian_noise | 98.1 | 95.8 | 91.0 | 83.4 | 73.5 |
| gaussian_blur | 98.8 | 98.7 | 98.6 | 98.2 | 97.4 |
| motion_blur | 98.7 | 98.3 | 97.2 | 92.6 | 85.6 |
| fog | 98.4 | 96.9 | 93.7 | 86.4 | 74.4 |
| rain | 97.5 | 95.7 | 89.9 | 80.6 | 68.7 |
| snow | 97.8 | 95.7 | 91.2 | 84.6 | 74.8 |
| dark | 98.3 | 93.1 | 83.5 | 72.8 | 62.1 |
| overexposure | 97.8 | 96.2 | 92.4 | 86.5 | 77.2 |
| occlusion | 98.2 | 97.0 | 93.9 | 88.0 | 77.7 |
