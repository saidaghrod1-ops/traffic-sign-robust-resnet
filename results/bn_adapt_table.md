_Précision moyenne sur les niveaux ; statistiques BN réestimées sur 2048 images dégradées non étiquetées (généré par `src/bn_adapt.py`)._

| Dégradation | baseline | baseline + BN | robust | robust + BN |
|---|---|---|---|---|
| clean | 99.1 % | 99.0 % | 98.8 % | 98.9 % |
| gaussian_noise | 72.8 % | 88.1 % | 88.3 % | 92.6 % |
| gaussian_blur | 98.2 % | 98.7 % | 98.3 % | 98.7 % |
| motion_blur | 93.9 % | 94.4 % | 94.5 % | 95.2 % |
| fog | 72.7 % | 80.0 % | 89.9 % | 95.4 % |
| rain | 52.8 % | 69.9 % | 86.5 % | 90.1 % |
| snow | 61.9 % | 78.5 % | 88.8 % | 93.1 % |
| dark | 82.1 % | 84.5 % | 82.0 % | 94.8 % |
| overexposure | 88.3 % | 88.1 % | 90.0 % | 92.5 % |
| occlusion | 79.9 % | 82.9 % | 91.0 % | 91.4 % |
| moyenne dégradée | 78.1 % | 85.0 % | 89.9 % | 93.7 % |