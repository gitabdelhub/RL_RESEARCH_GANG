# RL_RESEARCH_GANG

Projet de recherche en Reinforcement Learning (RL)

## Description
Ce dépôt contient le code, les scripts d'expériences, et les résultats pour des études sur l'apprentissage par renforcement appliqué à la modélisation de politiques de vaccination et d'autres scénarios robustes.

## Structure du projet

- `src/` : Code source principal
  - `models/` : Modèles de patients et environnements
  - `policies/` : Politiques RL et recherches robustes
  - `rl/` : Agents RL et environnements
  - `sim/` : Simulation de scénarios
  - `report/` : Génération de rapports
- `experiments/` : Scripts pour lancer les expériences, tests, figures, etc.
- `results/` : Résultats des expériences (figures, tables, configurations)
- `requirements.txt` : Dépendances Python
- `create_pdf.py` : Script pour générer un rapport PDF
- `rapport_exercice2.txt` : Rapport d'exercice

## Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/gitabdelhub/RL_RESEARCH_GANG.git
   cd RL_RESEARCH_GANG
   ```
2. Créer un environnement virtuel et installer les dépendances :
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   pip install -r requirements.txt
   ```

## Utilisation

- Lancer des expériences :
  ```bash
  python experiments/train_rl_agent.py
  python experiments/run_baselines.py
  # etc.
  ```
- Générer des figures :
  ```bash
  python experiments/generate_rl_figures.py
  ```
- Générer un rapport PDF :
  ```bash
  python create_pdf.py
  ```

## Organisation des dossiers

- `src/models/` : Modèles de patients, environnements simulés
- `src/policies/` : Politiques de base et robustes
- `src/rl/` : Agents RL (DQN, etc.), environnements personnalisés
- `src/sim/` : Outils de simulation
- `src/report/` : Génération de rapports
- `experiments/` : Scripts d’expérimentation et d’analyse
- `results/` : Figures, tables, résultats d’expériences

## Auteurs
- Abdel
- Collaborateurs : ...

## Licence
Ce projet est sous licence MIT.
