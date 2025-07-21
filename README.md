# COCO Implementation

## Requirements

- Python 3.7
- PyTorch 1.12
- psycopg2
- Ray 2.4.0

## Datasets

We build the training databases following the instructions from the repositories [zero-shot-cost-estimation](https://github.com/DataManagementLab/zero-shot-cost-estimation) and [PRICE](https://github.com/StCarmen/PRICE). The resulting collection includes the following schemas:

```text
['basketball', 'walmart', 'financial', 'movielens',
 'carcinogenesis', 'accidents', 'tournament',
 'employee', 'geneea', 'genome', 'seznam', 'fhnk', 'consumer',
 'ssb', 'hepatitis', 'credit', 'baseball', 'tpch', 'stats', 'imdb',
 'chembl', 'ergastf1', 'grants', 'legalacts', 'sap', 'talkingdata']
```

## Feature Preparation

Extracting features for COCO can take some time. Run the following scripts in order:

1. `python ./features/meta_info.py` – collect basic metadata.
2. `python ./features/column_data_distribution.py` – compute column-wise data-distribution statistics.
3. `python ./features/merge.py` – merge the features obtained in the previous steps.
4. `python ./features/get_column_stats.py` – sample data used by SQLGen for query generation.

## Running COCO

First, modify the configuration of PostgreSQL in GenConfig/gen_config.py and then execute the test workload to obtain the test set:

```bash
python ./TestWorkload/runSQL.py
```

Then start COCO training and roll-out evaluation:

```bash
python main.py
```
