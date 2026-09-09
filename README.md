# FakeNewsPropagation

Script e utility per raccolta e analisi dati Telegram.

## Configurazione

Le configurazioni sono state unificate in un solo file:

- config/config.json
- config/config.example.json

La cartella var e stata rinominata in config.

Il file include:

- credenziali Telegram (api_id, api_hash, session_name)
- keyword e filtri temporali per message collection
- seed channels e parametri snowball
- path output CSV

config/config.example.json e il template versionato.
config/config.json e il file locale usato dagli script.

## Snowball Sampling

Script: src/snowball_sampling.py

Esecuzione:

python src/snowball_sampling.py --config config/config.json

Output:

- data_collected/snowball_channels.csv
- data_collected/snowball_relations.csv

## Data Collection

Script: src/message_collection.py

Esecuzione:

python src/message_collection.py

Note:
- Nessun argomento CLI: tutto e gestito in config/config.json
- La lista canali viene letta dal CSV prodotto da snowball_sampling

## Esempio Config JSON

Vedi config/config.example.json per la struttura completa.

## Graph Analysis

Script: src/graph_analysis.py

Esecuzione:

python src/graph_analysis.py

Input:
- Usa il file relazionale prodotto da snowball_sampling (snowball_relations.csv)

Output:
- Matrice relazioni source->target in CSV
- Grafo in formato GEXF
- Preview del grafo in PNG
