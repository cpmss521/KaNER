

import json

path_AnatEM = "../DataSets/KaNER/query/AnatEM.json"
path_BC5CDR = "../DataSets/KaNER/query/BC5CDR.json"
path_BioNLP09 = "../DataSets/KaNER/query/BioNLP09.json"
path_BioNLP11EPI = "../DataSets/KaNER/query/BioNLP11EPI.json"
path_BioNLP11ID = "../DataSets/KaNER/query/BioNLP11ID.json"
path_BioNLP13GE = "../DataSets/KaNER/query/BioNLP13GE.json"
path_Ex_PTM = "../DataSets/KaNER/query/Ex_PTM.json"
path_GENIA = "../DataSets/KaNER/query/genia.json"
path_JNLPBA = "../DataSets/KaNER/query/JNLPBA.json"
path_NCBI = "../DataSets/KaNER/query/NCBI.json"



def load_query_map(query_map_path):
    with open(query_map_path, "r") as f:
        query_map = json.load(f)
    return query_map

query_AnatEM  = load_query_map(path_AnatEM)
query_BC5CDR = load_query_map(path_BC5CDR)
query_BioNLP09 = load_query_map(path_BioNLP09)
query_BioNLP11EPI = load_query_map(path_BioNLP11EPI)
query_BioNLP11ID = load_query_map(path_BioNLP11ID)
query_BioNLP13GE = load_query_map(path_BioNLP13GE)
query_Ex_PTM = load_query_map(path_Ex_PTM)
query_genia = load_query_map(path_GENIA)
query_jnlpba = load_query_map(path_JNLPBA)
query_ncbi = load_query_map(path_NCBI)





queries_for_dataset = {
    "AnatEM":query_AnatEM,
    "BC5CDR": query_BC5CDR,
    "BioNLP09":query_BioNLP09,
    "BioNLP11EPI":query_BioNLP11EPI,
    "BioNLP11ID":query_BioNLP11ID,
    "BioNLP13GE":query_BioNLP13GE,
    "Ex_PTM":query_Ex_PTM,
    "GENIA": query_genia,
    "JNLPBA": query_jnlpba,
    "NCBI": query_ncbi,
}
