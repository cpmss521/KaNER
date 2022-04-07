
import json
import joblib
from query_map import queries_for_dataset

def generate_query_ner_dataset(source_file_path, save_file_path, dataset_name=None, query_sign="default"):

    entity_queries = queries_for_dataset[dataset_name][query_sign]
    label_list = queries_for_dataset[dataset_name]["labels"]


    target_data = transform_examples_to_qa_features(entity_queries, label_list, source_file_path)

    with open(save_file_path, "w") as f:
        json.dump(target_data, f, sort_keys=True, ensure_ascii=False, indent=2)

def transform_examples_to_qa_features(query_map, entity_labels, org_data_file):
    mrc_ner_dataset = []

    data_sents, data_records, evidences, cui_, sem_type_, description = joblib.load(org_data_file)



    tmp_qas_id = 0
    for data_sent, data_record, data_evidence, data_cui, data_semType, data_description in zip(data_sents, data_records,
                                                                                       evidences, cui_, sem_type_,
                                                                                       description):

        tmp_query_id = 0
        for label_idx, tmp_label in enumerate(entity_labels):

            tmp_query_id += 1
            tmp_query = query_map[tmp_label]
            tmp_context = " ".join(data_sent)
            tmp_evidence = " ".join(data_evidence)
            tmp_description = " ".join(data_description)

            tmp_start_pos = []
            tmp_end_pos = []
            tmp_entity_pos = []

            start_end_label = [(item_label[0], item_label[1]) for item_label in data_record if
                               data_record[item_label] == tmp_label]

            if len(start_end_label) != 0:
                for span_item in start_end_label:
                    start_idx, end_idx = span_item
                    tmp_start_pos.append(start_idx)
                    tmp_end_pos.append(end_idx) ## end position not in original position but Subtraction 1
                    tmp_entity_pos.append("{};{}".format(str(start_idx), str(end_idx)))
                tmp_impossible = False
            else:
                tmp_impossible = True

            mrc_ner_dataset.append({
                "qas_id": "{}.{}".format(str(tmp_qas_id), str(tmp_query_id)),
                "query": tmp_query,
                "context": tmp_context,
                "evidence": tmp_evidence,
                "description": tmp_description,
                "CUI":' '.join(data_cui),
                "SemanticType":' '.join(data_semType),
                "entity_label": tmp_label,
                "start_position": tmp_start_pos,
                "end_position": tmp_end_pos,
                "span_position": tmp_entity_pos,
                "impossible": tmp_impossible
                })

        tmp_qas_id += 1



    return mrc_ner_dataset



if __name__ == '__main__':

    dataset_name = "Ex_PTM"

    source_train_data_path = "../DataSets/KaNER/{0}/train_know_evidence_description1.pkl".format(dataset_name)
    target_train_data_path = "../DataSets/KaNER/{0}/{1}_train_MRC.json".format(dataset_name,dataset_name)

    source_dev_data_path = "../DataSets/KaNER/{0}/dev_know_evidence_description1.pkl".format(dataset_name)
    target_dev_data_path = "../DataSets/KaNER/{0}/{1}_dev_MRC.json".format(dataset_name,dataset_name)


    source_test_data_path = "../DataSets/KaNER/{0}/test_know_evidence_description1.pkl".format(dataset_name)
    target_test_data_path = "../DataSets/KaNER/{0}/{1}_test_MRC.json".format(dataset_name,dataset_name)


    generate_query_ner_dataset(source_train_data_path, target_train_data_path, dataset_name=dataset_name,
                               query_sign="default")

    generate_query_ner_dataset(source_dev_data_path, target_dev_data_path, dataset_name=dataset_name,
                               query_sign="default")

    generate_query_ner_dataset(source_test_data_path, target_test_data_path, dataset_name=dataset_name,
                               query_sign="default")

