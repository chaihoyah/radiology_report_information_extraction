#!/usr/bin/env python
# brain_mri_evaluation.py

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import copy
from ast import literal_eval
import pickle

def calculate_f1_score(precision, recall):
    """Calculate F1-score given precision and recall."""
    if precision + recall == 0:
        return 0.0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return round(f1_score, 2)

def main():
    parser = argparse.ArgumentParser(description="Final evaluation script for Brain MRI parsing.")
    
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the final Brain MRI parsing results CSV file.")
    parser.add_argument("--disease_confmat_output_csv", type=str, required=True,
                        help="Path to store disease parsing confusion matrix in CSV.")
    parser.add_argument("--noncancer_confmat_output_pkl", type=str, required=True,
                        help="Path to store non-cancer parsing confusion matrix in pickle file.")
    parser.add_argument("--episode_matching_output_csv", type=str, required=True,
                        help="Path to store episode matching performance CSV.")
    
    args = parser.parse_args()

    # Pre-defined dictionaries
    base_disease_dict = ['Glioma', 'Lymphoma', 'Metastasis', 'Non-malignant tumor', 'Non-tumor disease']
    disease_dict_noncancer_ans = ['Demyelinating', 'Infection/inflammation', 'Hemorrhage/vascular lesion', 'Stroke']

    # Dictionary for confusion matrix of disease
    disease_conf_dict = {
        k: {f"{k}_tp": 0, f"{k}_fp": 0, f"{k}_tn": 0, f"{k}_fn": 0}
        for k in base_disease_dict
    }

    # Load CSV data
    brain_data = pd.read_csv(args.input_csv)
    total_len = len(brain_data)

    # Counters for metrics
    len_count = {k: 0 for k in base_disease_dict}
    specificity_len = 0
    recall_len = 0
    precision_len = 0

    accuracy_total = 0
    specificity_total = 0
    recall_total = 0
    precision_total = 0

    print("========== Disease parsing evaluation start ==========")

    for idx, row in tqdm(brain_data.iterrows(), total=total_len):
        # Model output
        model_dis_list = literal_eval(row['final_output_extract_with_template'])

        # Human disease label
        human_disease = row['human_disease']

        # Remove specific markers
        if 'No disease' in human_disease:
            human_disease.remove('No disease')
        if 'disease_etc' in human_disease:
            human_disease.remove('disease_etc')

        # Convert other brain tumors to 'Non-malignant tumor'
        for d in ['Other intra-axial brain tumors', 'Other extra-axial brain tumors']:
            if d in human_disease:
                human_disease.remove(d)
                human_disease.append('Non-malignant tumor')

        # Convert demyelinating, infection, hemorrhage, stroke to 'Non-tumor disease'
        for d in ['Demyelinating', 'Infection/inflammation', 'Hemorrhage/vascular lesion', 'Stroke/infarction', 'Stroke']:
            if d in human_disease:
                if d in human_disease:
                    human_disease.remove(d)
                human_disease.append('Non-tumor disease')
        
        human_disease = list(set(human_disease))

        # Confusion matrix for each disease in base_disease_dict
        for c in base_disease_dict:
            if c in human_disease:
                len_count[c] += 1
                if c in model_dis_list:
                    disease_conf_dict[c][f"{c}_tp"] += 1
                else:
                    disease_conf_dict[c][f"{c}_fn"] += 1
            else:
                if c in model_dis_list:
                    disease_conf_dict[c][f"{c}_fp"] += 1
                else:
                    disease_conf_dict[c][f"{c}_tn"] += 1

        # Accuracy
        if len(set(model_dis_list).intersection(set(human_disease))) == len(set(human_disease).union(set(model_dis_list))):
            accuracy_total += 1

        # Specificity
        if len(human_disease) == 0:
            specificity_len += 1
            if len(model_dis_list) == 0:
                specificity_total += 1

        # Precision
        if len(model_dis_list) != 0:
            precision_len += 1
            precision_total += len(set(human_disease).intersection(model_dis_list)) / len(model_dis_list)

        # Recall
        if len(human_disease) != 0:
            recall_len += 1
            recall_total += len(set(human_disease).intersection(model_dis_list)) / len(human_disease)

    accuracy_score = accuracy_total * 100 / total_len if total_len else 0
    specificity_score = specificity_total * 100 / specificity_len if specificity_len else 0
    precision_score = precision_total * 100 / precision_len if precision_len else 0
    recall_score = recall_total * 100 / recall_len if recall_len else 0
    f1_score = calculate_f1_score(precision_score, recall_score)

    print(f"TOTAL-len : {total_len}")
    print(f"RECALL-len : {recall_len}")
    print(f"PRECISION-len : {precision_len}")
    print(f"SPECIFICITY-len : {specificity_len}")

    print("ACCURACY-score :", round(accuracy_score, 2))
    print("SPECIFICITY-score :", round(specificity_score, 2))
    print("RECALL-score :", round(recall_score, 2))
    print("PRECISION-score :", round(precision_score, 2))
    print("F1-score:", f1_score)

    print("========== Disease parsing evaluation finished ==========")

    # Save disease confusion matrix to CSV
    disease_conf_df = pd.DataFrame(disease_conf_dict).T
    disease_conf_df.to_csv(args.disease_confmat_output_csv, index=True)

    print("========== Non-cancer disease parsing evaluation start ==========")

    # Confusion matrix for non-cancer diseases
    non_cancer_conf_dict = {
        k: {f"{k}_tp": 0, f"{k}_fp": 0, f"{k}_tn": 0, f"{k}_fn": 0}
        for k in disease_dict_noncancer_ans
    }

    non_cancer_total_len = len(brain_data)

    step1_wrong_2_right_cnt = 0
    step1_right_2_wrong_cnt = 0
    step1_wrong_cnt = 0

    for idx, row in tqdm(brain_data.iterrows(), total=non_cancer_total_len):
        # Skip if model output for non-cancer is NaN
        if pd.isna(row['final_output_noncancer_extract_with_template']):
            continue

        noncancer_disease = literal_eval(row['final_output_noncancer_extract_with_template'])
        human_noncancer_disease = row['human_disease']

        if 'No disease' in human_noncancer_disease:
            human_noncancer_disease.remove('No disease')
        if 'disease_etc' in human_noncancer_disease:
            human_noncancer_disease.remove('disease_etc')
        
        if human_noncancer_disease == []:
            human_noncancer_disease = ['']

        # Filter only disease_dict_noncancer_ans
        final_human_noncancer_disease = []
        for d in disease_dict_noncancer_ans:
            if d in human_noncancer_disease:
                final_human_noncancer_disease.append(d)

        # Step1 check (e.g., debugging or intermediate checks)
        if ('Hemorrhage/vascular lesion' not in final_human_noncancer_disease and
            'Stroke' not in final_human_noncancer_disease and
            'Infection/inflammation' not in final_human_noncancer_disease and
            'Demyelinating' not in final_human_noncancer_disease):
            step1_wrong_cnt += 1
            if ('Hemorrhage/vascular lesion' not in noncancer_disease and
                'Stroke' not in noncancer_disease and
                'Infection/inflammation' not in noncancer_disease and
                'Demyelinating' not in noncancer_disease):
                step1_wrong_2_right_cnt += 1
        else:
            if ('Hemorrhage/vascular lesion' not in noncancer_disease and
                'Stroke' not in noncancer_disease and
                'Infection/inflammation' not in noncancer_disease and
                'Demyelinating' not in noncancer_disease):
                step1_right_2_wrong_cnt += 1

        # Update confusion matrix
        for d in disease_dict_noncancer_ans:
            if (d in final_human_noncancer_disease) and (d in noncancer_disease):
                non_cancer_conf_dict[d][f"{d}_tp"] += 1
            elif (d in final_human_noncancer_disease) and (d not in noncancer_disease):
                non_cancer_conf_dict[d][f"{d}_fn"] += 1
            elif (d not in final_human_noncancer_disease) and (d in noncancer_disease):
                non_cancer_conf_dict[d][f"{d}_fp"] += 1
            else:
                non_cancer_conf_dict[d][f"{d}_tn"] += 1

    # Print or store non-cancer confusion dictionary
    print(f"Non-cancer confusion matrix: {non_cancer_conf_dict}")

    # Save non-cancer confusion matrix as pickle
    with open(args.noncancer_confmat_output_pkl, 'wb') as rf:
        pickle.dump(non_cancer_conf_dict, rf)

    print("========== Non-cancer disease parsing evaluation finished ==========")

    print("========== Episode matching evaluation start ==========")

    epi_disease_key = ['Glioma', 'Lymphoma', 'Metastasis']
    episode_mapping_dict = {
        'Stable': 'Stable',
        'Progression': 'Progression',
        'Improvement': 'Improvement'
    }

    # Counters for episode evaluation
    total_episode_count = {k: 0 for k in epi_disease_key}
    precision_episode_count = {k: 0 for k in epi_disease_key}
    recall_episode_count = {k: 0 for k in epi_disease_key}

    total_right_episode_count = {k: 0 for k in epi_disease_key}
    precision_right_episode_count = {k: 0 for k in epi_disease_key}
    recall_right_episode_count = {k: 0 for k in epi_disease_key}

    specificity_len_episode_count = {k: 0 for k in epi_disease_key}
    specificity_episode_count = {k: 0 for k in epi_disease_key}

    for idx, row in tqdm(brain_data.iterrows(), total=len(brain_data)):
        episode = literal_eval(row['final_output_episode_extract_with_template'])

        human_disease = row['human_disease']
        # Human episodes
        human_episode = {
            'Glioma': row['human_glioma_episode'],
            'Lymphoma': row['human_lymphoma_episode'],
            'Metastasis': row['human_metastasis_episode']
        }

        # Normalize episodes
        for key in epi_disease_key:
            episode[key] = list(set(episode[key]))
            if episode[key] == []:
                episode[key] = ['']
            human_stable = False
            # Convert stable text forms to a single 'Stable'
            for stable_text in ['Stable, no change', 'Stable, no disease or no recur', 'Stable with treatment change']:
                if stable_text in human_episode[key]:
                    human_episode[key].remove(stable_text)
                    human_stable = True
            if human_stable:
                human_episode[key].append('Stable')

            # Trim or replace partial strings
            if 'Progression ' in human_episode[key]:
                human_episode[key].remove('Progression ')
                human_episode[key].append('Progression')
            if 'episode_etc' in human_episode[key]:
                human_episode[key].remove('episode_etc')

            human_episode[key] = list(set(human_episode[key]))
            if human_episode[key] == []:
                human_episode[key] = ['']

        # Accuracy: if disease is present in human_disease, then compare episodes
        for key in total_right_episode_count.keys():
            if key in human_disease:
                total_episode_count[key] += 1
                if (len(set(human_episode[key]).intersection(set(episode[key]))) ==
                    len(set(human_episode[key]).union(set(episode[key])))):
                    total_right_episode_count[key] += 1

        # Specificity: no episode in human => model should also have none
        for key in epi_disease_key:
            if human_episode[key] == ['']:
                specificity_len_episode_count[key] += 1
                if episode[key] == ['']:
                    specificity_episode_count[key] += 1

        # Precision: evaluate when model extracted an episode
        episode_tmp = {k: v for k, v in episode.items() if v != ['']}
        for key in episode_tmp.keys():
            precision_episode_count[key] += 1
            extracted_episodes = episode_tmp[key]
            row_precision = 0.0
            if len(extracted_episodes) > 0:
                row_precision = len(set(human_episode[key]).intersection(extracted_episodes)) / len(extracted_episodes)
            precision_right_episode_count[key] += row_precision

        # Recall: evaluate when human has an episode
        human_episode_tmp = {k: v for k, v in human_episode.items() if v != ['']}
        for key in human_episode_tmp.keys():
            recall_episode_count[key] += 1
            extracted_episodes = episode[key]
            row_recall = 0.0
            if len(human_episode_tmp[key]) > 0:
                row_recall = len(set(human_episode_tmp[key]).intersection(extracted_episodes)) / len(human_episode_tmp[key])
            recall_right_episode_count[key] += row_recall

    # Compute metrics
    lesion_yes_only_recall_final = {
        k: [round(recall_right_episode_count[k] / recall_episode_count[k] * 100, 2)]
        if recall_episode_count[k] != 0 else [0.0]
        for k in recall_right_episode_count
    }
    lesion_yes_only_precision_final = {
        k: [round(precision_right_episode_count[k] / precision_episode_count[k] * 100, 2)]
        if precision_episode_count[k] != 0 else [0.0]
        for k in precision_right_episode_count
    }
    lesion_total_acc_final = {
        k: [round(total_right_episode_count[k] / total_episode_count[k] * 100, 2)]
        if total_episode_count[k] != 0 else [0.0]
        for k in total_right_episode_count
    }
    lesion_total_spec_final = {
        k: [round(specificity_episode_count[k] / specificity_len_episode_count[k] * 100, 2)]
        if specificity_len_episode_count[k] != 0 else [0.0]
        for k in specificity_episode_count
    }

    # F1 for episodes
    lesion_yes_only_f1_final = {}
    for i, (k_prec, k_rec) in enumerate(zip(lesion_yes_only_precision_final.items(), lesion_yes_only_recall_final.items())):
        lesion_key_p, p_val = k_prec
        lesion_key_r, r_val = k_rec
        # Should match the same key
        lesion_yes_only_f1_final[lesion_key_p] = [calculate_f1_score(p_val[0], r_val[0])]

    print("EPISODE ACCURACY-score :", lesion_total_acc_final)
    print("-" * 60)
    print("EPISODE SPECIFICITY-score :", lesion_total_spec_final)
    print("-" * 60)
    print("EPISODE RECALL-score :", lesion_yes_only_recall_final)
    print("-" * 60)
    print("EPISODE PRECISION-score :", lesion_yes_only_precision_final)
    print("-" * 60)
    print("EPISODE F1-score:", lesion_yes_only_f1_final)

    # Combine metrics into a dataframe
    tmp = pd.DataFrame(lesion_total_acc_final, index=['Accuracy'])
    tmp = pd.concat([tmp, pd.DataFrame(lesion_yes_only_precision_final, index=['precision'])])
    tmp = pd.concat([tmp, pd.DataFrame(lesion_yes_only_recall_final, index=['recall'])])
    tmp = pd.concat([tmp, pd.DataFrame(lesion_yes_only_f1_final, index=['F1-score'])])
    tmp = pd.concat([tmp, pd.DataFrame(lesion_total_spec_final, index=['Specificity'])])
    tmp = tmp.T

    # Macro average
    print("EPISODE ACCURACY-score macro avg:", round(tmp['Accuracy'].mean(), 2))
    print("EPISODE SPECIFICITY-score macro avg:", round(tmp['Specificity'].mean(), 2))
    print("EPISODE RECALL-score macro avg:", round(tmp['recall'].mean(), 2))
    print("EPISODE PRECISION-score macro avg:", round(tmp['precision'].mean(), 2))
    print("EPISODE F1-score macro avg:", round(tmp['F1-score'].mean(), 2))

    # Save episode metrics to CSV
    tmp.to_csv(args.episode_matching_output_csv)

    print("========== Episode matching evaluation finished ==========")


if __name__ == "__main__":
    main()