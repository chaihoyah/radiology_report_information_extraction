#!/usr/bin/env python
# chest_ct_evaluation.py

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
from ast import literal_eval

# 전처리에 사용할 사전 정의
lesion_dict = {
    'nodule': ['nodule', 'nodules', 'nodular'],
    'ggo': ['ggo', 'ggos', 'ground glass opacity', 'ground-glass opacity'],
    'consolidation': ['consolidation', 'consolidations'],
    'pleural effusion': ['pleural effusion', 'pleural effusions'],
    'atelectasis': ['atelectasis', 'atelectases'],
    'fibrosis': ['fibrosis', 'fibroses'],
    'bronchiectasis': ['bronchiectasis', 'bronchiectasises'],
    'mass': ['mass', 'masses'],
    'bronchial wall thickening': ['bronchial wall thickening', 'bronchial wall thickenings'],
    'interstitial thickening': ['interstitial thickening', 'interstitial thickenings'],
    'pleural thickening': ['pleural thickening', 'pleural thickenings']
}

lesion_key_for_location_mapping = [
    'nodule', 'ggo', 'consolidation', 'atelectasis', 'fibrosis',
    'bronchiectasis', 'mass', 'bronchial wall thickening', 'interstitial thickening'
]

def calculate_f1_score(precision, recall):
    """F1-score 계산 함수."""
    if precision + recall == 0:
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 2)


def main():
    parser = argparse.ArgumentParser(description="Final evaluation script for chest CT parsing.")
    
    # 입력 인자 정의
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to the final parsed CSV file.")
    parser.add_argument("--lesion_confmat_output_csv", type=str, required=True,
                        help="Path to store lesion parsing confusion matrix CSV.")
    parser.add_argument("--location_matching_output_csv", type=str, required=True,
                        help="Path to store location matching performance CSV.")
    parser.add_argument("--location_begin_col_index", type=int, required=True,
                        help="Starting column index for location columns.")
    parser.add_argument("--location_end_col_index", type=int, required=True,
                        help="Ending column index for location columns (exclusive).")
    
    args = parser.parse_args()

    # CSV 파일 로드
    chest_data = pd.read_csv(args.input_csv)

    # Grouping by note_id
    output_grouped = chest_data.groupby(['note_id'])
    total_len = len(output_grouped)

    # Metrics용 변수들
    accuracy_total = 0
    specificity_total = 0
    recall_total = 0
    precision_total = 0
    
    specificity_len = 0
    recall_len = 0
    precision_len = 0

    # 병변별 혼동행렬
    lesion_by_lesion_conf_mat = {k: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for k in lesion_dict.keys()}

    print("========== Lesion parsing evaluation start ==========")

    # Lesion 평가
    for _, rows in tqdm(output_grouped, total=total_len):
        # 사람(정답) 레이블 수집
        total_keywords = []
        for idx, row in rows.iterrows():
            if pd.isna(row['lesion_HUMAN']):
                continue
            # Negation이 없을 때만(또는 'n'일 때만) 추가
            if pd.isna(row['Negation']): 
                total_keywords.append(row['lesion_HUMAN'])

        total_keywords = [k for k in total_keywords if pd.notnull(k)]
        new_lesion_answer_dict = []
        for s in total_keywords:
            new_lesion_answer_dict.extend(s.split(' / '))

        human_lesion = []
        for na in new_lesion_answer_dict:
            for k, v in lesion_dict.items():
                if na in v and k not in human_lesion:
                    human_lesion.append(k)    
                    break
        
        # 모델 예측
        # (같은 note_id 그룹이므로 동일한 final_output_extract_with_template 가정)
        lesion = list(literal_eval(rows['final_output_extract_with_template'].iloc[0]).keys())
        # nodules -> nodule로 통일
        lesion = [l if l != 'nodules' else 'nodule' for l in lesion]

        # 병변별로 TP, FP, TN, FN 누적
        for k in lesion_dict.keys():
            if k in human_lesion:
                if k in lesion:
                    lesion_by_lesion_conf_mat[k]['TP'] += 1
                else:
                    lesion_by_lesion_conf_mat[k]['FN'] += 1
            else:
                if k not in lesion:
                    lesion_by_lesion_conf_mat[k]['TN'] += 1
                else:
                    lesion_by_lesion_conf_mat[k]['FP'] += 1

        # 전체 정확도(ACCURACY)
        # (human_lesion과 model 예측이 동일할 때)
        if len(set(lesion).intersection(set(human_lesion))) == len(set(human_lesion).union(set(lesion))):
            accuracy_total += 1

        # SPECIFICITY: (실제 병변이 없는 경우 -> 모델이 없는 것으로 판단한 비율)
        if len(human_lesion) == 0:
            specificity_len += 1
            if len(lesion) == 0:
                specificity_total += 1

        # PRECISION: (모델이 병변을 추출한 경우 -> 실제 정답과의 일치도)
        if len(lesion) != 0:
            precision_len += 1
            precision_total += len(set(human_lesion).intersection(lesion)) / len(lesion)

        # RECALL: (실제로 병변이 있는 경우 -> 모델이 얼마만큼 찾았는가)
        if len(human_lesion) != 0:
            recall_len += 1
            recall_total += len(set(human_lesion).intersection(lesion)) / len(human_lesion)

    # 지표 계산
    accuracy_score = (accuracy_total * 100) / total_len if total_len else 0
    specificity_score = (specificity_total * 100) / specificity_len if specificity_len else 0
    precision_score = (precision_total * 100) / precision_len if precision_len else 0
    recall_score = (recall_total * 100) / recall_len if recall_len else 0
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

    # 혼동행렬 저장
    conf_mat = pd.DataFrame(lesion_by_lesion_conf_mat).T
    conf_mat.to_csv(args.lesion_confmat_output_csv)
    print(conf_mat)

    print("========== Lesion parsing evaluation finished ==========")

    print("========== Location matching evaluation start ==========")

    # 위치 매칭용 변수
    location_columns = chest_data.columns[args.location_begin_col_index : args.location_end_col_index]
    
    total_location_count = {k: 0 for k in lesion_key_for_location_mapping}
    precision_location_count = {k: 0 for k in lesion_key_for_location_mapping}
    recall_location_count = {k: 0 for k in lesion_key_for_location_mapping}

    total_right_location_count = {k: 0 for k in lesion_key_for_location_mapping}
    precision_right_location_count = {k: 0 for k in lesion_key_for_location_mapping}
    recall_right_location_count = {k: 0 for k in lesion_key_for_location_mapping}

    specificity_len_location_count = {k: 0 for k in lesion_key_for_location_mapping}
    specificity_location_count = {k: 0 for k in lesion_key_for_location_mapping}

    for idx, row in tqdm(chest_data.iterrows(), total=chest_data.shape[0]):
        # Skip if negation
        if row['Negation'] == 'y':
            continue

        # parse human location
        tmp_location_series = row[args.location_begin_col_index : args.location_end_col_index]
        all_location_list = [
            i for i, val in enumerate(tmp_location_series.values.tolist()) if val == 'y'
        ]
        try:
            lesion_answer_list = row['lesion_HUMAN'].split(' / ')
        except:
            lesion_answer_list = []

        for lesion_answer in lesion_answer_list:
            lesion_answer = lesion_answer.strip()
            # del pleural effusion, pleural thickening for location mapping
            if lesion_answer in ['pleural effusion', 'pleural thickening']:
                continue

            key = ""
            for lk, v in lesion_dict.items():
                if lesion_answer in v:
                    key = lk
                    break
            if key == "":
                continue

            # human location (lowercase)
            human_location = list(location_columns[all_location_list])
            human_location = [loc.lower() for loc in human_location]

            # model parsed location
            model_parsed_dict = {}
            try:
                model_parsed_dict = dict(literal_eval(row['final_output_extract_with_template']))
            except:
                pass

            # nodules -> nodule 
            if "nodules" in model_parsed_dict.keys():
                model_parsed_dict["nodule"] = copy.deepcopy(model_parsed_dict["nodules"])
                del model_parsed_dict["nodules"]

            for mk in model_parsed_dict:
                model_parsed_dict[mk] = list(set([loc.lower() for loc in model_parsed_dict[mk]]))

            # ACCURACY
            total_location_count[key] += 1
            if key in model_parsed_dict:
                pred_location = model_parsed_dict[key]
                if len(set(human_location).intersection(set(pred_location))) == len(set(human_location).union(set(pred_location))):
                    total_right_location_count[key] += 1

            # SPECIFICITY - proportion of model output of none when actual human location is none
            if len(human_location) == 0:
                specificity_len_location_count[key] += 1
                if key in model_parsed_dict and len(model_parsed_dict[key]) == 0:
                    specificity_location_count[key] += 1

            # PRECISION - Caculate when LLM extracted location exists
            if key in model_parsed_dict and len(model_parsed_dict[key]) != 0:
                precision_location_count[key] += 1
                pred_locs = model_parsed_dict[key]
                row_precision = len(set(human_location).intersection(pred_locs)) / len(pred_locs) if len(pred_locs) else 0
                precision_right_location_count[key] += row_precision

            # RECALL - Caculate when actual human locationexists
            if len(human_location) != 0:
                recall_location_count[key] += 1
                if key in model_parsed_dict:
                    pred_locs = model_parsed_dict[key]
                    row_recall = len(set(human_location).intersection(pred_locs)) / len(human_location) if len(human_location) else 0
                    recall_right_location_count[key] += row_recall

    # In case of division-by-zero
    for k, v in specificity_len_location_count.items():
        if v == 0:
            specificity_len_location_count[k] = 1

    # Final evaluation matrix
    lesion_yes_only_recall_final = {
        k: [round(recall_right_location_count[k] / recall_location_count[k] * 100, 2)] 
        if recall_location_count[k] != 0 else [0.0]
        for k in recall_right_location_count
    }
    lesion_yes_only_precision_final = {
        k: [round(precision_right_location_count[k] / precision_location_count[k] * 100, 2)] 
        if precision_location_count[k] != 0 else [0.0]
        for k in precision_right_location_count
    }
    lesion_total_acc_final = {
        k: [round(total_right_location_count[k] / total_location_count[k] * 100, 2)] 
        if total_location_count[k] != 0 else [0.0]
        for k in total_right_location_count
    }
    lesion_total_spec_final = {
        k: [round(specificity_location_count[k] / specificity_len_location_count[k] * 100, 2)] 
        for k in specificity_location_count
    }

    # F1 score calculation
    lesion_yes_only_f1_final = {}
    for i, (k_prec, k_rec) in enumerate(zip(
        lesion_yes_only_precision_final.items(),
        lesion_yes_only_recall_final.items()
    )):
        lesion_key_p, p_val = k_prec
        lesion_key_r, r_val = k_rec
        lesion_yes_only_f1_final[lesion_key_p] = [calculate_f1_score(p_val[0], r_val[0])]

    print("LOCATION ACCURACY-score :", lesion_total_acc_final)
    print("-" * 60)
    print("LOCATION SPECIFICITY-score :", lesion_total_spec_final)
    print("-" * 60)
    print("LOCATION RECALL-score :", lesion_yes_only_recall_final)
    print("-" * 60)
    print("LOCATION PRECISION-score :", lesion_yes_only_precision_final)
    print("-" * 60)
    print("LOCATION F1-score:", lesion_yes_only_f1_final)

    # To pandas dataframe
    tmp = pd.DataFrame(lesion_yes_only_recall_final, index=['recall'])
    tmp = pd.concat([tmp, pd.DataFrame(lesion_yes_only_precision_final, index=['precision'])])
    tmp = pd.concat([tmp, pd.DataFrame(lesion_yes_only_f1_final, index=['F1-score'])])
    tmp = pd.concat([tmp, pd.DataFrame(lesion_total_acc_final, index=['Accuracy'])])
    tmp = pd.concat([tmp, pd.DataFrame(lesion_total_spec_final, index=['Specificity'])])
    tmp = tmp.T

    # macro average
    print("LOCATION ACCURACY-score macro avg:", round(tmp.Accuracy.mean(), 2))
    print("LOCATION SPECIFICITY-score macro avg:", round(tmp.Specificity.mean(), 2))
    print("LOCATION RECALL-score macro avg:", round(tmp.recall.mean(), 2))
    print("LOCATION PRECISION-score macro avg:", round(tmp.precision.mean(), 2))
    print("LOCATION F1-score macro avg:", round(tmp['F1-score'].mean(), 2))

    tmp.to_csv(args.location_matching_output_csv)
    print("========== Location matching evaluation finished ==========")


if __name__ == "__main__":
    main()
