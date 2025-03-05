import argparse
import pandas as pd
import numpy as np
import re
from ast import literal_eval

# Pre-defined disease dict
base_disease_dict = ['Glioma', 'Lymphoma', 'Metastasis', 'Non-malignant tumor', 'Non-tumor disease']
disease_dict_noncancer = ['Demyelinating', 'Infection/inflammation', 'Hemorrhage/vascular lesion', 'Stroke/infarction']

# Episode dict
episode_mapping_dict = {
    'Stable': 'Stable',
    'Progression': 'Progression',
    'Improvement': 'Improvement'
}


def parse_brain_mri(input_path: str, output_path: str, output_path_structured: str) -> None:
    """
    Brain MRI report를 파싱하는 함수입니다.
    
    Args:
        input_path (str): INPUT path to parse disease/episode info
        output_path (str): OUTPUT path to save total dataframe with parsed information
    """
    # Load data
    brain_data = pd.read_csv(input_path)
    
    brain_data['final_output_extract_with_template'] = np.nan
    brain_data['final_output_noncancer_extract_with_template'] = np.nan
    brain_data['final_output_episode_extract_with_template'] = np.nan

    for idx, row in brain_data.iterrows():
        # 1) Disease parsing
        tmp = row['inference_output'] \
                .split('</EXAMPLE 2>')[1] \
                .split('### Output:')[1] \
                .split('</EXAMPLE')[0] \
                .split('<EXAMPLE')[0] \
                .split('Explanation:')[0] \
                .split('Note:')[0]
        tmp = tmp.split('##')[0].replace('```','')

        # 'Diseases' part extraction
        L = tmp.split('\n\n')
        L = [l.strip() for l in L if re.match('Diseases', l.strip())]

        # 적절한 포맷으로 잘렸는지 확인
        if len(L) != 1:
            # If debug needed
            # print(f"Check parsing format at index: {idx}")
            pass
        
        split_L = L[0].split('\n')
        # Delete first line ('Diseases:')
        del split_L[0]
        split_L = split_L[:5]

        model_dis_list = []
        for line_disease, dict_disease in zip(split_L, base_disease_dict):
            if dict_disease in line_disease:
                if 'none' in line_disease.lower():
                    pass
                elif 'present' in line_disease.lower() or 'possible' in line_disease.lower():
                    model_dis_list.append(dict_disease)
                else:
                    pass
            else:
                # If debug needed
                # print(f"Mismatch: line='{line_disease}', dict_disease='{dict_disease}'")
                pass

        model_dis_list = list(set(model_dis_list))
        brain_data.loc[idx, 'final_output_extract_with_template'] = str(model_dis_list)

        # 2) Non-cancer detail parsing
        if not pd.isna(row['inference_output_noncancer']):
            tmp_nc = row['inference_output_noncancer'] \
                        .split('</EXAMPLE 2>')[1] \
                        .split('### Output:')[1] \
                        .split('</EXAMPLE')[0] \
                        .split('<EXAMPLE')[0] \
                        .split('Explanation:')[0] \
                        .split('Note:')[0]
            tmp_nc = tmp_nc.split('##')[0].replace('```','')
            L_nc = tmp_nc.split('\n\n')
            L_nc = [l.strip() for l in L_nc if re.match('Diseases', l.strip()) or re.match('Episode', l.strip())]
            L_nc = list(set(L_nc))
            
            L_nc = [[l.split('Episode')[0], 'Episode'+l.split('Episode')[1]]
                    if ('Diseases' in l and 'Episode' in l) else [l]
                    for l in L_nc]
            # to 1D list
            L_nc = sum(L_nc, [])

            # Non-cancer disease extraction
            try:
                disease_text = [l.strip() for l in L_nc if re.match('Diseases', l.strip())][0]
                disease_texts = disease_text.split('\n')[1:]
                disease = []
                for t in disease_texts:
                    for d in disease_dict_noncancer:
                        if d in t and 'present' in t.lower():
                            disease.append(d)
            except:
                # If not parsed
                disease_text = ''
                disease = [d for d in disease_dict_noncancer if d in disease_text]

            # Match original format
            if 'Stroke/infarction' in disease:
                disease.remove('Stroke/infarction')
                disease.append('Stroke')

            if disease == []:
                disease = ['']

            brain_data.loc[idx, 'final_output_noncancer_extract_with_template'] = str(disease)

        # 3) Episode parsing
        if not pd.isna(row['inference_output_epi']):
            tmp_epi = row['inference_output_epi'] \
                        .split('</EXAMPLE 2>')[1] \
                        .split('### Output:')[1] \
                        .split('</EXAMPLE')[0] \
                        .split('<EXAMPLE')[0] \
                        .split('Explanation:')[0] \
                        .split('Note:')[0]
            tmp_epi = tmp_epi.split('##')[0].replace('```','').strip()
            tmp_spt = tmp_epi.split('\n')

            episode = {'Glioma': [], 'Lymphoma': [], 'Metastasis': []}
            for s in tmp_spt:
                if 'Episode' in s or s.strip() == '':
                    continue
                for k in episode.keys():
                    if k in s:
                        for epi_model, epi_label in episode_mapping_dict.items():
                            if epi_model in s:
                                episode[k].append(epi_label)
        else:
            episode = {'Glioma': [], 'Lymphoma': [], 'Metastasis': []}

        brain_data.loc[idx, 'final_output_episode_extract_with_template'] = str(episode)

    # Parsing result to csv
    brain_data.to_csv(output_path, index=False)
    print("Output saved at:", output_path)

    disease_list = base_disease_dict[:-1]+disease_dict_noncancer
    episode_list = ['glioma_episode', 'lymphoma_episode', 'metastasis_episode']
    new_col_list = ['note_id', 'person_id', 'note_text', 'processed_text'] + disease_list + episode_list
    new_brain_df = pd.DataFrame(columns=new_col_list)
    for idx, row in brain_data.iterrows():
        new_data_to_concat = {c:[] for c in new_col_list}
        model_disease_list = [d for d in literal_eval(row['final_output_extract_with_template']) if d != 'Non-tumor disease']
        if not pd.isna(row['final_output_noncancer_extract_with_template']):
            non_tumor_list = literal_eval(row['final_output_noncancer_extract_with_template'])
            if non_tumor_list[0]!='':
                if 'Stroke' in non_tumor_list:
                    non_tumor_list.remove('Stroke')
                    non_tumor_list.append('Stroke/infarction')
                model_disease_list += non_tumor_list
        new_data_to_concat['note_id'].append(row['note_id'])
        new_data_to_concat['person_id'].append(row['person_id'])
        new_data_to_concat['note_text'].append(row['note_text'])
        new_data_to_concat['processed_text'].append(row['processed_text'])
        for d in disease_list:
            if d in model_disease_list:
                new_data_to_concat[d].append('y')
            else:
                new_data_to_concat[d].append('-')
        epi_dict = literal_eval(row['final_output_episode_extract_with_template'])
        if len(epi_dict['Glioma'])>0:
            new_data_to_concat['glioma_episode'].append(str(epi_dict['Glioma']))
        else:
            new_data_to_concat['glioma_episode'].append('-')
        if len(epi_dict['Lymphoma'])>0:
            new_data_to_concat['lymphoma_episode'].append(str(epi_dict['Lymphoma']))
        else:
            new_data_to_concat['lymphoma_episode'].append('-')
        if len(epi_dict['Metastasis'])>0:
            new_data_to_concat['metastasis_episode'].append(str(epi_dict['Metastasis']))
        else:
            new_data_to_concat['metastasis_episode'].append('-')
        new_brain_df = pd.concat([new_brain_df, pd.DataFrame(new_data_to_concat)], ignore_index=True)
    new_brain_df.to_csv(output_path_structured, index=False)
    print("Brain MRI parsing finished. Structured output saved at:", output_path_structured)


def main():
    parser = argparse.ArgumentParser(description="Brain MRI report parsing script.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input CSV file containing Brain MRI results.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output CSV file where parsed results will be saved.")
    parser.add_argument("--output_file_structured", type=str, required=True, 
                        help="Path to the output structured CSV file.")
    
    args = parser.parse_args()
    parse_brain_mri(input_path=args.input_file, output_path=args.output_file, output_path_structured=args.output_file_structured)


if __name__ == "__main__":
    main()
