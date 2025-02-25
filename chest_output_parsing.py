import argparse
import pandas as pd
import re
from tqdm import tqdm


# Pre-defined parsing dict
location_dict = {
    'Rt lung': ['right lung', 'rt[.|. | ]lung', 'both lung', 'both lungs'],
    'Lt lung': ['left lung', 'lt[.|. | ]lung', 'both lung', 'both lungs'],
    'Left Lingula': ['left lingula'],
    'BLL': ['both lower lobes', 'both lower lobe', 'bilateral lower lobe', 'bilateral lower lobes', 'bll', 'blls'],
    'RLL': ['right lower lobe', 'rt[.|. | ]lower lobe', 'rll'],
    'LLL': ['left lower lobe', 'lt[.|. | ]lower lobe', 'lll'],
    'RML': ['right middle lobe', 'rt[.|. | ]middle lobe', 'rml'],
    'BUL': ['both upper lobes', 'both upper lobe', 'bilateral upper lobe', 'bilateral upper lobes', 'bul', 'buls'],
    'RUL': ['right upper lobe', 'rt[.|. | ]upper lobe', 'rul'],
    'LUL': ['left upper lobe', 'lt[.|. | ]upper lobe', 'lul'],
    'pleural': ['pleural']
}

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

def parse_chest_ct(input_path: str, output_path: str) -> None:
    """
    Parse chest CT LLM output result and save in csv file.
    
    Args:
        input_path (str): INPUT path to parse lesion/location info
        output_path (str): OUTPUT path to save total dataframe with parsed information
    """
    
    print("===========Chest CT parsing start===========")
    
    # Assume input csv file
    chest_data = pd.read_csv(input_path)

    for idx, row in tqdm(chest_data.iterrows(), total=chest_data.shape[0]):
        text = row['final_output'].split('</EXAMPLE 2>')[1].split('### Output:')[1] \
                                 .split('<EXAMPLE')[0].split('</EXAMPLE')[0].lower().strip()
        final_match_output = {}

        # Case 1: with no "lesion" word
        if not re.match(r'\s*\**lesion', text):
            text_spt = text.split('\n')
            if 'no lesion' in text_spt[0]:
                if 'except one' in text_spt[0]:
                    # Exceptional case
                    pass
            else:
                if 'lesions' in text:
                    lidx = text.find('lesion')
                    final_text = text[lidx:]
                    final_text_spt = final_text.split('\n')

                    # Extract line index with lesion information
                    lesion_sent_idx = []
                    for i, s in enumerate(final_text_spt):
                        if ':' in s and '*' in s and 'none' not in s:
                            lesion_sent_idx.append(i)

                    for i, sent_idx in enumerate(lesion_sent_idx):
                        parsed_les = ''
                        start_location = sent_idx + 1

                        # Extract location for specific lesion
                        if i == len(lesion_sent_idx) - 1:
                            location_list = final_text_spt[start_location:]
                        else:
                            location_list = final_text_spt[start_location:lesion_sent_idx[i + 1]]

                        # lesion_dict matching
                        for lesion_key, pattern_list in lesion_dict.items():
                            for p in pattern_list:
                                if re.search(rf"\W+{p}\W+|\W+{p}$|^{p}\W+|^{p}$", final_text_spt[sent_idx]):
                                    parsed_les = lesion_key
                                    break
                            if parsed_les:
                                break

                        # With no parsed lesion
                        if not parsed_les:
                            continue

                        # location_dict matching
                        final_location_list = []
                        for loc in location_list:
                            for loc_key, loc_pattern_list in location_dict.items():
                                for pattern in loc_pattern_list:
                                    if re.search(rf"\W+{pattern}\W+|\W+{pattern}$", loc) and 'none' not in loc:
                                        final_location_list.append(loc_key)
                                        break

                        # Save in dataframe
                        if parsed_les not in final_match_output:
                            final_match_output[parsed_les] = final_location_list

        # Case 2: "lesion" word exists
        else:
            lidx = text.find('lesion')
            final_text = text[lidx:]
            if 'final answer' in final_text:
                final_text = final_text.split('final answer')[0]
            final_text_spt = final_text.split('\n')

            lesion_sent_idx = []
            for i, s in enumerate(final_text_spt):
                if ':' in s and '*' in s and 'none' not in s:
                    lesion_sent_idx.append(i)

            for i, sent_idx in enumerate(lesion_sent_idx):
                parsed_les = ''
                start_location = sent_idx + 1

                if i == len(lesion_sent_idx) - 1:
                    location_list = final_text_spt[start_location:]
                else:
                    location_list = final_text_spt[start_location:lesion_sent_idx[i + 1]]

                # lesion matching
                for lesion_key, pattern_list in lesion_dict.items():
                    for p in pattern_list:
                        if re.search(rf"\W+{p}\W+|\W+{p}$|^{p}\W+|^{p}$", final_text_spt[sent_idx]):
                            parsed_les = lesion_key
                            break
                    if parsed_les:
                        break

                if not parsed_les:
                    continue

                # location matching
                final_location_list = []
                for loc in location_list:
                    for loc_key, loc_pattern_list in location_dict.items():
                        for pattern in loc_pattern_list:
                            if re.search(rf"\W+{pattern}\W+|\W+{pattern}$", loc) and 'none' not in loc:
                                final_location_list.append(loc_key)
                                break

                # Save in dataframe
                if parsed_les not in final_match_output:
                    final_match_output[parsed_les] = final_location_list
                else:
                    final_match_output[parsed_les].extend(final_location_list)
                    final_match_output[parsed_les] = list(set(final_match_output[parsed_les]))

        # Save in string format -- need to use literal_eval afterward
        chest_data.loc[idx, 'final_output_extract_with_template'] = str(final_match_output)

    # Save in csv format
    chest_data.to_csv(output_path, index=False)
    print("===========Chest CT parsing finished===========")


def main():
    parser = argparse.ArgumentParser(description="Chest CT parsing script.")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the input file (pickle or CSV).")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Path to the output CSV file.")
    
    args = parser.parse_args()

    parse_chest_ct(input_path=args.input_file, output_path=args.output_file)


if __name__ == "__main__":
    main()
