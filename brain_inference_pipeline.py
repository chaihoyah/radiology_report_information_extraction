# script1.py

import os
import re
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import copy

import pytorch_lightning as pl
from tqdm import tqdm
from typing import List

from bert_model import LMModel  # <-- Import your model from bert_model.py
from custom_morpheme_aware_tokenizer import morphemeTokenizer

# HuggingFace / tokenizers
from tokenizers import Tokenizer, processors
from tokenizers.pre_tokenizers import (
    ByteLevel as ByteLevelPretokenizer,
    PreTokenizer,
    Sequence as PretokenizerSequence
)
from transformers import (
    PreTrainedTokenizerFast,
    AutoTokenizer,
    pipeline
)

def preprocessing(row, column_name):
    main_str = row[column_name]
    if pd.isna(main_str):
        return np.nan
    try:
        processed = re.sub(r'\n|\r|\t', ' ', main_str)
        processed = re.sub(r' +', ' ', processed)
        processed = re.sub(r"\-+\>", "", processed)
        processed = processed.strip()
        processed = re.sub(r"[^가-힣A-Za-z0-9\/\\\+\-\(\)\.\,\#\!\?\'\`\: ]", "", processed)
    except:
        print(main_str)
    return processed.lower()

def concat_finding_clsn(row, finding_col_name, clsn_col_name):
    if pd.isna(row[finding_col_name]):
        return row[clsn_col_name]
    elif pd.isna(row[clsn_col_name]):
        return row[finding_col_name]
    else:
        return row[finding_col_name]+'\n\n'+row[clsn_col_name]

# -------------------------------------------------------------------------
# BERT Inference Function
# -------------------------------------------------------------------------
def batch_infer_bert(texts, tokenizer, model, max_length=256, batch_size=16):
    """
    Run your BERT-like model in batches. Return a [num_samples, label_size] array of 0/1.
    """
    all_preds = []
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]
        encodings = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            logits = model(encodings)  # shape: [batch_size, LABEL_SIZE]

        binary_logits = (logits >= 0.5).int().cpu().numpy()
        all_preds.extend(binary_logits)

    return np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Script1: BERT + LLM Inference Pipeline.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to your input CSV dataframe.")
    parser.add_argument("--finding_column", type=str, default="note_text", help="Name of the column in CSV with raw finding text data.")
    parser.add_argument("--conclusion_column", type=str, default="note_text", help="Name of the column in CSV with raw conclusion text data.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final dataframe (pkl).")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to morphological tokenizer JSON file.")
    parser.add_argument("--bert_checkpoint", type=str, required=True, help="Path to the BERT checkpoint (.ckpt).")
    parser.add_argument("--llm_path", type=str, required=True, help="Path to your local LLM model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for BERT inference.")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length for BERT inference.")
    parser.add_argument("--label_list", nargs='+', default=[
        'Glioma', 'Lymphoma', 'Metastasis', 'Other intra-axial brain tumors',
        'Other extra-axial brain tumors', 'Demyelinating', 'Infection/inflammation',
        'Hemorrhage/vascular lesion', 'Stroke', 'No disease', 'disease_etc'
    ], help="List of labels (default 11).")
    args = parser.parse_args()

    # 1) Load your DataFrame
    df = pd.read_csv(args.input_csv)
    #df = df.loc[:10,:] ## For sample check
    print(f"Loaded data: {df.shape[0]} rows.")

    # 2) Load tokenizer from JSON (Morphological + ByteLevel)
    tokenizer_obj = Tokenizer.from_file(args.tokenizer_path)
    tokenizer_obj.unk_token = "[UNK]"
    tokenizer_obj.sep_token = "[SEP]"
    tokenizer_obj.pad_token = "[PAD]"
    tokenizer_obj.cls_token = "[CLS]"
    tokenizer_obj.mask_token = "[MASK]"

    cls_token_id = tokenizer_obj.token_to_id("[CLS]")
    sep_token_id = tokenizer_obj.token_to_id("[SEP]")

    tokenizer_obj.post_processor = processors.TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
    )

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    # Attach morphological pre-tokenizer if needed:
    fast_tokenizer.pre_tokenizer = PretokenizerSequence([
        PreTokenizer.custom(morphemeTokenizer()),
        ByteLevelPretokenizer()
    ])

    fast_tokenizer.unk_token = "[UNK]"
    fast_tokenizer.sep_token = "[SEP]"
    fast_tokenizer.pad_token = "[PAD]"
    fast_tokenizer.cls_token = "[CLS]"
    fast_tokenizer.mask_token = "[MASK]"

    # 3) Load your BERT-like model from the checkpoint
    label_size = len(args.label_list)
    # model = LMModel.load_from_checkpoint(
    #     checkpoint_path=args.bert_checkpoint,
    #     strict=False,
    #     model_name_or_path=args.bert_base_model,
    #     vocab_size=fast_tokenizer.vocab_size,
    #     LABEL_SIZE=label_size
    # )
    model = LMModel.load_from_checkpoint(
        checkpoint_path=args.bert_checkpoint,
        strict=False,
        LABEL_SIZE=label_size

    )
    print("BERT model loaded.")

    # 4) Run BERT inference if you need to fill a 'goldsilver_label' column
    #    If you already have it, you can skip. We'll demonstrate usage anyway.
    df.dropna(subset=[args.finding_column, args.conclusion_column], how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['processed_finding'] = df.apply(preprocessing, args=[args.finding_column], axis=1)
    df['processed_conclusion'] = df.apply(preprocessing, args=[args.conclusion_column], axis=1)
    df['processed_text'] = df.apply(concat_finding_clsn, args=[args.finding_column, args.conclusion_column], axis=1)
    texts = df['processed_text'].tolist()  # Adjust column name if needed
    all_preds = batch_infer_bert(
        texts,
        tokenizer=fast_tokenizer,
        model=model,
        max_length=args.max_length,
        batch_size=args.batch_size
    )
    # Save the results: each row is 0/1 for each label
    df['goldsilver_label'] = [row.tolist() for row in all_preds]

    # 5) LLM Inference pipeline
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    # Ensure we have a pad token if not present
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

    llm_pipe = pipeline(
        "text-generation",
        model=args.llm_path,
        tokenizer=llm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda",
        max_new_tokens=200,
        repetition_penalty=1.2,
        do_sample=False,
        num_beams=5
    )
    print("LLM pipeline ready.")

    # 6) Build LLM prompts from BERT inference, run generation row-by-row
    df['inference_output'] = None
    df['input_prompt'] = None
    df['inference_output_epi'] = None
    df['input_prompt_epi'] = None
    df['inference_output_noncancer'] = None
    df['input_prompt_noncancer'] = None

    non_cancer_list = ['Demyelinating', 'Infection/inflammation', 'Hemorrhage/vascular lesion', 'Stroke']

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM Inference"):
        # Convert BERT results -> label names
        binary_list = row['goldsilver_label']
        predicted_labels = [
            label for label_idx, label in enumerate(args.label_list) if binary_list[label_idx] == 1
        ]
        # Remove "No disease"
        predicted_labels = [lbl for lbl in predicted_labels if lbl != "No disease"]
        if not predicted_labels or predicted_labels == ["disease_etc"]:
            predicted_labels = ["None"]

        # Brain infer prompt
        disease_str = ", ".join(predicted_labels)
        prompt = f"""### System: You are a helpful radiologist specialized in analyzing radiology reports.

### Instruction: Describe what brain related diseases are present based on the input radiology report. Describe if any disease (Glioma, Lymphoma, Metastasis, Non-malignant tumor, Non-tumor disease) is present or not. Non-malignant tumors only include other intra-axial or extra-axial brain tumors. And non-tumor diseases only include the following diseases: demyelinating, infection/inflammation, hemorrhage/vascular lesion, stroke/infarction. Please refer to the possible diseases that could be written in the report. Be aware that diseases are not present if they are negated. Describe output with given output template and all present diseases must be categorized into the one of disease types mentioned in output template. Check examples.

### Output template:
Diseases
* Glioma: Present or None
* Lymphoma: Present or None
* Metastasis: Present or None
* Non-malignant tumor: Present or None
* Non-tumor disease: Present or None


<EXAMPLE 1>
### Input Radiology Report (brain MRI report): r/o vasculitis especially behcet disease, most likely. r/o demyelinating condition. r/o diffuse infiltrating tumor such as gliomatosis, less likely.

### Possible Diseases: Demyelinating, Hemorrhage/vascular lesion

### Output:
Diseases
* Glioma: Present
* Lymphoma: None
* Metastasis: None
* Non-malignant tumor: None
* Non-tumor disease: Present
</EXAMPLE 1>

<EXAMPLE 2>
### Input Radiology Report (brain MRI report): history: 1. s/p gks for suspicious small metastatic tumor. 2. compared to 2013-11-13 mr. findings: 1. no change of the 5mm sized tiny enhancing lesion (#401-9988)at right frontal lobe, orbital gyrus. r/o stable state. 2. nonspecific t2 high signal intensity lesions within the bilateral periventricular white matter and centrum semiovale. - no significant interval change. 3. small retention cyst in right maxillary sinus.

### Possible Diseases: Metastasis

### Output:
Diseases
* Glioma: None
* Lymphoma: None
* Metastasis: Present
* Non-malignant tumor: None
* Non-tumor disease: None
</EXAMPLE 2>


### Possible Diseases: {disease_str}

### Input Radiology Report (brain MRI report): {row['processed_text']}

### Output:
"""

        # Run LLM generation
        output = llm_pipe(prompt, max_new_tokens=300, do_sample=False, num_beams=5)
        df.loc[idx, 'inference_output'] = copy.deepcopy(output[0]['generated_text'])
        df.loc[idx, 'input_prompt'] = copy.deepcopy(prompt)

        # Optional: manage GPU memory
        del output
        del prompt
        torch.cuda.empty_cache()

        output_data_check = df.loc[idx, 'inference_output'].split('</EXAMPLE 2>')[1].split('### Output:')[1].split('</EXAMPLE')[0].split('<EXAMPLE')[0].split('Explanation:')[0].split('Note:')[0]
        output_data_list = output_data_check.lower().strip().split('\n')
        cancer_list = []
        ## glioma
        if 'present' in output_data_list[1].strip() or 'possible' in output_data_list[1].strip():
            cancer_list.append('Glioma')
        elif 'none' in output_data_list[1].strip() or 'less likely' in output_data_list[1].strip():
            pass
        else:
            print(f'Wrong glioma at index {idx}')
        ## lymphoma
        if 'present' in output_data_list[2].strip() or 'possible' in output_data_list[2].strip():
            cancer_list.append('Lymphoma')
        elif 'none' in output_data_list[2].strip() or 'less likely' in output_data_list[2].strip():
            pass
        else:
            print('Wrong lymphoma')
        ## meta
        if 'present' in output_data_list[3].strip() or 'possible' in output_data_list[3].strip():
            cancer_list.append('Metastasis')
        elif 'none' in output_data_list[3].strip() or 'less likely' in output_data_list[3].strip():
            pass
        else:
            print('Wrong meta')
        
        is_non_tumor_exist = False
        ## non-tumor disease
        if 'present' in output_data_list[5].strip() or 'possible' in output_data_list[5].strip():
            is_non_tumor_exist = True
        elif 'none' in output_data_list[5].strip() or 'less likely' in output_data_list[5].strip():
            pass
        else:
            print('Wrong non-tumor')

        if len(cancer_list)>0:
            print('in episode')
            print(output_data_list)
            print(cancer_list)
            print(is_non_tumor_exist)
            print('='*200)
            output_template = 'Episode\n'
            cancer_str = ', '.join(cancer_list)
            if len(cancer_str.split(', ')) == 1:
                instruction_string = f"{cancer_str.split(', ')[0]} is"
                output_template += f"* {cancer_str.split(', ')[0]}: (Progression, Stable, Improvement, No episode information)"
            elif len(cancer_str.split(', ')) == 2:
                instruction_string = f"{cancer_str.split(', ')[0]} and {cancer_str.split(', ')[1]} are"
                output_template += f"* {cancer_str.split(', ')[0]}: (Progression, Stable, Improvement, No episode information)\n* {cancer_str.split(', ')[1]}: (Progression, Stable, Improvement, No episode information)"
            elif len(cancer_str.split(', ')) == 3:
                instruction_string = f"{cancer_str.split(', ')[0]}, {cancer_str.split(', ')[1]}, and {cancer_str.split(', ')[2]} are"
                output_template += f"* {cancer_str.split(', ')[0]}: (Progression, Stable, Improvement, No episode information)\n* {cancer_str.split(', ')[1]}: (Progression, Stable, Improvement, No episode information)\n* {cancer_str.split(', ')[2]}: (Progression, Stable, Improvement, No episode information)"

            prompt=f"""### System: You are a helpful radiologist specialized in analyzing radiology reports.

### Instruction: Describe target diseases' episode if there is any comparison with previous report; {instruction_string} included in Target Diseases. Possible episode types are (Progression, Stable, Improvement). Describe output with given output template and all present diseases and episodes must be categorized into the one of disease and episode types mentioned in output template. Be aware that multiple episode types can be matched to one disease. Check examples.

### Output template:
{output_template}


<EXAMPLE 1>
### Input Radiology Report (brain MRI report): clinical information: 1. s/p chemotherapy for the diffuse large b cell lymphoma, primary cns lymphoma. 2. previous mr과 비교 판독함. findings: 1. decreased size of the ill-defined t2 high signal intensity lesion with hemorrhagic foci in the left frontoparietal lobe cingulate gyrus. 1) decreased enhancing foci within the lesion. 2) decreased mass effect 2. slightly increased size of the 1 cm sized t2 high signal intensity lesion with increased foci of the enhancement in the left frontal lobe subcortical white matter. 3. decreased size of the nonenhancing lesion around 0.3 cm size in the left cerebellar hemisphere.  overlly responsive disease of the primary cns lymphoma after chemotherapy since last mr. (left frontal lobe의 1 cm sized lesion은 크기가 약간 증가함)

### Target Diseases: Lymphoma

### Output:
Episode
* Lymphoma: Progression, Improvement
</EXAMPLE 1>

<EXAMPLE 2>
### Input Radiology Report (brain MRI report): history: 1. s/p gks for suspicious small metastatic tumor. 2. compared to previous mr. findings: 1. no change of the 5mm sized tiny enhancing lesion (#401-9988)at right frontal lobe, orbital gyrus. r/o stable state. 2. nonspecific t2 high signal intensity lesions within the bilateral periventricular white matter and centrum semiovale. - no significant interval change. 3. small retention cyst in right maxillary sinus.

### Target Diseases: Metastasis

### Output:
Episode
* Metastasis: Stable
</EXAMPLE 2>


### Target Diseases: {cancer_str}

### Input Radiology Report (brain MRI report): {row['processed_text']}

### Output:
"""
                # Run LLM generation
            output = llm_pipe(prompt, max_new_tokens=300, do_sample=False, num_beams=5)
            df.loc[idx, 'inference_output_epi'] = copy.deepcopy(output[0]['generated_text'])
            df.loc[idx, 'input_prompt_epi'] = copy.deepcopy(prompt)
            del output
            del prompt
            torch.cuda.empty_cache()
        
        if is_non_tumor_exist:
            print('in non_tumor')
            print(output_data_list)
            print(cancer_list)
            print(is_non_tumor_exist)
            print('='*200)
            non_cancer_bert_final_list = []
            for c in non_cancer_list:
                if c in predicted_labels:
                    if c== 'Stroke':
                        non_cancer_bert_final_list.append('Stroke/infarction')
                    else:
                        non_cancer_bert_final_list.append(c)
            final_bert_str = ''
            if len(non_cancer_bert_final_list) == 0 :
                final_bert_str = 'None'
            else:
                final_bert_str = ', '.join(non_cancer_bert_final_list)
            prompt=f"""### System: You are a helpful radiologist specialized in analyzing radiology reports.

### Instruction: Describe what brain related diseases are present based on the input radiology report. Describe if described target diseases are present or not. Be aware that diseases are not present if they are negated. Describe output with given output template and all present diseases must be categorized into the one of disease types mentioned in output template. Check examples.

### Output template:
Diseases
* Demyelinating: Present or None
* Infection/inflammation: Present or None
* Hemorrhage/vascular lesion: Present or None
* Stroke/infarction: Present or None

<EXAMPLE 1>
### Input Radiology Report (brain MRI report): history: 1. multiple myeloma. 2. oliguria (2da). findings and conclusions: 1. enlarged pituitary galnd with homogeneous enhancement.  r/o pituitary marcoadenoma. 2. old infarction in pons and both basal ganglia. 3. a tiny microbleed in posteior limb of right internal capsule. 4. confluent flair high signal intensity in both deep white matter.  r/o chronic ischemic change. 5. no newly appeared diffusion restriction in brain.

### Output:
Diseases
* Demyelinating: None
* Infection/inflammation: None
* Hemorrhage/vascular lesion: Present
* Stroke/infarction: Present
</EXAMPLE 1>

<EXAMPLE 2>
### Input Radiology Report (brain MRI report): history: fever findings: 1. confluent t2/flair high lesion in corpus callosum splenium and bilateral parietooccipital white matter (lt  rt).  no significant change on f/u. 2. congloeration of multifocal ring shaped, ill-defined contrast enhancing lesions especially in left side of corpus callosum splenium and parietooccipital white matter. perilesional edema and mild mass effect.  (1) d/dx lymphoma and tbc granuloma  (2) tumefactive demyelinating disease, less likely. 3. no steno-occlusive lesion in intra/extracranial arteries.

### Output:
Diseases
* Demyelinating: Present
* Infection/inflammation: None
* Hemorrhage/vascular lesion: None
* Stroke/infarction: None
</EXAMPLE 2>

### Target Diseases: {final_bert_str}

### Input Radiology Report (brain MRI report): {row['processed_text']}

### Output:
"""
            output = llm_pipe(prompt, max_new_tokens=300, do_sample=False, num_beams=5)
            df.loc[idx, 'inference_output_noncancer'] = copy.deepcopy(output[0]['generated_text'])
            df.loc[idx, 'inference_output_noncancer'] = copy.deepcopy(prompt)
            del output
            del prompt
            torch.cuda.empty_cache()
    
    ## Finally save df
    df.to_csv(args.output_path)
    print(f"Done. Saved to {args.output_path}")


if __name__ == "__main__":
    main()
