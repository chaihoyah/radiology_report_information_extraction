# inference_pipeline.py

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, processors
from tokenizers.pre_tokenizers import PreTokenizer, ByteLevel as ByteLevelPretokenizer, Sequence as PretokenizerSequence
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    pipeline
)
from konlpy.tag import Mecab  # Only if you need custom morphological splitting
from tqdm import tqdm

# Import your LMModel from bert_model.py
from bert_model import LMModel
from custom_morpheme_aware_tokenizer import morphemeTokenizer
import copy

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TOKENIZERS_PARALLELISM'] = "false"


def text_preprocessing(text: str) -> str:
    """
    Basic preprocessing for newline, extra spaces, certain patterns, etc.
    """
    if pd.isna(text):
        return np.nan
    processed = re.sub(r'\n|\r|\t', '', text)
    processed = re.sub('_x000D_', '', processed)
    processed = re.sub(r' +', ' ', processed)
    processed = re.sub(r"\-+\>", "", processed)
    processed = processed.strip()
    processed = re.sub(r"[^가-힣A-Za-z0-9\/\\\+\-\(\)\.\,\#\!\?\'\`\:\n ]", "", processed)
    return processed.lower()


def batch_infer_bert(texts, tokenizer, model, max_length=256, batch_size=32):
    """
    Batch inference using BERT model to determine presence of each lesion.
    Returns a list of lists (or np.array) of 0/1 predictions for each text.
    """
    all_preds = []
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]
        encodings = tokenizer(
            batch_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        encodings = {k: v.to(model.device) for k, v in encodings.items()}

        with torch.no_grad():
            logits = model(encodings)  # shape: [batch_size, LABEL_SIZE]

        binary_logits = (logits >= 0.5).int().cpu().numpy()
        all_preds.extend(binary_logits)

    return np.array(all_preds)


def create_llm_pipeline(llm_path):
    """
    Creates a text generation pipeline using a local LLM directory.
    Adjust device map or dtype as needed.
    """
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path)
    # Ensure we have a pad token if not present
    if tokenizer_llm.pad_token_id is None:
        tokenizer_llm.pad_token_id = tokenizer_llm.eos_token_id

    llm_pipe = pipeline(
        "text-generation",
        model=llm_path,
        tokenizer=tokenizer_llm,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",  # or "cuda:0" if you want to specify 
        max_new_tokens=200,
        repetition_penalty=1.2,
        do_sample=False,
        num_beams=5
    )
    return llm_pipe


def build_prompt(chest_ct_report: str, predicted_lesions: list) -> str:
    """
    Build the instruction prompt for the LLM given the chest CT report and predicted lesions.
    Customize as needed.
    """
    example_prompt = f"""### System: You are a helpful radiologist specialized in analyzing radiology reports.

### Instruction: Describe what lesions are present and each of their locations based on the input chest CT report. Describe if any described target lesion is present; if lesion is present, describe matching locations using given location group keywords for each lesion. Write ‘None’ after the lesion if specific lesion is not present or negated. Also, write ‘None’ after location group if lesion is present but location information is not available. Check examples.

### Output template:
Lesions
* lesion: Present or None
    + Location: specified location group keyword for each lesion

<EXAMPLE 1>
### Input Radiology Report (Chest CT report): ct, chest low-dose history: 1. s/p breast conserving operation for left breast cancer on 2003. s/p right upper lobar wedge resection for lung metastasis on 2006. 2. compared with ct on 2010-10-28 limited evaluation d/t poor quality ldct without enhancement esp. soft tissue. 1. no significant change of multiple nodules in both lungs. no significant change of focal osteoslcerotic lesions in t12.  stationary metastasis. 2. no change of minimal focal fibrosis, linear atelectasis around suture material in the right upper lobe.  postoperation change, suggested. 3. no change of multiple small lymph nodes in both paratracheal area, right axillary area.

### Target Lesions : nodule, atelectasis, fibrosis

### Output:
Lesions
* nodule : Present
    + Location : Rt lung, Lt lung
* atelectasis : Present
    + Location : RUL
* fibrosis : Present
    + Location : RUL
</EXAMPLE 1>

<EXAMPLE 2>
### Input Radiology Report (Chest CT report): 1. lung cancer in lll, s/p rtx. 2. hcc, tace state. compared with last chest ct. 1. irregular patchy opacity with minimal focal fibrosis in medial portion of lll.  no significant change since last ct. r/o rt-induced lung fibrosis, mixed with unknown portion of viable residual lung cancer. 2. no change of small lymph nodes at right side of cardiophrenic angle and right side of ivc.  r/o nonspecific reactive lymph nodes. r/o stationary state of ln metastasis. 3. no change of small subpleural nodule in rul.(#2-28)  r/o inflammatory nodule, more likely than metastasis. 4. minimal rt-induced lung fibrosis upper medial portion of both lungs. 5. no remarkable finding in bony thorax. 6. new peripheral atelectasis in rml.  refer to liver ct for covered upper abdomen.

### Target Lesions : ggo, nodule, atelectasis, fibrosis

### Output:
Lesions
* ggo : None
* nodule : Present
    + Location : RUL
* atelectasis : Present
    + Location : RML
* fibrosis : Present
    + Location : Rt lung, Lt lung, LLL
</EXAMPLE 2>

### Input Radiology Report (Chest CT report): {chest_ct_report}

### Location group: Right lung, Left lung, Left Lingula, BLL, BUL, RUL, RML, RLL, LUL, LLL, Pleural

### Target Lesions : {", ".join(predicted_lesions)}

### Output:
"""
    return example_prompt.strip()


def main():
    parser = argparse.ArgumentParser(description="End-to-end BERT -> LLM Inference Pipeline for Chest CT.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV containing the chest CT data.")
    parser.add_argument("--text_column", type=str, default="note_text", help="Name of the column in CSV with raw text data.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the final pickled dataframe.")
    parser.add_argument("--bert_checkpoint", type=str, required=True, help="Path to the BERT checkpoint (.ckpt) to load.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the BERT tokenizer JSON file.")
    parser.add_argument("--llm_path", type=str, required=True, help="Path to the local LLM model (HuggingFace format).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for BERT inference.")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length for BERT tokenizer.")
    parser.add_argument(
        "--lesion_labels",
        type=str,
        nargs='+',
        default=[
            'pleural effusion', 'ggo', 'consolidation', 'nodules',
            'mass', 'atelectasis', 'bronchiectasis', 'fibrosis',
            'bronchial wall thickening', 'pleural thickening',
            'interstitial thickening', 'etc'
        ],
        help="List of lesion labels. Must match the output dimension of your model."
    )
    args = parser.parse_args()

    # 1. Read and preprocess data
    df = pd.read_csv(args.input_csv)
    # df = df.loc[:8, :] ## Sample run for code check
    df.dropna(subset=[args.text_column], inplace=True)
    df['processed_text'] = df[args.text_column].apply(text_preprocessing)
    df.reset_index(drop=True, inplace=True)
    print(f"Loaded data: {df.shape[0]} rows.")

    # 2. Load tokenizer from JSON
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
    # Optionally add custom morphological pre-tokenizer
    fast_tokenizer.pre_tokenizer = PretokenizerSequence([
        PreTokenizer.custom(morphemeTokenizer()),
        ByteLevelPretokenizer()
    ])

    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj)
    fast_tokenizer.unk_token = "[UNK]"
    fast_tokenizer.sep_token = "[SEP]"
    fast_tokenizer.pad_token = "[PAD]"
    fast_tokenizer.cls_token = "[CLS]"
    fast_tokenizer.mask_token = "[MASK]"

    # 3. Load the BERT Model Checkpoint
    model = LMModel.load_from_checkpoint(
        checkpoint_path=args.bert_checkpoint,
        strict=False
    )

    # 4. BERT Inference
    texts = df['processed_text'].tolist()
    all_preds = batch_infer_bert(
        texts,
        tokenizer=fast_tokenizer,
        model=model,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    # 5. Create LLM Pipeline
    llm_pipe = create_llm_pipeline(args.llm_path)

    # 6. For each sample, build an LLM prompt and run generation
    final_outputs = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="LLM Inference"):
        predicted_vector = all_preds[i]  # shape: [LABEL_SIZE]
        selected_lesions = []
        for label_idx, val in enumerate(predicted_vector):
            if val == 1 and args.lesion_labels[label_idx] != 'etc':
                lesion = args.lesion_labels[label_idx]
                # small rename example
                if lesion == 'nodules':
                    lesion = 'nodule'
                selected_lesions.append(lesion)

        # default to 'etc' if no other lesion is found
        if len(selected_lesions) == 0:
            selected_lesions = ['etc']

        prompt = build_prompt(row['processed_text'], selected_lesions)
        llm_result = llm_pipe(prompt, max_new_tokens=200, do_sample=False, num_beams=5)
        final_outputs.append(copy.deepcopy(llm_result[0]['generated_text']))
        del prompt
        del llm_result
        torch.cuda.empty_cache()
    df['final_output'] = final_outputs

    # 7. Save the final DataFrame once
    df.to_csv(args.output_path)
    print(f"Final results saved to {args.output_path}")


if __name__ == "__main__":
    main()
