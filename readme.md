# Project Overview



## Description

This project extracts lesion and disease information from CDM data of Chest CT and Brain MRI reports.

- **Chest CT:** Extracts information on lesions including 'pleural effusion', 'GGO', 'consolidation', 'nodules', 'mass', 'atelectasis', 'bronchiectasis', 'fibrosis', 'bronchial wall thickening', 'pleural thickening', and 'interstitial thickening'. It also maps the location information of detected lesions.
- **Brain MRI:** Extracts disease information such as 'Glioma', 'Lymphoma', 'Metastasis', 'Other intra-axial brain tumors', 'Other extra-axial brain tumors', 'Demyelinating', 'Infection/inflammation', 'Hemorrhage/vascular lesion', and 'Stroke/infarction'. If malignant tumors (Glioma, Lymphoma, Metastasis) are detected, their progression state is categorized into progression, stable, or improvement.

## Requirements

- Minimum **24GB VRAM GPU** is required.
- Execution time: Approximately **10 seconds per sample** for LLM inference.

## Trained BERT Models

[Download Link](#)

### Setup Instructions

1. **Download the models** from the provided link and organize them as follows:

   - Create the folder `trained_models/finetuned_bert/` and place the downloaded BERT checkpoint inside.
   - Create the folder `trained_models/tokenizer/` and place `tokenizer_0530.json` inside.

2. **Set up the environment** (Assuming Linux system):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run inference scripts**:

   - **Chest CT Inference Example:**

     ```bash
     python3 chest_inference_pipeline.py \
       --input_csv "chest_ct.csv" \
       --text_column note_text \
       --output_path "/inference_result/chest_output.csv" \
       --bert_checkpoint "./trained_models/finetuned_bert/cxrbert_amc_data_pretrain_finetune_chest_ct.ckpt" \
       --tokenizer_path "./trained_models/tokenizer/tokenizer_0530.json" \
       --llm_path "meta-llama/Meta-Llama-3-8B" \
       --batch_size 32 \
       --max_length 256
     ```

   - **Brain MRI Inference Example:**

     ```bash
     python3 brain_inference_pipeline.py \
       --input_csv "brain_mri.csv" \
       --finding_column note_text \
       --conclusion_column note_cnclsn \
       --output_path "./inference_result/brainmri_output.csv" \
       --bert_checkpoint "./trained_models/finetuned_bert/cxrbert_amc_data_pretrain_finetune_brain_mri.ckpt" \
       --tokenizer_path "./trained_models/tokenizer/tokenizer_0530.json" \
       --llm_path "meta-llama/Meta-Llama-3-8B" \
       --batch_size 32 \
       --max_length 256
     ```

4. **Verify output**:

   - If CSV files appear in the `output_path`, the process is complete.

