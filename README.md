# SAMGPT

This is my final year project,works on inverse design of self-assembled-monolayers (Also other types of molecule if you want)

**prop_finetuning**\
This file contains code for both structure-conditional and property-conditional molecular generation.\
Compared with the original MolGPT, I introduced an adapter module and classifier-free guidance, which enable the model to be fine-tuned using a mixture of data (structures with or without property values (N/A)). This method allows the model to incorporate information from both the pre-trained structural features and the fine-tuned property values.

**scaf_finetuning**\
This file contains codes for just strcture constrained generation.

**salience_map**
This file contains codes for visualization of the generation. Highlighting next token with gradient colors to represent its probability to be generated.


## Dataset I used.

| Source                        | Dataset             | Samples          | Block Size (SMILES Len)         | Maximum Scaffold Length |
|-------------------------------|---------------------|------------------|---------------------------------|---------------------------|
| Zinc                          |       MOSES         | 1.9 million      | 54 (train), 51 (validation)      | 48                          |
| ChEMBL                        |      Guacamol       | 1.6 million      | 100 (train), 99 (validation)     | 100                       |
| 10.1021/acs.jcim.6b00340      | frontier orbitals   | 111,725 mols     | 148                              | 115                      |
| PSC literatures               | SAMs                | 200              | 202                              | 123                       |


## Training steps
1. Before getting the pre-trained model weight, remember to check the vocabulary of you dataset. Vocabulary size mismatch might leads to failure!
2. I suggest first traing the base model with MOSES and Guacamol with scaffold constrained. Each pre-trained has its different applications due to the distribution of SMILES length
3. Do scaffold finetuning (SAMs or frontier orbitals) or property finetuning (frontier orbitals).

## Script for finetuning.

```bash
nohup python ./prop_finetuning/cond_finetune.py 
  --run_name homo_lumo_Fine_tuning 
  --train_data_name full_prop_train_mix 
  --val_data_name full_prop_valid 
  --cond_props homo lumo 
  --pretrained_model_weight ./weights/scaffold_guacamol_all.pt 
  --output_ckpt homo_lumo_fine_tunning_mix.pt 
  --batch_size 200 
  --max_epochs 10 
  >> ./homo_lumo_Fine_tuning_mix.log 2>&1 &
```
## Script for conditional generation
```bash
nohup python ./prop_finetuning/scaf_prop_generate.py 
  --model_weight ./homo_lumo_fine_tunning_mix.pt 
  --scaffold 
  --csv_name gen_homo_lumo_mix.csv 
  --cond_props homo lumo 
  --gen_size 1000 
  --vocab_size 143 
  --block_size 202 
  --batch_size 200 
  >> ./prop_homo_lumo_generated_fine_tune.log 2>&1 &
```
## Model weight for reproducing

[Google Drive: Model Weights](https://drive.google.com/drive/folders/17zEuHTCbWWszKTqHvQ01PQsOClcYiG-t?usp=sharing)


## Property conditional generation results
### Single property (LogP)
![LogP_dis](<figures/logp_hist.jpg>)
### Multi-properties (LogP, TPSA)
![LogP_TPSA_dis](<figures/logp_tpsa_hist.jpg>)



## Generated structures (-5.3 eV HOMO, -1.02 eV LUMO)
![SAM Candidates](<SAM candidates/candidates.jpg>)
![SAM Candidates](<SAM candidates/candidates2.jpg>)
