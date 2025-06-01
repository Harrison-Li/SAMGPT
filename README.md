# SAMGPT

This is my final year project,works on inverse design of self-assembled-monolayers (Also other types of molecule if you want)

**prop_finetuning**\
This file contains codes for both struture and property conditional generation.\
Compared with original MolGPT, I added an adapter module and classfier-free guidance allow users finetuning the model with a mixure of data (Structures with property or not). This method allows the model both incorperate the information of pre-train learned structure and finetuning property values.

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

## Generated structures (-5.3 eV HOMO, -1.02 eV LUMO)
![SAM Candidates](<SAM candidates/candidates.jpg>)
![SAM Candidates](<SAM candidates/candidates2.jpg>)
