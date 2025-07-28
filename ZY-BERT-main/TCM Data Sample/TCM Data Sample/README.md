The folder contains three files:

1. **TCM-SD_data_sample.json**: contains five examples from train set. Each example contains the following parts:
   - example_id: unique id for current example
   - lCD_ID & Name: explained in paper.
   - syndrome: original syndrome without syndrome normalization
   - chief_complaint: explained in paper.
   - medical_history: explained in paper.
   - FDMR: explained in paper.
   - norm_syndrome: syndrome after syndrome normalization
2. **TCM-SD_data_sample_with_knowledge.json:**  contains additional knowledge for the five examples presented in **TCM-SD_data_sample.json**:
   - knowledge_option: 5 syndromes used in MRC setting
   - knowledge_para: 5 syndromes with knowledge used in MRC setting.
3. **Knowledge_corpus_sample.txt**: contains five knowledge samples from constructed knowledge corpus.

