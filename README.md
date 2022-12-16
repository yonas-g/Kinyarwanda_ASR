# Kinyarwanda STT GIZ

## Monolingual Data Preparation

- Convert the mp3 files to wav with sample rate of 16000
- Prepare JSON manifest file for NeMo training
  - clean the characters
  
## Multilinguage Data Creation
Please refere to [this link](https://github.com/NVIDIA/NeMo/blob/main/scripts/speech_recognition/code_switching/code_switching_audio_data_creation.py)
  

## Tokenization
NeMo provides a code to get the tokens as follows
```
python ${NEMO_ROOT}/scripts/tokenizers/process_asr_text_tokenizer.py \
  --manifest=dev_decoded_processed.json,train_decoded_processed.json \
  --vocab_size=1024 \
  --data_root=tokenizer_bpe_maxlen_4 \
  --tokenizer="spe" \
  --spe_type=bpe \
  --spe_character_coverage=1.0 \
  --spe_max_sentencepiece_length=4 \
  --log
```

## Tarring
To make the training faster, we can tar the dataset. This creates a new data where the sample are concatenated. Please refer to the link attached below for more information on how to tar your dataset

## Training
One the preprocessing and tokenization is done, we can train our model as follows.
```
TOKENIZER=tokenizers/tokenizer_spe_bpe_v1024_max_4/
TRAIN_MANIFEST=data/train_tarred_1bk/tarred_audio_manifest.json
TRAIN_FILEPATHS=data/train_tarred_1bk/audio__OP_0..1023_CL_.tar
VAL_MANIFEST=data/dev_decoded_processed.json
TEST_MANIFEST=data/test_decoded_processed.json

python ${NEMO_ROOT}/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
--config-path=../conf/conformer/ \
--config-name=conformer_ctc_bpe \
exp_manager.name="Some name of our experiment" \
exp_manager.resume_if_exists=true \
exp_manager.resume_ignore_no_checkpoint=true \
exp_manager.exp_dir=results/ \
model.tokenizer.dir=$TOKENIZER \
model.train_ds.is_tarred=true \
model.train_ds.tarred_audio_filepaths=$TRAIN_FILEPATHS \
model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
model.validation_ds.manifest_filepath=$VAL_MANIFEST \
model.test_ds.manifest_filepath=$TEST_MANIFEST
```

### Finetuning pretrained model
To finetune the available model, you've to add the following code snipper to your configuration yaml file.
```
init_from_pretrained_model:
  model0:
    name: "stt_rw_conformer_ctc_large"
    exclude: ["decoder"]
```
Here excluding the decoder allows you the flexibility to experiment with different tokenizers and sub-token values. If you don't want to change the sub-token values, you can set it to 128 and train without excluding the decoder.


## Reference
https://github.com/NVIDIA/NeMo/blob/main/docs/source/asr/examples/kinyarwanda_asr.rst
https://github.com/NVIDIA/NeMo/blob/main/scripts/speech_recognition/code_switching/code_switching_audio_data_creation.py
