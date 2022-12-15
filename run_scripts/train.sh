#!/bin/bash

# usrun.sh --gpus=4 --mem=64G --cpus-per-task=8 --tasks-per-node=4 --pty bash

# ${KIN_ASR} = /data/chanie/asr
#  \\\n -> \t
# TOKENIZER=/data/chanie/asr/manifest/tokenizer_bpe_maxlen_4/tokenizer_spe_bpe_v1024_max_4
TOKENIZER=/data/chanie/asr/manifest/tokenizer_bpe_maxlen_2/tokenizer_spe_bpe_v128_max_2
# TRAIN_MANIFEST=/data/chanie/asr/manifest/train_tarred_1bk/tarred_audio_manifest.json
TRAIN_MANIFEST=/data/chanie/asr/manifest/train_manifest_clean_decoded.json
TRAIN_FILEPATHS=/data/chanie/asr/manifest/train_tarred_1bk/audio_0..1023.tar # ? How is this represented??
VAL_MANIFEST=/data/chanie/asr/manifest/dev_manifest_clean_decoded.json
TEST_MANIFEST=/data/chanie/asr/manifest/test_manifest_clean_decoded.json

# ${NEMO_ROOT} = /data/chanie/NeMo
python /data/chanie/NeMo/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
    --config-path=/data/chanie/asr/manifest \
    --config-name=conformer_ctc_bpe \
    exp_manager.name="Kinyarwanda ASR" \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.exp_dir=/data/chanie/asr/results/ \
    model.train_ds.batch_size=128 \
    model.train_ds.is_tarred=false \
    model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
    model.validation_ds.manifest_filepath=$VAL_MANIFEST \
    model.test_ds.manifest_filepath=$TEST_MANIFEST \
    model.tokenizer.dir=$TOKENIZER \
    model.tokenizer.type="bpe" \
    trainer.devices= -1\
    trainer.accelerator="gpu" \
    trainer.precision=16 \
    trainer.strategy="ddp" \
    trainer.max_epochs=150 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000 \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="KinASR_1" \
    exp_manager.wandb_logger_kwargs.project="KinASR"