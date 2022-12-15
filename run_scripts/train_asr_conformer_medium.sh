TOKENIZER=/data/chanie/asr/manifest/tokenizer_bpe_maxlen_4/tokenizer_spe_bpe_v1024_max_4
TRAIN_MANIFEST=/data/chanie/asr/manifest/train_tarred_1bk/tarred_audio_manifest.json
TRAIN_FILEPATHS=/data/chanie/asr/manifest/train_tarred_1bk/audio__OP_0..1023_CL_.tar # ? How is this represented??
VAL_MANIFEST=/data/chanie/asr/manifest/dev_manifest_clean_decoded.json
TEST_MANIFEST=/data/chanie/asr/manifest/test_manifest_clean_decoded.json


# model.train_ds.manifest_filepath=/data/chanie/asr/manifest/train_manifest_clean_decoded.json \
# model.validation_ds.manifest_filepath=/data/chanie/asr/manifest/dev_manifest_clean_decoded.json \
# model.test_ds.manifest_filepath=/data/chanie/asr/manifest/test_manifest_clean_decoded.json \
# model.tokenizer.dir=/data/chanie/asr/manifest/tokenizer_bpe_maxlen_4/tokenizer_spe_bpe_v1024_max_4 \

#     model.optim.sched.name="ReduceLROnPlateau" \
    

python NeMo/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py \
    --config-path=/data/chanie/asr/manifest \
    --config-name=conformer_ctc_bpe \
    exp_manager.name="Kinyarwanda ASR Medium" \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.exp_dir=/data/chanie/asr/results/ \
    model.train_ds.batch_size=64 \
    model.train_ds.is_tarred=true \
    model.train_ds.tarred_audio_filepaths=$TRAIN_FILEPATHS \
    model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
    model.validation_ds.manifest_filepath=$VAL_MANIFEST \
    model.test_ds.manifest_filepath=$TEST_MANIFEST \
    model.encoder.d_model=256 \
    model.encoder.n_heads=4 \
    model.spec_augment.time_masks=5 \
    model.tokenizer.dir=$TOKENIZER \
    model.tokenizer.type="bpe" \
    model.optim.name="adamw" \
    model.optim.lr=1.0 \
    model.optim.betas=[0.9,0.98] \
    model.optim.weight_decay=3e-7 \
    trainer.accelerator="gpu" \
    trainer.precision=16 \
    trainer.strategy="ddp_find_unused_parameters_false" \
    trainer.max_epochs=120 \
    trainer.log_every_n_steps=50 \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.name="KinASR_Medium" \
    exp_manager.wandb_logger_kwargs.project="KinASR" \