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
# model.optim.sched.name="CosineAnnealing" \
# weight_decay was 3e-7


'''
need to comment d_model: ${model.encoder.d_model} in .yml to make the scheduler work
'''
    

python asr/finetune_ctc_bpe.py \
    --config-path=/data/chanie/asr/manifest \
    --config-name=conformer_ctc_bpe \
    exp_manager.name="Finetune_2" \
    exp_manager.resume_if_exists=true \
    exp_manager.resume_ignore_no_checkpoint=true \
    exp_manager.exp_dir=/data/chanie/asr/results/ \
    model.train_ds.batch_size=24 \
    model.train_ds.is_tarred=true \
    model.train_ds.tarred_audio_filepaths=$TRAIN_FILEPATHS \
    model.train_ds.manifest_filepath=$TRAIN_MANIFEST \
    model.validation_ds.manifest_filepath=$VAL_MANIFEST \
    model.test_ds.manifest_filepath=$TEST_MANIFEST \
    model.tokenizer.dir=$TOKENIZER \
    model.tokenizer.type="bpe" \
    model.optim.name="adamw" \
    model.optim.lr=3e-5 \
    model.optim.betas=[0.9,0.98] \
    model.optim.weight_decay=0 \
    model.optim.sched.name="CosineAnnealing" \
    trainer.accelerator="gpu" \
    trainer.precision=16 \
    trainer.strategy="ddp_find_unused_parameters_false" \
    trainer.max_epochs=70 \
    trainer.log_every_n_steps=50 \
    exp_manager.create_wandb_logger=true \
    exp_manager.wandb_logger_kwargs.name="Finetune_2" \
    exp_manager.wandb_logger_kwargs.project="KinASR" \
    # +model.optim.sched.monitor=val_loss \
    # +model.optim.sched.reduce_on_plateau=false