import pytorch_lightning as pl
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="../conf/citrinet/", config_name="config_bpe")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_rw_conformer_ctc_large")

    # asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)
    asr_model = nemo_asr.models.EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

    # Initialize the weights of the model from another model, if provided via config
    # asr_model.maybe_init_from_pretrained_checkpoint(cfg)

    # load ssl checkpoint
    asr_model.load_state_dict(model.state_dict(), strict=False)

    del model

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()
