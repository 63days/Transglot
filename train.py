import pytorch_lightning as pl
import torch
import random
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from transglot.models.listener import Transglot
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import hydra
import omegaconf
import datetime

def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)

@hydra.main("config/config.yaml")
def main(cfg):
    hparams = hydra_params_to_dotdict(cfg)
    print(hparams)
    listener = Transglot(hparams)
    file_format = '{epoch}-{val_loss:.2f}-{val_acc:.1f}'

    version_format = datetime.datetime.now().strftime("%m%d-%H:%M")

    if hparams["log"]:
        tb_logger = TensorBoardLogger("logs", name=None, version=version_format)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_acc',
            mode="max",
            save_top_k=2,
            dirpath=f'checkpoints/{version_format}',
            filename=file_format,
            verbose=True,
            save_last=True
        )
    else:
        tb_logger = False
        checkpoint_callback = False

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=hparams["epochs"],
        callbacks=checkpoint_callback,
        logger=tb_logger
    )

    trainer.fit(listener)
    trainer.test(listener, ckpt_path=checkpoint_callback.best_model_path)

if __name__ == "__main__":
    random_seed = 63
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    main()
