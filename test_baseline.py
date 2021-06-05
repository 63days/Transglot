import pytorch_lightning as pl
import os
import os.path as osp
from transglot.models.listener import Shapeglot
import omegaconf


def main():
    outputs_top_dir = "./outputs"

    # if you want to test your own trained model, change this to your sub_dir e.g. "0529-14:32"
    sub_ver_dir = "baseline"

    # find automatically best val_acc checkpoint in top-2
    best_acc = -1
    for file in os.listdir(osp.join(outputs_top_dir, f"checkpoints/{sub_ver_dir}")):
        if file[-4:] != "ckpt" or file == "last.ckpt":
            continue
        val_acc = float(file[-9:-5])
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = osp.join(outputs_top_dir, f"checkpoints/{sub_ver_dir}/{file}")

    # load hyper-parameter settings
    hparams_path = osp.join(outputs_top_dir,"logs", sub_ver_dir, "hparams.yaml")
    hparams = omegaconf.OmegaConf.load(hparams_path)

    # load the pre-trained model
    listener = Shapeglot.load_from_checkpoint(ckpt_path, hparams=hparams).cuda()
    for p in listener.parameters():
        p.requires_grad = False

    trainer = pl.Trainer(gpus=1,
                        checkpoint_callback=False,
                        logger=False)
    trainer.test(listener)


if __name__ == "__main__":
    main()


