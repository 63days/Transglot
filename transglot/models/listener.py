import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os.path as osp

from transglot.models.point_encoder.pointnet import *
from transglot.models.point_encoder.pointnet2 import *
from transglot.models.point_encoder.point_transformer import PT
from transglot.models.point_encoder.pct import PCT

from transglot.models.encoders import LanguageEncoder, PretrainedFeatures
from transglot.models.neural_utils import MLPDecoder, smoothed_cross_entropy, MultiHeadAttention
from transglot.in_out.geometry import vgg_image_features, pc_ae_features
from transglot.in_out.shapeglot_dataset import ShapeglotWithPCDataset
from transglot.in_out.rnn_data_preprocessing import make_dataset_for_rnn_based_model
from transglot.simple_utils import unpickle_data


class BaseListener(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._build_model()
        self.incorrect_indices = []

    def _build_model(self):
        raise NotImplementedError

    def forward(self, chairs, chairs_idx, tokens, dropout_rate=0.5):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        chairs, chairs_idx, targets, tokens = batch
        outputs = self(chairs, chairs_idx, tokens)
        loss = smoothed_cross_entropy(outputs['logits'], targets, 0)
        if self.hparams["use_tnet"]:
            mat_diff_loss = self.get_align_loss()
            loss += 1e-3 * mat_diff_loss
        preds = torch.max(outputs['logits'], 1)[1]
        acc = (targets == preds).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc * 100, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        chairs, chairs_idx, targets, tokens = batch
        outputs = self(chairs, chairs_idx, tokens)
        loss = smoothed_cross_entropy(outputs["logits"], targets, 0)
        if self.hparams["use_tnet"]:
            mat_diff_loss = self.get_align_loss()
            loss += 1e-3 * mat_diff_loss
        preds = torch.max(outputs["logits"], 1)[1]
        acc = (targets == preds).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc * 100, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        chairs, chairs_idx, targets, tokens = batch
        outputs = self(chairs, chairs_idx, tokens)
        loss = smoothed_cross_entropy(outputs['logits'], targets, 0)
        if self.hparams["use_tnet"]:
            mat_diff_loss = self.get_align_loss()
            loss += 1e-3 * mat_diff_loss
        preds = torch.max(outputs['logits'], 1)[1]
        check_correct = (targets==preds)
        acc = check_correct.float().mean()
        incorrect_idx = (check_correct==False).nonzero(as_tuple=True)[0]
        incorrect_idx = incorrect_idx + batch_idx * self.hparams["batch_size"]
        self.incorrect_indices += incorrect_idx.tolist()
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc * 100, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"],
                                      weight_decay=self.hparams["weight_decay"])
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

        return optimizer
#         return {'optimizer': optimizer,
#                 'lr_scheduler': {
#                     'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(self.train_ds),
#                                                                                 eta_min=0, last_epoch=-1),
#                     'interval': 'step',
#                     'frequency': 1
#                      }
#                     }

    def get_align_loss(self):
        mat_diff = torch.matmul(self.pc_encoder.transformk, self.pc_encoder.transformk.transpose(1, 2)) - \
                   torch.eye(64).cuda()
        label_mat = torch.eye(64, dtype=torch.float32, device='cuda').expand(mat_diff.size(0), -1, -1)
        mat_diff_loss = F.mse_loss(mat_diff, label_mat)
        return mat_diff_loss

    def prepare_data(self):
        cur_dir = osp.dirname(__file__)
        top_data_dir = osp.join(cur_dir, '../../data/main_data_for_chairs')
        data_name='game_data.pkl'
        game_data, word_to_int, int_to_word, int_to_sn_model, sn_model_to_int, sorted_sn_models = \
            unpickle_data(osp.join(top_data_dir, data_name))

        max_seq_len = 33  # Maximum size (in tokens) per utterance.
        split_sizes = [0.8, 0.1, 0.1]  # Train-val-test sizes.
        random_seed = 2004
        unique_test_geo = True  # If true, the test/train/val splits have 'targets' that are disjoint sets.
        only_correct = True  # Drop all not correctly guessed instances.

        net_data, split_ids, _, net_data_mask = \
            make_dataset_for_rnn_based_model(game_data,
                                             split_sizes,
                                             max_seq_len,
                                             drop_too_long=True,
                                             only_correct=only_correct,
                                             unique_test_geo=unique_test_geo,
                                             replace_not_in_train=True,
                                             geo_condition=None,
                                             bias_train=False,
                                             seed=random_seed,
                                             only_easy=True)

        self.train_ds = ShapeglotWithPCDataset(net_data['train'], num_points=self.hparams["num_points"])
        self.val_ds = ShapeglotWithPCDataset(net_data['val'], num_points=self.hparams["num_points"])
        self.test_ds = ShapeglotWithPCDataset(net_data['test'], num_points=self.hparams["num_points"])
        self.int_to_sn_model = int_to_sn_model
        self.vocab_size = len(int_to_word)

    def _build_dataloader(self, ds, mode):
        return DataLoader(ds,
                          batch_size=self.hparams["batch_size"],
                          shuffle=mode=='train',
                          num_workers=4,
                          pin_memory=True,
                          drop_last= mode=='train')

    def train_dataloader(self):
        return self._build_dataloader(self.train_ds, 'train')

    def val_dataloader(self):
        return self._build_dataloader(self.val_ds, 'val')

    def test_dataloader(self):
        return self._build_dataloader(self.test_ds, 'test')

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        if running_train_loss is not None:
            avg_training_loss = running_train_loss.cpu().item()
        else:
            avg_training_loss = float('NaN')
        items['loss'] = f'{avg_training_loss:.4f}'
        items.pop('v_num', None)
        return items


class Transglot(BaseListener):

    def _build_model(self):
        self.prepare_data()
        self.hidden_dim = self.hparams["hidden_dim"]

        attn_dim = self.hidden_dim // 2

        self.cross_attn_layers = nn.ModuleList()
        for i in range(self.hparams["attn_layers"]):
            self.cross_attn_layers.append(MultiHeadAttention(n_head=self.hparams["num_heads"],
                                                             query_dim=attn_dim,
                                                             point_dim=self.hparams["point_output_dim"],
                                                             d_k=attn_dim // self.hparams["num_heads"],
                                                             d_v=attn_dim // self.hparams["num_heads"]))
        mlp_dim = self.hparams["hidden_dim"]

        self.language_encoder_attn = LanguageEncoder(n_hidden=self.hidden_dim//2,
                                                     embedding_dim=self.hparams["embedding_dim"],
                                                     vocab_size=self.vocab_size)

        self.language_encoder_concat = LanguageEncoder(n_hidden=self.hidden_dim//2,
                                                       embedding_dim=self.hparams["embedding_dim"],
                                                       vocab_size=self.vocab_size)

        pc_encoder_type = self.hparams["pc_encoder_type"]
        pc_output_dim = self.hparams["point_output_dim"]

        if pc_encoder_type == "pn":
            self.pc_encoder = PN(output_dim=pc_output_dim)
        elif pc_encoder_type == "pn2ssg":
            self.pc_encoder = PN2SSG(output_dim=pc_output_dim)
        elif pc_encoder_type == "pn2msg":
            self.pc_encoder = PN2MSG(output_dim=pc_output_dim)
        elif pc_encoder_type == "pt":
            self.pc_encoder = PT(output_dim=pc_output_dim)
        elif pc_encoder_type == "pct":
            self.pc_encoder = PCT(output_dim=pc_output_dim)

        if self.hparams["use_image"]:
            top_pretrained_feat_dir = "./data/main_data_for_chairs/pretrained_features"
            vgg_feats_file = osp.join(top_pretrained_feat_dir, 'shapenet_chair_vgg_fc7_embedding.pkl')
            vgg_feats = vgg_image_features(self.int_to_sn_model, 'chair', vgg_feats_file, python2_to_3=True)
            self.image_encoder = PretrainedFeatures(torch.Tensor(vgg_feats),
                                                    embed_size=self.hidden_dim)

        self.logit_encoder = MLPDecoder(in_feat_dims=mlp_dim,
                                        out_channels=[100, 50, 1],
                                        use_b_norm=True)

    def forward(self, chairs, chairs_idx, tokens, dropout_rate=0.5):
        lang_feats = self.language_encoder_attn(tokens, init_feats=None)
        lang_concat_feats = self.language_encoder_concat(tokens, init_feats=None)

        # extract point cloud features #
        B, k, N, _ = chairs.size()
        chairs_group = chairs.contiguous().view(B*k, N, 3)
        pc_feats = self.pc_encoder(chairs_group)
        pc_feats = pc_feats.contiguous().view(B, k, -1, self.hparams["point_output_dim"]) #[B,3,num_point,point_output_dim]
        #################################

        logits = []
        attn_weights = [[] for _ in range(self.hparams["attn_layers"])] #[i][j]=i-th attn_layer's j-th object

        for i, l_feats in enumerate(lang_feats):
            l_f_attn = l_feats.unsqueeze(1)
            l_f_cat = lang_concat_feats[i]
            p_f = pc_feats[:,i]

            # Cross Attention j-iteration #
            attn_feat = l_f_attn
            for j in range(self.hparams["attn_layers"]):
                attn_feat, attn_weight = self.cross_attn_layers[j](q=attn_feat,
                                                                   k=p_f,
                                                                   v=p_f)
                attn_weight = attn_weight.squeeze(2) # [B,num_head,num_points]
                attn_weights[j].append(attn_weight)

            attn_feat = attn_feat.squeeze(1)
            final_feat = torch.cat([l_f_cat, attn_feat], 1)

            logits.append(self.logit_encoder(final_feat))
        
        outputs = dict()
        outputs["logits"] = torch.cat(logits, 1)
        outputs["attn_weights"] = attn_weights

        return outputs


class Shapeglot(BaseListener):

    def _build_model(self):
        self.prepare_data()
        top_pretrained_feat_dir = "./data/main_data_for_chairs/pretrained_features"
        vgg_feats_file = osp.join(top_pretrained_feat_dir, 'shapenet_chair_vgg_fc7_embedding.pkl')
        vgg_feats = vgg_image_features(self.int_to_sn_model, 'chair', vgg_feats_file, python2_to_3=True)
        pc_feats_file = osp.join(top_pretrained_feat_dir, 'shapenet_chair_pcae_128bneck_chamfer_embedding.npz')
        pc_feats = pc_ae_features(self.int_to_sn_model, pc_feats_file)

        if self.hparams["use_image"]:
            self.image_encoder = PretrainedFeatures(torch.Tensor(vgg_feats),
                                                    embed_size=self.hparams["hidden_dim"])

        self.language_encoder = LanguageEncoder(n_hidden=self.hparams["hidden_dim"],
                                                embedding_dim=self.hparams["embedding_dim"],
                                                vocab_size=self.vocab_size)

        if self.hparams["pretrained"]:
            self.pc_encoder = PretrainedFeatures(torch.Tensor(pc_feats),
                                                 embed_size=self.hparams["hidden_dim"])
        else:
            self.pc_encoder = PNCls(output_dim=self.hparams["point_output_dim"])

        self.logit_encoder = MLPDecoder(in_feat_dims=self.hparams["hidden_dim"]*2,
                                        out_channels=[100, 50, 1],
                                        use_b_norm=True)

    def forward(self, chair_pcs, chair_ids, padded_tokens, dropout_rate=0.5):
        if self.hparams["pretrained"]:
            logits = self._forward_pretrained(chair_ids, padded_tokens, dropout_rate)
        else:
            logits = self._forward_pointnet(chair_pcs, chair_ids, padded_tokens, dropout_rate)
        outputs = dict()
        outputs["logits"] = logits
        return outputs

    def _forward_pretrained(self, item_ids, padded_tokens, dropout_rate=0.5):
        if self.hparams["use_image"]:
            visual_feats = self.image_encoder(item_ids, dropout_rate)
            lang_feats = self.language_encoder(padded_tokens, init_feats=visual_feats)
        else:
            lang_feats = self.language_encoder(padded_tokens, init_feats=None)

        pc_feats = self.pc_encoder(item_ids, dropout_rate, pre_drop=False)

        logits = []
        for i, l_feats in enumerate(lang_feats):
            if pc_feats is not None:
                feats = torch.cat([l_feats, pc_feats[:, i]], 1)
            else:
                feats = l_feats

            logits.append(self.logit_encoder(feats))
        return torch.cat(logits, 1)

    def _forward_pointnet(self, chair_pcs, chair_ids, padded_tokens, dropout_rate=0.5):
        if self.hparams["use_image"]:
            visual_feats = self.image_encoder(chair_ids, dropout_rate)
            lang_feats = self.language_encoder(padded_tokens, init_feats=visual_feats)
        else:
            lang_feats = self.language_encoder(padded_tokens, init_feats=None)

        B, k, N, _ = chair_pcs.size()
        chairs_group = chair_pcs.contiguous().view(B * k, N, 3)
        pc_feats = self.pc_encoder(chairs_group)
        pc_feats = pc_feats.reshape(B, k, -1)

        logits = []
        for i, l_feats in enumerate(lang_feats):
            feats = torch.cat([l_feats, pc_feats[:, i]], 1)
            logits.append(self.logit_encoder(feats))

        return torch.cat(logits, 1)












