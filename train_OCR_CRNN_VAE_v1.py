# import sys
# sys.path.append("/mnt/ai_filestore/home/cain/code/github/lib-ocr")
from ocr.cannet.utils.alignCollate import alignCollate
from ocr.cannet.utils.imageDataset import imageDataset
import ocr.cannet.utils.levenshtein as levenshtein
import ocr.cannet.utils.loadConfig as loadConfig
import ocr.cannet.utils.crnn_utils as crnnUtils
from jadec.sequencecodec import SequenceCodec
# from utils.parse_data_from_txt import parse_data_from_txt

import torch
# sys.path.append("/mnt/ai_filestore/home/cain/code/train_utils")
from train_utils.trainer import Trainer
# from data import Cinnamon_OCR
import cv2
import numpy as np
import editdistance


learning_rate = None
DEFAULT_CONFIG = "/data/cain/src/lib-ocr/ocr/cannet/configs/CannetOCR_default_configs.ini"
TRAIN_SIZE = 500

#TODO: load data
# datapath = r"D:\Workspace\cinnamon\data\ocr\showa_mini_1k2\showa_mini_1k2\showa_mini_1k_train.txt"
# dataroot = r"D:\Workspace\cinnamon\data\ocr\showa_mini_1k2\showa_mini_1k2"

# x, y = parse_data_from_txt(datapath, root_dir=dataroot)
from IAMdataset import parse_IAM_data

datapath = "/data/cain/data/IAM/lines.txt"
x_sup, y_sup, x_unsup, y_unsup, x_val, y_val = parse_IAM_data(datapath, train_size=TRAIN_SIZE)

#TODO: load dictionary
with open("IAM_charset.txt", "r", encoding="utf-8") as f:
    alphabet = f.read()

config = {
    "alpha": 1,
    "beta": 1,
    "nrow": 1,
    "pretrain": "CRNN_RF_VAE_pretrain.pt",
    "converter_Trad": crnnUtils.strLabelConverter(alphabet, ctc_blank='-'),
    "debug_freqz": 250,
    "evaluate_freqz": 500,
    "n_epochs": 1000,
}

# load CRNN
# from lucas.core import CRNN_RF
# from backbones import Decoder
# model = CRNN_RF(droput_p=0.5, nh=256, nclass=nclass_Trad)

# sys.path.append(r"D:\Workspace\cinnamon\code\github\IDEA_OCR\CODE\cain_robert_ssl_ocr\dev\textline_VAE_CNN_recognition")
from vae_cnn import Encoder, Decoder
from blstm_ctc import CRNN_RF
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import tqdm


class OCR_Trainer(Trainer):

    def fit(self, dataloader, valloader=None, output="model.pt"):
        if not isinstance(dataloader, list):
            print(f"Input should be a list which contains dataloaders for supervised and unsupervised datasets")
            return None

        self.writer = SummaryWriter()

        for e in range(self.epoch_start, self.n_epochs):
            with tqdm.trange(len(dataloader[0]), ascii=True) as t:
                
                for iteration in t:
                    #TODO: activate train mode
                    if isinstance(self.model, dict):
                        for key, value in self.model.items():
                            value.train()
                    else:
                        self.model.train()

                    #TODO: load supervised data batch
                    try:
                        sup_data = sup_data_iter.next()
                    except:
                        sup_data_iter = iter(dataloader[0])
                        sup_data = sup_data_iter.next()

                    #TODO: load unsupervised data batch
                    try:
                        unsup_data = unsup_data_iter.next()
                    except:
                        unsup_data_iter = iter(dataloader[1])
                        unsup_data = unsup_data_iter.next()

                    #TODO: infer and back propagation
                    out = self._infer([sup_data, unsup_data])

                    #TODO: printing to screen
                    t.set_description(f"[{e}]" + out.get("screen", ""))

                    #TODO: adding scalar
                    if out.get("scalar"):
                        for criterion, value in out.get("scalar").items():
                            self.writer.add_scalar(f'{criterion}', value, e*len(dataloader[0]) + iteration)

                    #TODO: adding image
                    if out.get("image"):
                        if ((e*len(dataloader[0]) + iteration + 1) % self.debug_freqz) == 0:
                            for name, value in out.get("image").items():
                                grid = make_grid(value, nrow=self.nrow, normalize=True, range=(-1, 1))
                                self.writer.add_image(name, grid, e*len(dataloader[0]) + iteration + 1)

                    #TODO: adding embedding
                    if out.get("embedding"):
                        if ((e*len(dataloader[0]) + iteration + 1) % self.debug_freqz) == 0:
                            embedding_info = out.get('embedding')
                            self.writer.add_embedding(embedding_info.get('embedding'), metadata=embedding_info.get('metadata'), global_step=e*len(dataloader[0]) + iteration + 1)

                    #TODO: evaluating
                    if ((e*len(dataloader[0]) + iteration + 1) % self.evaluate_freqz) == 0:
                        if valloader is not None:
                            val_out = self.evaluate(valloader)

                            #TODO: adding scalar
                            if val_out.get("scalar"):
                                for criterion, value in val_out.get("scalar").items():
                                    self.writer.add_scalar(f'val_{criterion}', value, e*len(dataloader[0]) + iteration)

                            #TODO: adding image
                            if val_out.get("image"):
                                for name, value in val_out.get("image").items():
                                    grid = make_grid(value, nrow=self.nrow, normalize=True, range=(-1, 1))
                                    self.writer.add_image("val_" + name, grid, e*len(dataloader[0]) + iteration + 1)
            
            # #TODO: add scheduler
            # self.scheduler

            # store weights
            if isinstance(self.model, dict):
                store_dict = {}
                for key, value in self.model.items():
                    store_dict.update({
                        key: value.state_dict(),
                    })
                torch.save(store_dict, output)

            else:
                torch.save(self.model.state_dict(), output)

        self.writer.close()

    def _infer(self, list_data, mode="train"):
        #TODO: init
        loss = 0

        if mode == "train":
            #TODO: load unsupervised data <==============================
            unsup_data = list_data[1]
            #TODO: fetch into device
            images = unsup_data[0].to(self.device)
            ## padding if needed for divisible by 4
            if images.shape[3] % 8 != 4:
                right_pad = (4 - images.shape[3] % 8) % 8
                pad = nn.ZeroPad2d((0, right_pad, 0, 0))
                images = pad(images)

            #TODO: inference
            # reconstruction
            zs, z_dist = self.model["encoder"](images)
            reconstructed = self.model["decoder"](zs, self.model["encoder"].max_pool_idx, batch_size=images.shape[0])
            unsup_recon_loss = self.criteria['ReconLoss'](reconstructed.view(images.shape), images)

            # VAE speciality
            mu = z_dist["mean"]
            logvar = z_dist["logvar"]
            unsup_KL_loss =  -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            unsup_recon_loss = 0
            unsup_KL_loss = 0





        #TODO: load supervised data <=================================
        data = list_data[0]
        #TODO: fetch into device
        images = data[0].to(self.device)


        ## padding if needed for divisible by 4
        if images.shape[3] % 8 != 4:
            right_pad = (4 - images.shape[3] % 8) % 8
            pad = nn.ZeroPad2d((0, right_pad, 0, 0))
            images = pad(images)
        
        texts_Trad = data[1]
        texts_radical = data[2]

        text_Trad = data[3].type(torch.LongTensor).to(self.device)
        length_Trad = data[4].type(torch.LongTensor).to(self.device)

        batch_size = images.shape[0]


        #TODO: inference
        # reconstruction
        zs, z_dist = self.model["encoder"](images)
        reconstructed = self.model["decoder"](zs, self.model["encoder"].max_pool_idx, batch_size=images.shape[0])
        recon_loss = self.criteria['ReconLoss'](reconstructed.view(images.shape), images)

        # VAE speciality
        mu = z_dist["mean"]
        logvar = z_dist["logvar"]
        KL_loss =  -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Recognition
        preds_Trad = self.model["recognition"](zs.view(batch_size, -1, zs.shape[-1]).transpose(1, 2))
        preds_Trad = F.log_softmax(preds_Trad, dim=2)

        preds_size_Trad = torch.LongTensor([preds_Trad.size(0)] * batch_size)
        ctc_loss = self.criteria.get('CTCLoss')(preds_Trad, text_Trad, preds_size_Trad, length_Trad) / batch_size


        #TODO: loss
        loss = 784 * (unsup_recon_loss) + (unsup_KL_loss) * self.model["encoder"].latent_size + ctc_loss
        if mode == "train":
            loss_info = {
                'TotalLoss': loss.item(),
                'RecLoss': ctc_loss.item(),
                'KLLoss': KL_loss.item(),
                'KLLoss_unsup': unsup_KL_loss.item(),
                'ReconLoss': recon_loss.item(),
                'ReconLoss_unsup': unsup_recon_loss.item(),
            }
        else:
            loss_info = {
                'TotalLoss': loss.item(),
                'RecLoss': ctc_loss.item(),
                'KLLoss': KL_loss.item(),
                'ReconLoss': recon_loss.item(),
            }


        #TODO: back-propagation
        if mode == "train":
            self.optimizer['encoder'].zero_grad()
            self.optimizer['decoder'].zero_grad()
            self.optimizer['recognition'].zero_grad()

            if crnnUtils.loss_isvalid(loss) and not torch.equal(loss, torch.tensor(0.0).to(self.device)):
                loss.backward()

            # Clip gradient to avoid misbehaved CTC loss
            clipping_value = 1.0
            for mname, mfunc in self.model.items(): 
                torch.nn.utils.clip_grad_norm_(mfunc.parameters(), clipping_value)

            self.optimizer['encoder'].step()
            self.optimizer['decoder'].step()
            self.optimizer['recognition'].step()

        #TODO: accuracy
        # acc_info = self._accuracy(preds_Trad, texts_Trad)

        preds_Trad = preds_Trad.permute(1, 0, 2)        # [batch, width, nclass_Trad]
        batch_size = preds_Trad.shape[0]
        preds_size_Trad = torch.LongTensor([preds_Trad.size(1)] * batch_size)

        _, preds_Trad = preds_Trad.max(2, keepdim=True)
        preds_Trad = preds_Trad.squeeze(2)

        preds_Trad = preds_Trad.view(-1)
        
        pred_texts_Trad = self.converter_Trad.decode(preds_Trad.data.cpu(), preds_size_Trad.data, raw=False)
        pred_texts_Trad = pred_texts_Trad if isinstance(pred_texts_Trad, list) else [pred_texts_Trad]
        
        n_error = 0
        n_correct = 0
        tchar = 0
        tmaxchar = 0
        for pred, target in zip(pred_texts_Trad, texts_Trad):
            dis = editdistance.eval(pred, target)
            sim, _ = levenshtein.similarity(pred, target, distance=dis)

            tchar += len(target)
            tmaxchar += max(len(pred), len(target))
            n_correct += sim
            n_error += dis

        acc_info = {
            "tchar": tchar,
            "tmaxchar": tmaxchar,
            "n_correct": n_correct,
            "n_error": n_error,
        }

        if mode == "train":
            # acc calculation
            # AC3 = 1.0-n_error/float(tchar)
            ACC = acc_info.get('n_correct')/float(acc_info.get('tchar'))
            CER = acc_info.get('n_error')/float(acc_info.get('tchar'))

            loss_info.update({
                "ACC": ACC,
                "CER": CER,
            })

        elif mode == "val":
            loss_info.update(acc_info)

        #TODO: adding info to print
        tprint = ""
        if mode == "train":
            for criterion, loss_value in loss_info.items():
                tprint += f"| {criterion}[{loss_value:.4f}]"

            # tprint += "%s: %.5f |" % ("lr", lr)

        return {
            "screen": tprint,
            "scalar": loss_info,
            "image": {
                "input": images[:5].detach().cpu(),
                "reconstruct": reconstructed[:5].detach().cpu(),
            }
        }

    def _loss(self, preds_Trad, text_Trad, length_Trad, images, reconstructed):
        batch_size = preds_Trad.shape[1]

        # Permute for width first order
        # preds_Trad = preds_Trad.permute(1, 0, 2)        # [width, batch, nclass_Trad]

        # Log softmax for pytorch's CTCloss
        preds_Trad = F.log_softmax(preds_Trad, dim=2)

        
        preds_size_Trad = torch.LongTensor([preds_Trad.size(0)] * batch_size)
        ctc_loss = self.criteria.get('CTCLoss')(preds_Trad, text_Trad, preds_size_Trad, length_Trad) / batch_size

        #TODO: reconstruction loss
        min_w = min(reconstructed.shape[-1], images.shape[-1])
        recon_loss = self.criteria.get("ReconLoss")(reconstructed[:,:,:,:min_w], images[:,:,:,:min_w])

        loss = ctc_loss + recon_loss

        loss_info = {
            "RecLoss": recon_loss.item(),
            "CTCLoss": ctc_loss.item(),
            "TotalLoss": loss.item(),
        }

        return loss, loss_info

    def _accuracy(self, preds_Trad, texts_Trad):
        preds_Trad = preds_Trad.permute(1, 0, 2)        # [batch, width, nclass_Trad]
        batch_size = preds_Trad.shape[0]
        preds_size_Trad = torch.LongTensor([preds_Trad.size(1)] * batch_size)

        _, preds_Trad = preds_Trad.max(2, keepdim=True)
        preds_Trad = preds_Trad.squeeze(2)

        preds_Trad = preds_Trad.view(-1)
        
        pred_texts_Trad = self.converter_Trad.decode(preds_Trad.data.cpu(), preds_size_Trad.data, raw=False)

        n_error = 0
        n_correct = 0
        tchar = 0
        tmaxchar = 0
        for pred, target in zip(pred_texts_Trad, texts_Trad):
            dis = editdistance.eval(pred, target)
            sim, _ = levenshtein.similarity(pred, target, distance=dis)

            tchar += len(target)
            tmaxchar += max(len(pred), len(target))
            n_correct += sim
            n_error += dis

        return {
            "tchar": tchar,
            "tmaxchar": tmaxchar,
            "n_correct": n_correct,
            "n_error": n_error,
        }

    def evaluate(self, dataloader):
        if isinstance(self.model, dict):
            for mname, mmodel in self.model.items():
                mmodel.eval()
        else:
            self.model.eval()
        out = []
        for data in val_loader:
            out.append(self._infer([data], mode="val"))

        # ACC calculation
        n_correct = sum([item.get('scalar').get('n_correct') for item in out])
        n_error = sum([item.get('scalar').get('n_error') for item in out])
        tchar = sum([item.get('scalar').get('tchar') for item in out])
        tmaxchar = sum([item.get('scalar').get('tmaxchar') for item in out])
        loss = np.mean([item.get('scalar').get('TotalLoss') for item in out])

        ACC = n_correct / float(tchar)
        CER = n_error / float(tchar)

        print("====================== EVALUATE ===============")
        print(f"{loss} | ACC:[{ACC}] | CER: [{CER}] | {n_correct, n_error}/{tchar}")

        return {
            "scalar": {
                "ACC": ACC,
                "CER": CER,
                "TotalLoss": loss,
            }
        }






if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F

    import argparse
    parser = argparse.ArgumentParser(description='Process OCR training.')
    parser.add_argument('--__normalizeText_pre', default=lambda x: x,
                        help='sum the integers (default: find the max)')
    parser.add_argument('--batch_size', default=4,
                        help='batch size')
    parser.add_argument('--latent_size', default=512,
                        help='latent size')
    parser.add_argument('--learning_rate', default=0.0001,
                        help='learning rate')

    parser.add_argument('--n_workers', default=0,
                        help='sum the integers (default: find the max)')
                        

    args = parser.parse_args()

    transform = transforms.Compose([
                                transforms.ToTensor(),
                                ])
        
    dictConfig = loadConfig.loadconfig(DEFAULT_CONFIG)

    sc = SequenceCodec(all_char_list=alphabet,
                           encoding_name=dictConfig['JPCodec'],
                           space_code=dictConfig['cangjie_separated_char'],
                           ctc_code=dictConfig.get('ctc_blank_decomp', '˜'),
                           allowDuplicate=dictConfig['JPCodec_allowDuplicate'])
    alphabet_Decomp = sc.alphabet

    nclass_Trad = len(alphabet) + 1
    nclass_Decomp = len(alphabet_Decomp) + 1
    
    #load cannet packages
    # weights_path = r"D:\Workspace\cinnamon\code\weights\pretrain\ocr\ocr_cannetOCR_20190918_normalizePhase2_charset4685.pt"
    # package_dict = torch.load(weights_path)
    # with open("train/ocr/charset.txt", "w", encoding="utf-8") as f:
    #     f.write(package_dict.get('alphabet'))
    
    train_sup_dataset = imageDataset(impaths=x_sup, labels=y_sup, dictConfig=dictConfig,
                                     normalizeMethod=args.__normalizeText_pre, alphabet=alphabet,
                                     debug_OOM=dictConfig['debug_OOM'], include_radicaltarget=True,
                                     transform=None)
    
    train_sup_loader = DataLoader(
            train_sup_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
            pin_memory=True,
            collate_fn=alignCollate(imgH=dictConfig["img_height"], imgW=dictConfig["img_width"],
                                    keep_ratio=True, isUseBinary=dictConfig['use_binary'],
                                    isUseInverse=dictConfig['use_inverse'], inference_mode=False)
            )

    train_unsup_dataset = imageDataset(impaths=x_unsup, labels=y_unsup, dictConfig=dictConfig,
                                     normalizeMethod=args.__normalizeText_pre, alphabet=alphabet,
                                     debug_OOM=dictConfig['debug_OOM'], include_radicaltarget=True,
                                     transform=None)
    
    train_unsup_loader = DataLoader(
            train_unsup_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
            pin_memory=True,
            collate_fn=alignCollate(imgH=dictConfig["img_height"], imgW=dictConfig["img_width"],
                                    keep_ratio=True, isUseBinary=dictConfig['use_binary'],
                                    isUseInverse=dictConfig['use_inverse'], inference_mode=False)
            )

    val_dataset = imageDataset(impaths=x_val, labels=y_val, dictConfig=dictConfig,
                                     normalizeMethod=args.__normalizeText_pre, alphabet=alphabet,
                                     debug_OOM=dictConfig['debug_OOM'], include_radicaltarget=True,
                                     transform=None)

    val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
            pin_memory=True,
            collate_fn=alignCollate(imgH=dictConfig["img_height"], imgW=dictConfig["img_width"],
                                    keep_ratio=True, isUseBinary=dictConfig['use_binary'],
                                    isUseInverse=dictConfig['use_inverse'], inference_mode=False)
            )                             

    # load model
    model = {
        'encoder': Encoder(input_dim=[64, 64],
                                hidden_size=1,
                                latent_size=args.latent_size),
        "decoder": Decoder(input_dim=[64, 64],
                                hidden_size=1,
                                latent_size=args.latent_size),
        "recognition": CRNN_RF(nh=256, nclass=nclass_Trad),
    }
    
    if config.get('pretrain'):
        print(f"NOT READY !!!!! load {config.get('pretrain')}")
        # model.load_state_dict(torch.load(config.get('pretrain')))

    optimizers = {'encoder': optim.Adam(model["encoder"].parameters(), lr=args.learning_rate),
                 'decoder': optim.Adam(model["decoder"].parameters(), lr=args.learning_rate),
                 'recognition': optim.Adam(model["recognition"].parameters(), lr=args.learning_rate)}
    criteria = {
        "CTCLoss": nn.CTCLoss(
                    reduction="sum", zero_infinity=True),
        "ReconLoss": F.binary_cross_entropy,#nn.MSELoss(),
    }

    trainer = OCR_Trainer(model, optimizers, criteria, scheduler=None, config=config)
    trainer.fit([train_sup_loader, train_unsup_loader], valloader=val_loader, output=f"CRNN_RF_VAE_{TRAIN_SIZE}_ssl.pt")

    
    pass