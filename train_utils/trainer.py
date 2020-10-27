from torchsummary import summary 
import torch
import tqdm
import sys
import os
# sys.path.append("/mnt/sda1/code/github/workspace/pix2pix")
# sys.path.append("/mnt/sda1/code/github/vq-vae-2-pytorch")
# from data import AlignedDataset
# from optimizer import get_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
# from scheduler import CycleScheduler


def normalize(img):
    return img * 2. - 1.

def invert_normalize(img):
    return (img + 1.) / 2


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Trainer:

    _default = {
        "nrow": 5,
        "curve_freqz": 10,
        "debug_freqz": 10,
        "evaluate_freqz": 500,
        'n_epochs': 300,
        'epoch_start': 0,
        'pretrain': '',
    }

    def __init__(self, model, optimizer, criteria, scheduler=None, config={}):
        self.__dict__.update(self._default)
        self.__dict__.update(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(model, dict):
            self.model = model
            for key, value in self.model.items():
                value = value.to(self.device)
        else:
            self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criteria = criteria

        for _, value in self.criteria.items():
            try:
                value = value.to(self.device)
            except:
                continue

        #TODO: load pretrain
        # if os.path.isfile(self.pretrain):
        #     print(f"Loading pretrain {self.pretrain}")
        #     self.model.load_state_dict(torch.load(self.pretrain))

    def _loss(self, pred, label):
        loss_info = {}
        loss = 0
        for criterion, cfunc in self.criteria.items():
            closs = cfunc(pred, label)
            loss += closs
            loss_info.update({
                criterion: closs.item(),
            })

        return loss, loss_info

    def _infer(self, data):
        #TODO: fetch into device
        input_ = data[0].to(self.device)
        label_ = data[1].to(self.device)

        #TODO: inference
        pred = self.model(input_)

        #TODO: loss
        loss, loss_info = self._loss(pred, label_)

        #TODO: back-propagation
        for opt, ofunc in self.optimizer.items():
            ofunc.zero_grad()
            loss.backward()

            ofunc.step()

        lr = ofunc.param_groups[0]["lr"]

        #TODO: adding info to print
        tprint = ""
        for criterion, loss_value in loss_info.items():
            tprint += f"{criterion, loss_value:.5}"

        tprint += "%s: %.5f |" % ("lr", lr)

        return {
            "screen": tprint,
            "scalar": loss_info,
            "image": {
                "mask": torch.cat((
                    torch.cat((input_, input_, input_), 1)[:self.nrow].detach().cpu(),
                    pred[:self.nrow].detach().cpu(),
                    label_[:self.nrow].detach().cpu(),
                ), 0)
            }
        }

    def evaluate(self, dataloader):
        pass


    def fit(self, dataloader, valloader=None, output="model.pt"):
        self.writer = SummaryWriter()

        for e in range(self.epoch_start, self.n_epochs):
            with tqdm.trange(len(dataloader), ascii=True) as t:
                # data_iter = iter(dataloader)
                for iteration in t:
                    #TODO: activate train mode
                    if isinstance(self.model, dict):
                        for key, value in self.model.items():
                            value.train()
                    else:
                        self.model.train()

                    try:
                        data = data_iter.next()
                    except:
                        data_iter = iter(dataloader)
                        data = data_iter.next()

                    #TODO: infer and back propagation
                    out = self._infer(data)

                    #TODO: printing to screen
                    t.set_description(f"[{e}]" + out.get("screen", ""))

                    #TODO: adding scalar
                    if out.get("scalar"):
                        for criterion, value in out.get("scalar").items():
                            self.writer.add_scalar(f'{criterion}', value, e*len(dataloader) + iteration)

                    #TODO: adding image
                    if out.get("image"):
                        if ((e*len(dataloader) + iteration + 1) % self.debug_freqz) == 0:
                            for name, value in out.get("image").items():
                                grid = make_grid(value, nrow=self.nrow, normalize=True, range=(-1, 1))
                                self.writer.add_image(name, grid, e*len(dataloader) + iteration + 1)

                    #TODO: adding embedding
                    if out.get("embedding"):
                        if ((e*len(dataloader) + iteration + 1) % self.debug_freqz) == 0:
                            embedding_info = out.get('embedding')
                            self.writer.add_embedding(embedding_info.get('embedding'), metadata=embedding_info.get('metadata'), global_step=e*len(dataloader) + iteration + 1)

                    #TODO: evaluating
                    if ((e*len(dataloader) + iteration + 1) % self.evaluate_freqz) == 0:
                        if valloader is not None:
                            val_out = self.evaluate(valloader)

                            #TODO: adding scalar
                            if val_out.get("scalar"):
                                for criterion, value in val_out.get("scalar").items():
                                    self.writer.add_scalar(f'val_{criterion}', value, e*len(dataloader) + iteration)

                            #TODO: adding image
                            if val_out.get("image"):
                                for name, value in val_out.get("image").items():
                                    grid = make_grid(value, nrow=self.nrow, normalize=True, range=(-1, 1))
                                    self.writer.add_image("val_" + name, grid, e*len(dataloader) + iteration + 1)
            
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
                
    def __repr__(self):
        return summary(self.model, input_size=(3, 224, 224))


if __name__ == "__main__":
    
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    from data import Cinnamon
    input_transform = transforms.Compose([transforms.Resize(286),
                                transforms.RandomCrop(256),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    label_transform = transforms.Compose([transforms.Resize(286),
                                transforms.RandomCrop(256),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

    additional_transform = transforms.Compose([
        transforms.ColorJitter(brightness=[0.9, 1.5], contrast=0.1, saturation=0.1, hue=0.01),
    ])

    trainset = Cinnamon(r"D:\Workspace\cinnamon\data\layout\Invoice_Train", 
                            transform=input_transform, 
                            target_transform=label_transform)

    trainloader = DataLoader(trainset, batch_size=32,
                        shuffle=True, num_workers=4)
    
    # from sample_net import ResNetUNet
    # model = ResNetUNet(n_class=3)


    # input_nc = 3
    # output_nc = 3
    # ngf = 64
    # from models import define_G
    # model = define_G(input_nc, output_nc, ngf, num_downs=8 ,norm="batch", netG=None, use_dropout=True)

    # from backbone import ARUNET
    # model = ARUNET(scale=1, class_=3, use_attetion=False, freeze_bn=False)
    # model.load_state_dict(torch.load("Jeff.pt"))


    optimizers = {
        'unet': torch.optim.Adam(model.parameters(), lr=0.0003)#, betas=(0.5, 0.999))
    }
    
    criteria = {
        # 'L1': torch.nn.L1Loss()
        "MSELoss": torch.nn.MSELoss(),
    }
    trainer = Trainer(model, optimizers, criteria, scheduler=None)
    trainer.fit(trainloader)

