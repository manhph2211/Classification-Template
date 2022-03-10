from comet_ml import Experiment
import torch
import torch.nn as nn
from dataset import get_loader, IR_ContrasDataset
from models.build import build_model
from losses import ContrastiveLoss
from engine import Trainer, training_experiment
from utils import get_config


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfgs = get_config()
    with open('./experiment_apikey.txt','r') as f:
        api_key = f.read()
    experiment = Experiment(
        api_key = api_key,
        project_name = "Dog Cat Classification",
        workspace = "maxph2211",
    )
    experiment.log_parameters(cfgs)

    model = build_model(cfgs).to(device)

    criterion1 = ContrastiveLoss().to(device)
    criterion2 = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfgs['train']['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, verbose=True)
    train_loader, test_loader = get_loader(cfgs, IR_ContrasDataset)
    trainer = Trainer(model, criterion1, criterion2, optimizer,
                      cfgs['train']["loss_ratio"], cfgs['train']["clip_value"], device=device)

    training_experiment(train_loader, test_loader, experiment, trainer, cfgs['train']['epoch_n'], scheduler)
    print("DONE!")
