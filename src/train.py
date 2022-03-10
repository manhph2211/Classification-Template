from comet_ml import Experiment
import torch
import torch.nn as nn
from dataset import get_loader, IR_Dataset
from models.build import build_model
from engine import Trainer, training_experiment
from utils import get_config
from ema import EMA
from losses import LabelSmoothingCrossEntropy 


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfgs = get_config()
    with open('./experiment_apikey.txt','r') as f:
        api_key = f.read()
    experiment = Experiment(
        api_key = api_key,
        project_name = "IR Project",
        workspace = "maxph2211",
    )
    experiment.log_parameters(cfgs)

    model = build_model(cfgs).to(device)
    ema_model = EMA(model.parameters(), decay_rate=0.995, num_updates=0)
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion = LabelSmoothingCrossEntropy().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfgs['train']['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=3, verbose=True)
    train_loader, test_loader = get_loader(cfgs, IR_Dataset)
    trainer = Trainer(model, criterion, optimizer, ema_model, cfgs['train']["clip_value"], device=device)

    training_experiment(train_loader, test_loader, experiment, trainer, cfgs['train']['epoch_n'], scheduler)
    print("DONE!")
