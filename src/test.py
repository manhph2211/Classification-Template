import torch
import pandas as pd


def predict(test_loader, model, checkpoint, save_path):
    model.load_state_dict(torch.load(checkpoint))
    image_names = []
    predicts = []
    for images, names in test_loader:
        output = model(images)
        output = output.numpy()

        batch_predict = output.max(axis=1)
        image_names += names
        predicts += batch_predict

    result = {"ID": image_names, "label": predicts}
    df = pd.DataFrame.from_dict(result)
    df.to_csv(save_path)


if __name__ == '__main__':
    pass
