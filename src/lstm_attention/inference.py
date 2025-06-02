from pathlib import Path

import numpy as np
import torch
import yaml

from lstm_attention import augm, token
from lstm_attention.data import Data
from lstm_attention.model import LSTMAttention


def make_model(
    model_dir: str | Path,
    device: str = "cpu",
):
    params = yaml.safe_load(open(model_dir.joinpath("all_params.yml")))
    token_params = params["token"]
    max_length = token_params["max_length"]
    tokens = token_params["vocabulary"]
    model = LSTMAttention(**params["hyper_parameters"]["model"])
    model.load_state_dict(torch.load(model_dir.joinpath("model_params.pth")))
    model = model.to(device)
    return max_length, tokens, model


def list_to_array(
    smiles_list: list[str],
    tokens: list[str],
):
    if "*" in tokens:
        smiles_array = [
            smiles if (smiles is not None) and (smiles.count("*") >= 2) else None
            for smiles in smiles_list
        ]
    else:
        smiles_array = [
            smiles if (smiles is not None) and (smiles.count("*") == 0) else None
            for smiles in smiles_list
        ]
    return smiles_array

def make_dataset(
    smiles_list: list[str],
    tokens: list[str],
    max_length: int
):
    if "*" in tokens:
        smiles_array = [
            smiles if (smiles is not None) and (smiles.count("*") >= 2) else None
            for smiles in smiles_list
        ]
    else:
        smiles_array = [
            smiles if (smiles is not None) and (smiles.count("*") == 0) else None
            for smiles in smiles_list
        ]
    smiles_array = np.array(smiles_array)
    y = np.zeros(len(smiles_array))

    augmentated_smilse, enum_card, y = augm.data_augmentation(
        smiles_array,
        y,
        augmentation=True
    )
    tokenized_smiles = token.get_tokens(augmentated_smilse)
    token_tensor, y_tensor = token.convert_to_int_tensor(
        tokenized_smiles,
        y,
        max_length,
        tokens
    )
    dataset = Data(token_tensor.unsqueeze(0), y_tensor.view(-1, 1))
    return dataset, enum_card


def inference(
    model_dir: str | Path,
    smiles_list: list[str],
    device: str = "cpu",
):
    if type(model_dir) is str:
        model_dir = Path(model_dir)
    max_length, tokens, model = make_model(model_dir, device)
    smiles_array = list_to_array(smiles_list, tokens)
    y = np.zeros(len(smiles_array))
    y_pred_list = []
    dataset, enum_card = make_dataset(smiles_list, tokens, max_length)
    y_pred_list = []
    results = {smiles: None for smiles in smiles_list}
    model.eval()
    with torch.no_grad():
        for X, _ in dataset:
            X = X.to(device)
            y_pred = model(X)
            y_pred_list.extend(y_pred.cpu().numpy().flatten())
    y_pred_list, _ = augm.mean_std_result(enum_card, y_pred_list)
    for smiles, y in zip(smiles_array, y_pred_list):
        results[smiles] = float(y)
    return results
