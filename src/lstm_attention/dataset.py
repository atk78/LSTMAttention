from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset

from lstm_attention import augm, token


CORES = 2


class Data(Dataset):
    def __init__(self, X: Tensor, prop: Tensor):
        self.X = X
        self.prop = prop

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.prop[index]


class SmilesDataset:
    original_datasets: dict[str, list[np.ndarray]]
    enum_cards: dict[str, list[int]]
    tokenized_datasets: dict[str, list[np.ndarray]]
    tensor_datasets: dict[str, Data]

    def __init__(
        self,
        smiles: np.ndarray,
        y: np.ndarray,
        augmentation: bool,
        batch_size=1,
        dataset_ratio=[0.8, 0.1, 0.1],
        random_state=42,
    ):
        if sum(dataset_ratio) != 1.0:
            raise RuntimeError("Make sure the sum of the ratios is 1.")
        # ========== 変数の設定 ==========
        self.smiles = smiles
        self.y = y
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.dataset_ratio = dataset_ratio
        self.random_state = random_state
        # ========= データセットの分割 ==========
        X_train, X_test, y_train, y_test = train_test_split(
            self.smiles,
            self.y,
            test_size=1 - self.dataset_ratio[0],
            shuffle=True,
            random_state=self.random_state,
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_test,
            y_test,
            test_size=self.dataset_ratio[2] / (self.dataset_ratio[2] + self.dataset_ratio[1]),
            shuffle=True,
            random_state=self.random_state,
        )
        X_train, self.enum_card_train, y_train = augm.data_augmentation(
            X_train, y_train, self.augmentation
        )
        X_valid, self.enum_card_valid, y_valid = augm.data_augmentation(
            X_valid, y_valid, self.augmentation
        )
        X_test, self.enum_card_test, y_test = augm.data_augmentation(
            X_test, y_test, self.augmentation
        )
        self.train_dataset = [X_train, y_train]
        self.valid_dataset = [X_valid, y_valid]
        self.test_dataset = [X_test, y_test]

    def split_dataset(
        self,
        validation_method: Literal["holdout", "cv"] = "holdout"
    ):
        datasets = dict()
        enum_cards = dict()
        # ========= バリデーション方法の設定 ==========
        # ホールドアウト法によるデータ分割
        if validation_method == "holdout":
            splited_datasets = {
                "train": self.train_dataset,
                "valid": self.valid_dataset,
                "test": self.test_dataset,
            }
            splited_enum_cards = {
                "train": self.enum_card_train,
                "valid": self.enum_card_valid,
                "test": self.enum_card_test
            }
        else:
            # k-fold法によるデータ分割
            X_train = np.concatenate(
                [
                    self.train_dataset[0],
                    self.valid_dataset[0]
                ], axis=0
            )
            y_train = np.concatenate(
                [
                    self.train_dataset[1],
                    self.valid_dataset[1]
                ], axis=0
            )
            enum_cards_train = self.enum_card_train + self.enum_card_valid
            splited_datasets = {
                "train": [X_train, y_train],
                "test": [self.test_dataset[0], self.test_dataset[1]]
            }
            splited_enum_cards = {
                "train": enum_cards_train,
                "test": self.enum_card_test
            }

        for phase, [X, y], enum_card in zip(
            splited_datasets.keys(),
            splited_datasets.values(),
            splited_enum_cards.values()
        ):
            datasets[phase] = [X, y]
            enum_cards[phase] = enum_card
        return datasets, enum_cards

    def tokenize_smiles(self, datasets: dict[str, list[np.ndarray]]):
        tokenized_smiles_datasets = dict()
        for phase, [smiles, y] in datasets.items():
            tokenized_smiles_datasets[phase] = [token.get_tokens(smiles), y]
        return tokenized_smiles_datasets

    def tensorize_datasets(
        self,
        max_length: int,
        tokenized_smiles_datasets: dict[str, list[np.ndarray]],
        tokens: list[str]
    ):
        tensor_datasets = dict()
        for phase, [tokenized_smiles, y] in tokenized_smiles_datasets.items():
            token_tensor, y_tensor = token.convert_to_int_tensor(
                tokenized_smiles, y, max_length, tokens
            )
            tensor_datasets[phase] = Data(X=token_tensor, prop=y_tensor)
        return tensor_datasets
