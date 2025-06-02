import shutil
import os
import platform
import argparse
import warnings
import random
import shutil
from logging import Logger
from pathlib import Path
from typing import Literal

import yaml
import polars as pl
import numpy as np
import optuna
import torch

from lstm_attention.model import LSTMAttention
from lstm_attention.data import SmilesDataset
from lstm_attention.trainer import HoldOutTrainer, CrossValidationTrainer, EarlyStopping
from lstm_attention.evaluate import evaluate_model
from lstm_attention import token, utils


warnings.simplefilter("ignore")


class BayOptLoss:
    loss = None
    r2 = None
    number = 0


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def bayopt_hparams(
    output_dir: Path,
    bayopt_datasets: dict[str, SmilesDataset],
    logger: Logger,
    vocab_size: int,
    bayopt_bounds: dict,
    validation_method: Literal["holdout", "cv"] = "holdout",
    cv_n_splits=5,
    n_epochs=10,
    batch_size=32,
    n_trials=10,
    num_of_outputs=1,
    seed=42,
    device="cpu"
):
    """
    Optunaを用いてハイパーパラメータの探索を行う関数

    Parameters
    ----------
    output_dir : Path
        出力ディレクトリのPathオブジェクト
    bayopt_datasets : dict[str, SmilesDataset]
        学習用のデータセット
    logger : Logger
        ロガー
    vocab_size : int
        ボキャブラリのサイズ
    bayopt_bounds : dict
        ハイパーパラメータの探索範囲
    validation_method : Literal["holdout", "cv"], optional
        検証方法, by default "holdout"
    cv_n_splits : int, optional
        validation_method = "cv"の場合、クロスバリデーションの分割数, by default 5
    n_epochs : int, optional
        エポック数, by default 10
    batch_size : int, optional
        バッチサイズ, by default 32
    n_trials : int, optional
        ハイパーパラメータ探索の試行回数, by default 10
    num_of_outputs : int, optional
        出力の数, by default 1
    seed : int, optional
        乱数シード, by default 42
    device : str, optional
        使用するデバイス, by default "cpu"

    Returns
    -------
    best_hparameters : dict
        ベイズ最適化によって導出されたハイパーパラメータの辞書
    """
    bayopt_dir = output_dir.joinpath("bayes_opt")
    if bayopt_dir.exists():
        shutil.rmtree(bayopt_dir)
    bayopt_dir.mkdir()
    optuna.logging.enable_propagation()
    # Optunaの学習用関数を内部に作成
    def _objective(trial: optuna.trial.Trial):
        lr = trial.suggest_float(
            "learning_rate",
            float(bayopt_bounds["learning_rate"][0]),
            float(bayopt_bounds["learning_rate"][1]),
            log=True,
        )

        bayopt_model = make_opt_model(
            bayopt_bounds,
            vocab_size,
            trial,
            num_of_outputs
        )
        trial_path = bayopt_dir.joinpath(f"trial_{trial.number}")
        trial_path.mkdir(exist_ok=True)

        if validation_method == "cv":
            bayopt_trainer = CrossValidationTrainer(
                trial_path,
                learning_rate=lr,
                scheduler=None,
                n_epochs=n_epochs,
                batch_size=batch_size,
                cv_n_splits=cv_n_splits,
                early_stopping=None,
                device=device,
            )
        else:
            bayopt_trainer = HoldOutTrainer(
                trial_path,
                learning_rate=lr,
                scheduler=None,
                n_epochs=n_epochs,
                batch_size=batch_size,
                early_stopping=None,
                device=device,
            )
        # モデルの学習
        bayopt_trainer.fit(bayopt_model, bayopt_datasets)
        if BayOptLoss.loss is None:
            BayOptLoss.loss = bayopt_trainer.loss
        else:
            if BayOptLoss.loss > bayopt_trainer.loss:
                BayOptLoss.loss = bayopt_trainer.loss
                BayOptLoss.number = trial.number
        return bayopt_trainer.loss

    # ハイパーパラメータの探索の開始
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    study.optimize(_objective, n_trials=n_trials, n_jobs=1)
    # 探索のうち、一番損失が少なかった条件でのハイパーパラメータを保存
    trial = study.best_trial
    logger.info(
        f"Best Trial: {trial.number} with RMSE value: {trial.value}"
    )
    best_hparams = {
        "vocab_size": int(vocab_size),
        "lstm_dim": 2 ** trial.params["n_lstm_dim"],
        "dense_dim": 2 ** trial.params["n_dense_dim"],
        "embedding_dim": 2 ** trial.params["n_embedding_dim"],
        "learning_rate": trial.params["learning_rate"],
        "num_of_outputs": num_of_outputs,
    }
    return best_hparams


def make_opt_model(
    bayopt_bounds: dict,
    vacab_size: int,
    trial: optuna.trial.Trial,
    num_of_outputs=1,
):
    """
    Optunaのハイパーパラメータ探索用のモデルを作成する関数

    Parameters
    ----------
    bayopt_bounds : dict
        ハイパーパラメータの探索範囲
    vacab_size : int
        ボキャブラリのサイズ
    trial : optuna.trial.Trial
        Optunaのトライアルオブジェクト
    num_of_outputs : int, optional
        出力の数, by default 1

    Returns
    -------
    opt_model : LSTMAttention
        作成されたLSTMAttentionモデル
    """
    # ハイパーパラメータの探索範囲から値を取得
    n_lstm_dim = trial.suggest_int(
        "n_lstm_dim",
        bayopt_bounds["lstm_dim"][0],
        bayopt_bounds["lstm_dim"][1],
    )
    n_dense_dim = trial.suggest_int(
        "n_dense_dim",
        bayopt_bounds["dense_dim"][0],
        bayopt_bounds["dense_dim"][1],
    )
    n_embedding_dim = trial.suggest_int(
        "n_embedding_dim",
        bayopt_bounds["embedding_dim"][0],
        bayopt_bounds["embedding_dim"][1],
    )
    # 探索用に生成されたハイパーパラメータを用いてモデルを生成
    opt_model = LSTMAttention(
        vocab_size=vacab_size,
        lstm_dim=n_lstm_dim,
        dense_dim=n_dense_dim,
        embedding_dim=n_embedding_dim,
        return_proba=False,
        num_of_outputs=num_of_outputs
    )
    return opt_model


def training_model(
    model: LSTMAttention,
    output_dir: Path,
    datasets: dict[str, SmilesDataset],
    learning_rate: float,
    n_epochs=100,
    batch_size=32,
    early_stopping_patience=0,
    device="cpu",
):
    """
    モデルの本学習を行う関数

    Parameters
    ----------
    model : LSTMAttention
        学習するLSTMAttentionモデル
    output_dir : Path
        出力ディレクトリのPathオブジェクト
    datasets : dict[str, SmilesDataset]
        学習用のデータセット
    learning_rate : float
        学習率
    n_epochs : int, optional
        エポック数, by default 100
    batch_size : int, optional
        バッチサイズ, by default 32
    early_stopping_patience : int, optional
        Early Stoppingのエポック数, by default 0
    device : str, optional
        使用するデバイス, by default "cpu"

    Returns
    -------
    model : LSTMAttention
        学習後のLSTMAttentionモデル
    """
    training_dir = output_dir.joinpath("training")
    training_dir.mkdir()
    # EarlyStoppingの設定
    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            early_stopping_patience,
            output_dir.joinpath("model"),
        )
    else:
        early_stopping = None
    # 本学習はホールドアウト法で行う
    trainer = HoldOutTrainer(
        training_dir,
        learning_rate,
        scheduler=None,
        n_epochs=n_epochs,
        batch_size=batch_size,
        early_stopping=early_stopping,
        device=device,
    )
    trainer.fit(model, datasets)
    return model


def copy_model_params(output_dir: Path):
    """学習後のモデルパラメータを指定のディレクトリにコピーする関数

    Parameters
    ----------
    output_dir : Path
        学習結果の出力ディレクトリのPathオブジェクト
    """
    output_dir_name = output_dir.stem
    parent_dir = output_dir.parent.parent
    model_dir = parent_dir.joinpath("models").joinpath(output_dir_name)
    if not model_dir.exists():
        model_dir.mkdir()
    shutil.copy(
        output_dir.joinpath("model/all_params.yml"),
        model_dir.joinpath("all_params.yml")
    )
    shutil.copy(
        output_dir.joinpath("model/model_params.pth"),
        model_dir.joinpath("model_params.pth")
    )


def run(config_filepath: str):
    with open(config_filepath) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    # *****************************************
    # 変数の設定
    # *****************************************
    # ハイパーパラメータの最適化条件設定
    bayopt_on = config["bayopt_hparams"]["bayopt_on"]

    if bayopt_on:
        bayopt_bounds = config["bayopt_bounds"]
        bayopt_validation_method = config["bayopt_hparams"]["validation_method"]
        if bayopt_validation_method not in ["holdout", "cv"]:
            raise ValueError(
                "bayopt_validation_method must be 'holdout' or 'cv'."
            )
        if bayopt_validation_method == "cv":
            bayopt_cv_n_splits = config["bayopt_hparams"]["cv_n_splits"]
        bayopt_n_epochs = config["bayopt_hparams"]["n_epochs"]
        bayopt_n_trials = config["bayopt_hparams"]["n_trials"]
        bayopt_batch_size = config["bayopt_hparams"]["batch_size"]
    else:
        lstm_dim_ref = config["ref_hyperparam"]["lstm_dim"]
        dense_dim_ref = config["ref_hyperparam"]["dense_dim"]
        embedding_dim_ref = config["ref_hyperparam"]["embedding_dim"]
        learning_rate_ref = config["ref_hyperparam"]["learning_rate"]
    # 学習の条件設定
    augmentation = config["train"]["augmentation"]
    batch_size = config["train"]["batch_size"]
    n_epochs = config["train"]["n_epochs"]
    early_stopping_patience = config["train"]["early_stopping_patience"]
    tf16 = config["tf16"]
    seed = config["seed"]
    # scaling = config["train"]["scaling"]

    # データセットの設定
    dataset_filepath = config["dataset"]["filepath"]
    smiles_col_name = config["dataset"]["smiles_col_name"]
    prop_col_name = config["dataset"]["prop_col_name"]
    output_dir = config["dataset"]["output_path"]
    dataset_ratio = config["dataset"]["dataset_ratio"]

    if type(prop_col_name) is str:
        prop_col_name = [prop_col_name]

    # *****************************************
    # データセットの読み込み
    # *****************************************
    num_of_outputs = len(prop_col_name)
    dataset = pl.read_csv(dataset_filepath)
    dataset = dataset.select(smiles_col_name, *prop_col_name)
    print(dataset.head())

    # *****************************************
    # 出力ディレクトリの設定
    # *****************************************
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    log_dir = output_dir.joinpath("logs")
    log_dir.mkdir()
    model_dir = output_dir.joinpath("model")
    model_dir.mkdir()

    logger = utils.log_setup(log_dir, "training", verbose=True)
    logger.info(f"OS: {platform.system()}")

    # *****************************************
    # 計算精度の設定
    # *****************************************
    if tf16:
        precision = "16"
        logger.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("medium")
    else:
        precision = "32"
        logger.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("high")
    seed_everything(seed)

    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # *****************************************
    # データセットの前処理
    # *****************************************
    logger.info("***Sampling and splitting of the dataset.***")
    smiles_dataset = SmilesDataset(
        smiles=dataset[smiles_col_name].to_numpy(),
        y=dataset[*prop_col_name].to_numpy(),
        augmentation=augmentation,
        batch_size=batch_size,
        dataset_ratio=dataset_ratio,
        random_state=seed,
    )
    if smiles_dataset.train_dataset[0][0].count("*") <= 1:
        logger.info("Setup Molecule Tokens.")
    else:
        logger.info("Setup Polymer Tokens.")

    if augmentation:
        logger.info("***Data augmentation is True.***")
        logger.info("Augmented SMILES data size:")
    else:
        logger.info("***No data augmentation has been required.***")
        logger.info("SMILES data size:")
    # トークン化処理
    all_smiles_list = list(
        dataset.select(smiles_col_name).to_numpy().flatten()
    )
    # トークンのリストを作成
    all_tokenized_smiles = token.get_tokens(all_smiles_list)
    tokens = token.extract_vocab(all_tokenized_smiles)
    # トークンの数を取得
    vocab_size = len(tokens)
    logger.info("Tokens:")
    logger.info(tokens)
    logger.info(f"Of size: {vocab_size}")
    tokens, vocab_size = token.add_extract_tokens(tokens, vocab_size)
    max_length = max([len(i_smiles) for i_smiles in all_tokenized_smiles])
    max_length += 2  # ["pad"]の分
    logger.info(
        f"Maximum length of tokenized SMILES: {max_length} tokens"
        "(termination spaces included)"
    )
    # *****************************************
    # ハイパーパラメータの最適化
    # *****************************************
    if bayopt_on:
        logger.info(f"Dava validation method: {bayopt_validation_method}.")
        if bayopt_validation_method == "cv":
            logger.info(f"Num of fold: {bayopt_cv_n_splits}")
        splited_datasets, _ = smiles_dataset.split_dataset(
            bayopt_validation_method
        )
        tokenized_smiles_datasets = smiles_dataset.tokenize_smiles(
            splited_datasets
        )
        bayopt_datasets = smiles_dataset.tensorize_datasets(
            max_length,
            tokenized_smiles_datasets,
            tokens
        )
        best_hparams = bayopt_hparams(
            output_dir,
            bayopt_datasets,
            logger,
            vocab_size,
            bayopt_bounds,
            bayopt_validation_method,
            bayopt_cv_n_splits,
            bayopt_n_epochs,
            bayopt_batch_size,
            bayopt_n_trials,
            num_of_outputs,
            seed,
            device
        )
    else:
        best_hparams = {
            "vocab_size": vocab_size,
            "lstm_dim": lstm_dim_ref,
            "dense_dim": dense_dim_ref,
            "embedding_dim": embedding_dim_ref,
            "learning_rate": learning_rate_ref,
        }
    logger.info("Best Params")
    logger.info(f"LSTM dim       |{best_hparams['lstm_dim']}")
    logger.info(f"Dense dim      |{best_hparams['dense_dim']}")
    logger.info(f"Embedding dim  |{best_hparams['embedding_dim']}")
    logger.info(f"learning rate    |{best_hparams['learning_rate']}")
    logger.info("")
    # *****************************************
    # ハイパーパラメータの保存
    # *****************************************
    config["hyper_parameters"] = {
        "model": {
            "vocab_size": best_hparams["vocab_size"],
            "lstm_dim": best_hparams["lstm_dim"],
            "dense_dim": best_hparams["dense_dim"],
            "embedding_dim": best_hparams["embedding_dim"],
            "num_of_outputs": best_hparams["num_of_outputs"],
        },
        "other": {
            # "batch_size": best_hparams["batch_size"],
            "learning_rate": best_hparams["learning_rate"],
        }
    }
    config["token"] = {
        "max_length": max_length,
        "vocabulary": tokens
    }
    with open(model_dir.joinpath("all_params.yml"), mode="w") as f:
        yaml.dump(config, f)
    # *****************************************
    # モデルの本学習
    # *****************************************
    model = LSTMAttention(
        vocab_size=best_hparams["vocab_size"],
        lstm_dim=best_hparams["lstm_dim"],
        dense_dim=best_hparams["dense_dim"],
        embedding_dim=best_hparams["embedding_dim"],
        num_of_outputs=best_hparams["num_of_outputs"]
    )
    lr = best_hparams["learning_rate"]

    logger.info("***Training of the best model.***")
    splited_datasets, enum_cards = smiles_dataset.split_dataset("holdout")
    tokenized_smiles_datasets = smiles_dataset.tokenize_smiles(splited_datasets)
    training_datasets = smiles_dataset.tensorize_datasets(
        max_length,
        tokenized_smiles_datasets,
        tokens
    )

    training_model(
        model,
        output_dir,
        training_datasets,
        lr,
        n_epochs,
        batch_size,
        early_stopping_patience,
        device
    )
    logger.info("Training Finished !!!")
    # *****************************************
    # モデルの評価
    # *****************************************
    evaluate_model(
        model,
        logger,
        output_dir,
        training_datasets,
        enum_cards,
        batch_size,
        device,
    )
    copy_model_params(output_dir)


def main():
    parser = argparse.ArgumentParser(description="SMILES-X")
    parser.add_argument("config", help="config fileを読み込む")
    parser.add_argument("--devices")
    args = parser.parse_args()
    config_filepath = args.config
    devices = args.devices
    run(config_filepath, devices)


if __name__ == "__main__":
    main()
