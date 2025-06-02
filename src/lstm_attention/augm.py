from rdkit import Chem
import pandas as pd
import numpy as np
from scipy.ndimage import shift


def rotate_atoms(li: list[int], x: int) -> list[int]:
    """
    リスト内の要素を指定した数だけ回転させた新しいリストを返します。

    Parameters
    ----------
    li : list[int]
        回転させる整数のリスト
    x : int
        回転させる要素数（正の値で右回転、負の値で左回転）

    Returns
    -------
    list[int]
        回転後の新しいリスト

    Examples
    ---
    >>> rotate_atoms([1, 2, 3, 4], 1)
    [2, 3, 4, 1]
    >>> rotate_atoms([1, 2, 3, 4], -1)
    [4, 1, 2, 3]
    """
    return li[x % len(li):] + li[: x % len(li)]


def augmente_smiles(
    smiles: str,
    augmentation: bool = True,
    kekule: bool = False
) -> list[str]:
    """SMILESを並び変えて1つのSMILESから複数のSMILESを生成（データ拡張）する関数

    Parameters
    ----------
    smiles : str
        拡張したいSMILES
    augmentation : bool, optional
        `True`の場合SMILESの拡張を実施する, by default True
    kekule : bool, optional
        `True`の場合ケクレ化を実施する, by default False

    Returns
    -------
    list[str]
        元のSMILESと拡張されたSMILESのリスト

    Examples
    --------
    >>> smiles_list = generate_smiles('CCO')
    >>> print(smiles_list)
    ['CCO', 'OCC', 'COC']
    """
    smiles_list = list()
    # SMILESがrdkitのmolに変換できない（間違った構造）場合はNoneを返す
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        mol = None
    if mol is not None:
        n_atoms = mol.GetNumAtoms()
        n_atoms_list = [n for n in range(n_atoms)]
        # SMILESを並び変えて複数のSMILESを生成
        # 例えばCCOならばCCO, OCC, COCの3つのSMILESを生成
        # ただし、n_atomsが0の場合はNoneを返す
        if augmentation:
            if n_atoms != 0:
                for i_atoms in range(n_atoms):
                    n_atoms_list_tmp = rotate_atoms(n_atoms_list, i_atoms)
                    # atomの順番を並び替えたmolを生成
                    renum_mol = Chem.RenumberAtoms(mol, n_atoms_list_tmp)
                    try:
                        # renum_molをSMILESに変換
                        smiles = Chem.MolToSmiles(
                            renum_mol,
                            isomericSmiles=True,
                            kekuleSmiles=kekule,
                            rootedAtAtom=-1,
                            canonical=False,
                            allBondsExplicit=False,
                            allHsExplicit=False,
                        )
                    except Exception as e:
                        smiles = "None"
                    smiles_list.append(smiles)
            else:
                smiles = "None"
                smiles_list.append(smiles)
        else:
            # データ拡張をしない場合canonical SMILESを生成
            try:
                smiles = Chem.MolToSmiles(
                    mol,
                    isomericSmiles=True,
                    kekuleSmiles=kekule,
                    rootedAtAtom=-1,
                    canonical=True,
                    allBondsExplicit=False,
                    allHsExplicit=False,
                )
            except Exception as e:
                smiles = "None"
            smiles_list.append(smiles)
    else:
        smiles = "None"
        smiles_list.append(smiles)

    # smiles_list = (
    #     pd.DataFrame(smiles_list).drop_duplicates().iloc[:, 0].values.tolist()
    # )  # duplicates are discarded
    smiles_list = list(set(smiles_list))  # duplicates are discarded
    return smiles_list


def data_augmentation(
    smiles_array: np.ndarray,
    prop_array: np.ndarray,
    augmentation: bool = True
) -> tuple[list[str], list[int], list[float]]:
    """SMILESを拡張して、その拡張されたSMILESと対応する物性を返す関数

    Parameters
    ----------
    smiles_array : np.ndarray
        拡張したいSMILESの配列（例: np.array(['CCO', 'CO'])）
    prop_array : np.ndarray
        拡張するSMILESに対応する物性の配列（例: np.array([0.5, 1.0])）
    augmentation : bool, optional
        `True`の場合データを拡張する, デフォルトはTrue

    Returns
    -------
    tuple[list[str], list[int], list[float]]
        拡張後のSMILESリスト、各SMILESごとの拡張数リスト、拡張後の物性値リスト

    Example
    ---
    >>> smiles_array = np.array(['CCO'])
    >>> prop_array = np.array([0.5])
    >>> data_augmentation(smiles_array, prop_array, augmentation=True)
    (['CCO', 'OCC', 'COC', 'CO', 'OC'], [3, 2], [0.5, 0.5, 0.5, 1.0, 1.0])
    """
    smiles_enum = list()
    prop_enum = list()
    smiles_enum_card = list()
    for idx, i_smiles in enumerate(smiles_array):
        enumerated_smiles = augmente_smiles(i_smiles, augmentation)
        if "None" not in enumerated_smiles:
            smiles_enum_card.append(len(enumerated_smiles))
            smiles_enum.extend(enumerated_smiles)
            prop_enum.extend([prop_array[idx]] * len(enumerated_smiles))
    return smiles_enum, smiles_enum_card, prop_enum


def mean_std_result(x_cardinal_tmp: list[int], y_pred_tmp: list[float]):
    """
    x_cardinal_tmpで指定された各セグメントごとに、y_pred_tmpの平均値と標準偏差を計算する。

    Parameters
    ----------
    x_cardinal_tmp : list[int]
        y_pred_tmpを分割する各セグメントの長さを指定する整数リスト。
        例: [3, 2, 4] の場合、y_pred_tmpは最初の3つ、次の2つ、最後の4つの要素で分割される。
    y_pred_tmp : list[float]
        分割・解析対象となる予測値（float）のリスト。
        例: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Returns
    -------
    y_mean : np.ndarray
        各セグメントごとの平均値の配列。
        例: array([0.2, 0.45, 0.75])
    y_std : np.ndarray
        各セグメントごとの標準偏差の配列。
        例: array([0.0816, 0.0707, 0.0816])

    Examples
    --------
    >>> x_cardinal_tmp = [3, 2, 4]
    >>> y_pred_tmp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    >>> mean, std = mean_std_result(x_cardinal_tmp, y_pred_tmp)
    >>> print(mean)
    [0.2  0.45 0.75]
    >>> print(std)
    [0.08164966 0.07071068 0.08164966]
    """
    x_card_cumsum = np.cumsum(np.array(x_cardinal_tmp))
    x_card_cumsum_shift = shift(x_card_cumsum, 1, cval=0)
    y_mean = np.array(
        [
            np.mean(y_pred_tmp[x_card_cumsum_shift[cenumcard]: ienumcard])
            for cenumcard, ienumcard in enumerate(x_card_cumsum)
        ]
    )
    y_std = np.array(
        [
            np.std(y_pred_tmp[x_card_cumsum_shift[cenumcard]: ienumcard])
            for cenumcard, ienumcard in enumerate(x_card_cumsum)
        ]
    )
    return y_mean, y_std
