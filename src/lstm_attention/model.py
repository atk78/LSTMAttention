import torch
import torch.nn as nn


class TimeDistributedDense(nn.Module):
    def __init__(self, lstm_untis, dense_units, batch_first: bool = False):
        super().__init__()
        self.dense = nn.Linear(lstm_untis, dense_units)
        self.batch_first = batch_first

    def forward(self, X):
        if len(X.size()) <= 2:
            return self.module(X)
        # Squash samples and timesteps into a single axis
        X_reshape = X.contiguous().view(
            -1, X.size(-1)
        )  # (samples * timesteps, input_size)
        y = self.dense(X_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                X.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, X.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class Attention(nn.Module):
    def __init__(self, dense_units, return_proba=False):
        super().__init__()
        self.return_proba = return_proba
        self.inner_dense = nn.Linear(
            in_features=dense_units, out_features=1
        )  # nn.bmmでも可
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X, mask=None):
        # e = tanh(H・W + b)
        et = self.inner_dense(X)
        et = self.tanh(et)
        et = torch.squeeze(input=et)  # et = et.view(et.shape[0], -1)
        # α = exp(e) / Σexp(e_i)
        at = self.softmax(et)
        if mask is not None:
            at *= mask.type(torch.float32)
        atx = torch.unsqueeze(input=at, dim=-1)  # atx = at.view(at.shape[0], at.shape[1], 1)
        # c = H^T・α
        c = X * atx
        if self.return_proba:
            return atx
        else:
            return torch.sum(c, dim=1)


class LSTMAttention(nn.Module):
    def __init__(
        self,
        vocab_size,
        lstm_dim=16,
        dense_dim=16,
        embedding_dim=32,
        return_proba=False,
        num_of_outputs=1,
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm_layer = nn.LSTM(
            embedding_dim, lstm_dim, bidirectional=True, batch_first=True
        )
        self.timedistributed_dense_layer = TimeDistributedDense(
            2 * lstm_dim, dense_dim, batch_first=True
        )
        self.attention_layer = Attention(dense_dim, return_proba)
        self.output_layer = nn.Linear(dense_dim, out_features=num_of_outputs)

    def forward(self, X):
        X = self.embedding_layer(X)
        X, _ = self.bilstm_layer(X)
        X = self.timedistributed_dense_layer(X)
        X = self.attention_layer(X)
        X = self.output_layer(X)
        return X

    # def predict(
    #     self,
    #     smiles_list: list[str],
    #     max_length: int,
    #     tokens: list[str],
    #     augmentation: bool,
    #     device="cpu",
    # ) -> torch.Tensor:
    #     smiles_array = np.array(smiles_list)
    #     y = np.zeros_like(smiles_array)
    #     y_pred_list = []
    #     smiles, enum_card = augm.data_augmentaion(
    #         smiles_array,
    #         y,
    #         augmentation=augmentation,
    #         )
    #     tokenized_smiles = token.get_tokens(smiles)
    #     token_tensor, y_tensor = token.convert_to_int_tensor(
    #         tokenized_smiles, y, max_length, tokens
    #     )
    #     dataset = Data(X=token_tensor, prop=y_tensor)
    #     self.eval()
    #     results = {smiles: None for smiles in smiles_list}
    #     self.to(device)
    #     with torch.no_grad():
    #         for X, _ in dataset:
    #             X = X.to(device)
    #             y_pred = self(X)
    #             y_pred_list.extend(y_pred.cpu().detach().numpy().flatten())
    #     y_pred_list, _ = augm.mean_std_result(enum_card, y_pred_list)
    #     for smiles, y in zip(smiles_list, y_pred_list):
    #         results[smiles] = y
    #     return results
