"""
Simple 1-layer transformer models that can be used to train on toy datasets of
a few hundred to a few thousand samples.

With help from the PyTorch tutorial on transformers:

https://github.com/pytorch/examples/blob/main/word_language_model/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=10):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SimpleTransformerClassifier(nn.Transformer):
    """
    A simple transformer model that can be used to train on toy datasets of
    a few hundred to a few thousand samples. The model regards its input as a
    sequence of tokens and outputs a single vector to predict the next token,
    that is, the target class.

    It is thus not a full-fledged language model, but a classifier, that does
    not employ masking or positional embeddings.
    """
    def __init__(self, n_token, d_model=64, nhead=4, num_layers=1, dim_feedforward=128, dropout=0.1):
        super(SimpleTransformerClassifier, self).__init__(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                                num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
                                                dropout=dropout, )

        self.input_emb = nn.Embedding(n_token, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_token)
        self.n_token = n_token

        # initialize weights
        #range = 0.1
        #nn.init.uniform_(self.input_emb.weight, -range, range)
        #nn.init.zeros_(self.decoder.bias)
        #nn.init.uniform_(self.decoder.weight, -range, range)

    def forward(self, src):
    
        src = self.input_emb(src)  * math.sqrt(self.d_model)
        
        output = self.encoder(src, mask=None)
        output = self.decoder(output)
        
        output = output[:, -1, :].softmax(dim=-1)

        return output
    
class SimplePositionalEmbedding(nn.Module):
    
    def __init__(self, d_model, max_len=10):
        super(SimplePositionalEmbedding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # print(x.shape, self.pe.shape)
        return torch.add(x, self.pe[:, :x.size(1), :])
    
    
class SingleLayerTransformerClassifier(nn.Module):
    """
    A simple transformer model that can be used to train on toy datasets of
    a few hundred to a few thousand samples. The model regards its input as a
    sequence of tokens and outputs a single vector to predict the next token,
    that is, the target class.

    It is thus not a full-fledged language model, but a classifier, that does
    not employ masking or positional embeddings.
    """
    def __init__(self, n_token, d_model=64, nhead=4,  dim_mlp=128, dropout=0.0):
        super(SingleLayerTransformerClassifier, self).__init__()
        
        # input embedding - no need for positional embeddings as we only have two tokens and
        # the target is invariant to the order of the tokens
        self.input_emb = nn.Embedding(n_token, d_model)

        self.positional_encoding = SimplePositionalEmbedding(d_model)

        # one layer encoder
        # self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # replace with sceld_dot_product_attention
        self.head1_encoder = nn.Linear(d_model, int(d_model/nhead), bias=False)
        self.head2_encoder = nn.Linear(d_model, int(d_model/nhead), bias=False)
        self.head3_encoder = nn.Linear(d_model, int(d_model/nhead), bias=False)
        self.head4_encoder = nn.Linear(d_model, int(d_model/nhead), bias=False)

        self.head1_decoder = nn.Linear(int(d_model/nhead), d_model, bias=False)
        self.head2_decoder = nn.Linear(int(d_model/nhead), d_model, bias=False)
        self.head3_decoder = nn.Linear(int(d_model/nhead), d_model, bias=False)
        self.head4_decoder = nn.Linear(int(d_model/nhead), d_model, bias=False)

        self.heads_encoders = [self.head1_encoder, self.head2_encoder, self.head3_encoder, self.head4_encoder]
        self.heads_decoders = [self.head1_decoder, self.head2_decoder, self.head3_decoder, self.head4_decoder]

        # self.layer_norm = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, d_model)

        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_token, bias=False)
        self.n_token = n_token

    def apply_attn(self, input, encoders, decoders):
        heads = []
        for i, (encoder, decoder) in enumerate(zip(encoders, decoders)):
            head = encoder(input)
            head = nn.functional.scaled_dot_product_attention(head, head, head)
            head = decoder(head)
            heads.append(head)
        return torch.stack(heads).sum(dim=0)
        
    def forward(self, src, output_activations=False):

        activations = dict()
    
        src = self.input_emb(src)  * math.sqrt(self.d_model)
        # src = self.positional_encoding(src)

        if output_activations:
            activations['input_embeddings'] = src

        # output = src + self.attention(src, src, src)[0]
        output = self.apply_attn(src, self.heads_encoders, self.heads_decoders)

        if output_activations:
            activations['attention_out'] = output

        output = self.dropout(output)
        intermediate = F.relu(self.linear1(output))
        intermediate = self.dropout(intermediate)
        output = output + F.relu(self.linear2(intermediate))


        if output_activations:
            activations['encoder_output'] = output

        output = self.decoder(output)

        if output_activations:
            activations['decoder_output'] = output
        
        output = output[:, -1, :].log_softmax(dim=-1)

        if output_activations:
            return output, activations

        return output
