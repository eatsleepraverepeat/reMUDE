"""https://arxiv.org/pdf/1911.09789.pdf"""


from torch import nn
from torch.nn import functional as torch_func
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


MAX_CHARS = 24


# TRANSFORMER
class TransformerEncoder(nn.Module):

    def __init__(self, depth, dim, attn_heads, dim_ff, dropout):
        super(TransformerEncoder, self).__init__()

        self.depth = depth
        self.tdim = dim
        self.tattn_heads = attn_heads
        self.tdim_ff = dim_ff
        self.dropout = dropout

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.tdim,
            nhead=self.tattn_heads,
            dim_feedforward=self.tdim_ff,
            dropout=self.dropout,
            activation='relu'
        )
        self.encoder_model = nn.TransformerEncoder(self.encoder_layer, num_layers=self.depth)

    def forward(self, x, mask):
        return self.encoder_model(x, src_key_padding_mask=mask)

    def _reset(self):
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return


# FULL FRAMEWORK
class MUDE(nn.Module):
    """Follows https://arxiv.org/pdf/1911.09789.pdf"""

    def __init__(
        self,
        dim,
        characters_vocab_size,
        tokens_vocab_size,
        encoder_depth,
        encoder_attn_heads,
        encoder_dimff,
        encoder_dropout,
        top_rec_hidden_dim,
        top_rec_proj_dropout
    ):
        super(MUDE, self).__init__()
        self.DIM = dim

        self.embedding = nn.Embedding(
            num_embeddings=characters_vocab_size,
            embedding_dim=self.DIM,
            scale_grad_by_freq=True
        )

        self.encoder = TransformerEncoder(
            depth=encoder_depth,
            dim=self.DIM,
            attn_heads=encoder_attn_heads,
            dim_ff=encoder_dimff,
            dropout=encoder_dropout
        )

        self.decoder = nn.GRU(self.DIM, self.DIM, batch_first=True, bidirectional=False)
        self.char_seq_pred = nn.Linear(self.DIM, characters_vocab_size)

        self.top_rec_unit = nn.LSTM(
            dim,
            top_rec_hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.top_proj_dropout = nn.Dropout(top_rec_proj_dropout)
        self.token_seq_pred = nn.Linear(top_rec_hidden_dim * 2, tokens_vocab_size)

    def forward(self, x, xmask, xlengths):
        batch_size, l, c = x.shape

        embedded = self.embedding(x)
        embedded = embedded.view(-1, MAX_CHARS, self.DIM)  # batch_size, l, c, d_emb -> batch_size x l, c, d_emb

        # https://pytorch.org/tutorials/beginner/transformer_tutorial.html#functions-to-generate-input-and-target-sequence
        encoder_input = embedded.transpose(1, 0)  # batch_size x l, c, d_emb > c, batch_size x l, d_emb

        x = self.encoder(encoder_input, xmask.view(-1, MAX_CHARS)).transpose(0, 1)
        x = x[:, 0, :]  # taking c0
        x = x.view(batch_size, l, self.DIM)  # reshaping as batch_size, l, <c0-embedding>

        # rnn decoder block
        decoder_hidden = x.view(1, -1, self.DIM)  # hidden state is now 1, dim, dim
        decoder_input = embedded[:, :-1, :]
        char_seq_output, char_seq_hidden = self.decoder(decoder_input, decoder_hidden)
        char_seq_projected = self.char_seq_pred(char_seq_output)
        char_seq_logits = torch_func.log_softmax(char_seq_projected, dim=1).view(batch_size, l, c-1, -1)

        packed_x = pack_padded_sequence(x, lengths=xlengths, batch_first=True, enforce_sorted=True)  # pack

        tok_seq_output, tok_seq_hidden = self.top_rec_unit(packed_x)
        tok_seq_output = pad_packed_sequence(tok_seq_output, batch_first=True)[0]  # pad

        tok_seq_output = self.top_proj_dropout(tok_seq_output)
        tok_seq_projected = self.token_seq_pred(tok_seq_output)
        tok_seq_logits = torch_func.log_softmax(tok_seq_projected, dim=-1)

        return char_seq_logits, tok_seq_logits
