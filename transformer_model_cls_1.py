"""Defines Transformer model in tf.keras API."""
import tensorflow as tf

# import utils
from commons.tokenization import SOS_ID
from commons.tokenization import EOS_ID
from commons import beam_search
from commons.layers import EmbeddingLayer
from commons.layers import FeedForwardNetwork
from commons.layers import Attention
from commons.layers_attention_mask_1 import Attention_mask
#from commons.layers_attention_mask_new import Attention_mask
import tensorflow.keras.backend as K
from einops import rearrange, repeat


def get_padding_mask(inputs, padding_value=0):
    """Creates a binary tensor to mask out padded tokens.

    Args:
      inputs: int tensor of shape [batch_size, src_seq_len], token ids
        of source sequences.
      padding_value: int scalar, the vocabulary index of the PAD token.

    Returns:
      mask: binary tensor of shape [batch_size, 1, 1, src_seq_len], storing ones
        for padded tokens and zeros for regular tokens.
    """
    batch_size = tf.shape(inputs)[0]
    pad_1 = tf.ones([batch_size,1])
    inputs_pad = tf.concat([pad_1,inputs],axis=1)
    mask = tf.cast(tf.equal(inputs_pad, padding_value), 'float32')
    mask = mask[:, tf.newaxis, tf.newaxis, :]
    return mask


def get_look_ahead_mask(seq_len):
    """Creates a tensor to mask out future tokens in the target sequences when in
    training mode.

    Given sequence length `L` of target sequence, the mask would be a L x L
    matrix (when `tf.squeeze`'ed) where upper diagonal entries are ones and all
    other entries zeros.

    0, 1, 1, ..., 1
    0, 0, 1, ..., 1

        ... ...

    0, 0, 0, ..., 0

    Args:
      seq_len: int scalar tensor, sequence length.

    Returns:
      mask: float tensor of shape [1, 1, seq_len, seq_len], the mask tensor.
    """
    mask = 1 - tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
    mask = mask[tf.newaxis, tf.newaxis, :, :]
    return mask


def get_positional_encoding(seq_len, hidden_size, reverse=False):
    """Creates a tensor that encodes positional information.

    Args:
      seq_len: int scalar tensor, sequence length.
      hidden_size: int scalar, the hidden size of continuous representation.
      reverse: bool, whether to reverse the sequence. Defaults to False.

    Returns:
      positional_encoding: float tensor of shape [seq_len, hidden_size], the
        tensor that encodes positional information.
    """
    distances = tf.cast(tf.range(seq_len), 'float32')
    hidden_size //= 2
    inverse_frequencies = 1 / (
            10000 ** (tf.cast(tf.range(hidden_size), 'float32') / (hidden_size - 1)))
    positional_encoding = tf.einsum('i,j->ij', distances, inverse_frequencies)
    positional_encoding = tf.concat([tf.sin(positional_encoding),
                                     tf.cos(positional_encoding)], axis=1)
    return positional_encoding


def compute_loss(labels, logits, smoothing, vocab_size, padding_value=0):
    """Computes average (per-token) cross entropy loss.

    1. Applies label smoothing -- all entries in the groundtruth label tensor
       get non-zero probability mass.
    2. Computes per token loss of shape [batch_size, tgt_seq_len], where padded
       positions are masked, and then the sum of per token loss is normalized by
       the total number of non-padding entries.

    Args:
      labels: int tensor of shape [batch_size, tgt_seq_len], the groundtruth
        token ids.
      logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size], the
        predicted logits of tokens over the vocabulary.
      smoothing: float scalar, the amount of label smoothing applied to the
        one-hot class labels.
      vocab_size: int scalar, num of tokens (including SOS and EOS) in the
        vocabulary.
      padding_value: int scalar, the vocabulary index of the PAD token.

    Returns:
      loss: float scalar tensor, the per-token cross entropy
    """
    # effective_vocab = vocab - {SOS_ID}
    effective_vocab_size = vocab_size - 1

    # prob mass allocated to the token that should've been predicted
    on_value = 1.0 - smoothing
    # prob mass allocated to all other tokens
    off_value = smoothing / (effective_vocab_size - 1)

    # [batch_size, tgt_seq_len, vocab_size]
    labels_one_hot = tf.one_hot(
        labels,
        depth=vocab_size,
        on_value=on_value,
        off_value=off_value)

    # compute cross entropy over all tokens in vocabulary but SOS_ID (i.e. 0)
    # because SOS_ID should never appear in the decoded sequence
    # [batch_size, tgt_seq_len]
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_one_hot[:, :, 1:], logits=logits[:, :, 1:])

    # this is the entropy when the softmax'ed logits == groundtruth labels
    # so it should be deducted from `cross_entropy` to make sure the minimum
    # possible cross entropy == 0
    normalizing_constant = -(on_value * tf.math.log(on_value) +
                             (effective_vocab_size - 1) * off_value * tf.math.log(off_value + 1e-20))
    cross_entropy -= normalizing_constant

    # mask out predictions where the labels == `padding_value`
    weights = tf.cast(tf.not_equal(labels, padding_value), 'float32')
    cross_entropy *= weights
    loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(weights)
    return loss


class EncoderLayer(tf.keras.layers.Layer):
    """The building block that makes the encoder stack of layers, consisting of an
    attention sublayer and a feed-forward sublayer.
    """

    def __init__(self, hidden_size, num_heads, filter_size, dropout_rate):
        """Constructor.

        Args:
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          filter_size: int scalar, the depth of the intermediate dense layer of the
            feed-forward sublayer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(EncoderLayer, self).__init__()
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        self._mha = Attention(hidden_size, num_heads, dropout_rate)
        self._layernorm_mha = tf.keras.layers.LayerNormalization()
        self._dropout_mha = tf.keras.layers.Dropout(dropout_rate)

        self._ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self._layernorm_ffn = tf.keras.layers.LayerNormalization()
        self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, padding_mask, training):
        """Computes the output of the encoder layer.

        Args:
          inputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
            input source sequences.
          padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
            populated with either 0 (for tokens to keep) or 1 (for tokens to be
            masked).
          training: bool scalar, True if in training mode.

        Returns:
          outputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
            output source sequences.
        """
        query = reference = self._layernorm_mha(inputs)
        outputs = self._mha(query, reference, padding_mask, training)
        ffn_inputs = self._dropout_mha(outputs, training=training) + inputs

        outputs = self._layernorm_ffn(ffn_inputs)
        outputs = self._ffn(outputs, training)
        outputs = self._dropout_ffn(outputs, training=training) + ffn_inputs
        return outputs


class DecoderLayer(tf.keras.layers.Layer):
    """The building block that makes the decoder stack of layers, consisting of a
    self-attention sublayer, cross-attention sublayer and a feed-forward sublayer.
    """

    def __init__(self, hidden_size, num_heads, filter_size, dropout_rate,attention_mask_path):
        """Constructor.

        Args:
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          filter_size: int scalar, the depth of the intermediate dense layer of the
            feed-forward sublayer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(DecoderLayer, self).__init__()
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        self._mha_intra = Attention(hidden_size, num_heads, dropout_rate)
        self._layernorm_mha_intra = tf.keras.layers.LayerNormalization()
        self._dropout_mha_intra = tf.keras.layers.Dropout(dropout_rate)

        self._mha_inter = Attention_mask(hidden_size, num_heads, dropout_rate,attention_mask_path) # use attention mask
        #self._mha_inter = Attention(hidden_size, num_heads, dropout_rate) # not use attention mask
        self._layernorm_mha_inter = tf.keras.layers.LayerNormalization()
        self._dropout_mha_inter = tf.keras.layers.Dropout(dropout_rate)

        self._ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self._layernorm_ffn = tf.keras.layers.LayerNormalization()
        self._dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self,
             inputs,
             encoder_outputs,
             look_ahead_mask,
             padding_mask,
             training,
             cache=None):
        """Computes the output of the decoder layer.

        Args:
          inputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size], the
            input target sequences.
          encoder_outputs: float tensor of shape [batch_size, src_seq_len,
            hidden_size], the encoded source sequences to be used as reference.
          look_ahead_mask: float tensor of shape [1, 1, tgt_seq_len, tgt_seq_len],
            populated with either 0 (for tokens to keep) or 1 (for tokens to be
            masked).
          padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
            populated with either 0 (for tokens to keep) or 1 (for tokens to be
            masked).
          training: bool scalar, True if in training mode.
          cache: (Optional) dict with entries
            'k': tensor of shape [batch_size * beam_width, seq_len, num_heads,
              size_per_head],
            'v': tensor of shape [batch_size * beam_width, seq_len, num_heads,
              size_per_head],
            'tgt_tgt_attention': tensor of shape [batch_size * beam_width,
              num_heads, tgt_seq_len, tgt_seq_len],
            'tgt_src_attention': tensor of shape [batch_size * beam_width,
              num_heads, tgt_seq_len, src_seq_len].
            Must be provided in inference mode.

        Returns:
          outputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size], the
            output target sequences.
        """
        query = reference = self._layernorm_mha_intra(inputs)
        outputs = self._mha_intra(
            query, reference, look_ahead_mask, training, cache=cache)
        mha_inter_inputs = self._dropout_mha_intra(outputs, training=training
                                                   ) + inputs

        query, reference = self._layernorm_mha_inter(mha_inter_inputs
                                                     ), encoder_outputs
        outputs = self._mha_inter(
            query, reference, padding_mask, training, cache=cache)
        ffn_inputs = self._dropout_mha_inter(outputs, training=training
                                             ) + mha_inter_inputs

        outputs = self._layernorm_ffn(ffn_inputs)
        outputs = self._ffn(outputs, training)
        outputs = self._dropout_ffn(outputs, training=training) + ffn_inputs
        return outputs


class Encoder(tf.keras.layers.Layer):
    """The Encoder that consists of a stack of structurally identical layers."""

    def __init__(
            self, stack_size, hidden_size, num_heads, filter_size, dropout_rate):
        """Constructor.

        Args:
          stack_size: int scalar, num of layers in the stack.
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          filter_size: int scalar, the depth of the intermediate dense layer of the
            feed-forward sublayer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(Encoder, self).__init__()
        self._stack_size = stack_size
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        self._stack = [EncoderLayer(hidden_size,
                                    num_heads,
                                    filter_size,
                                    dropout_rate) for _ in range(self._stack_size)]
        self._layernorm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, padding_mask, training):
        """Computes the output of the encoder stack of layers.

        Args:
          inputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
            input source sequences.
          padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
            populated with either 0 (for tokens to keep) or 1 (for tokens to be
            masked).
          training: bool scalar, True if in training mode.

        Returns:
          outputs: float tensor of shape [batch_size, src_seq_len, hidden_size], the
            output source sequences.
        """
        for layer in self._stack:
            inputs = layer.call(inputs, padding_mask, training)
        outputs = self._layernorm(inputs)
        return outputs


class Decoder(tf.keras.layers.Layer):
    """Decoder that consists of a stack of structurally identical layers."""

    def __init__(
            self, stack_size, hidden_size, num_heads, filter_size, dropout_rate,attention_mask_path):
        """Constructor.

        Args:
          stack_size: int scalar, the num of layers in the stack.
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          filter_size: int scalar, the depth of the intermediate dense layer of the
            feed-forward sublayer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
        """
        super(Decoder, self).__init__()
        self._stack_size = stack_size
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate

        self._stack = [DecoderLayer(
            hidden_size, num_heads, filter_size, dropout_rate,attention_mask_path)
            for _ in range(self._stack_size)]
        self._layernorm = tf.keras.layers.LayerNormalization()

    def call(self,
             inputs,
             encoder_outputs,
             look_ahead_mask,
             padding_mask,
             training,
             cache=None):
        """Computes the output of the decoder stack of layers.

        Args:
          inputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size], the
            input target sequences.
          encoder_outputs: float tensor of shape [batch_size, src_seq_len,
            hidden_size], the encoded source sequences to be used as reference.
          look_ahead_mask: float tensor of shape [1, 1, tgt_seq_len, tgt_seq_len],
            populated with either 0 (for tokens to keep) or 1 (for tokens to be
            masked).
          padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
            populated with either 0 (for tokens to keep) or 1 (for tokens to be
            masked).
          training: bool scalar, True if in training mode.
          cache: (Optional) dict with keys 'layer_0', ...
            'layer_[self.num_layers - 1]', where the value
            associated with each key is a dict with entries
              'k': tensor of shape [batch_size * beam_width, seq_len, num_heads,
                size_per_head],
              'v': tensor of shape [batch_size * beam_width, seq_len, num_heads,
                size_per_head],
              'tgt_tgt_attention': tensor of shape [batch_size * beam_width,
                num_heads, tgt_seq_len, tgt_seq_len],
              'tgt_src_attention': tensor of shape [batch_size * beam_width,
                num_heads, tgt_seq_len, src_seq_len].
            Must be provided in inference mode.

        Returns:
          outputs: float tensor of shape [batch_size, tgt_seq_len, hidden_size], the
            output target sequences.
        """
        for i, layer in enumerate(self._stack):
            inputs = layer.call(inputs,
                                encoder_outputs,
                                look_ahead_mask,
                                padding_mask,
                                training,
                                cache=cache['layer_%d' % i]
                                if cache is not None else None)
        outputs = self._layernorm(inputs)
        return outputs


class TransformerModel_cls_1(tf.keras.layers.Layer):
    """Transformer model as described in https://arxiv.org/abs/1706.03762

    The model implements methods `call` and `transduce`, where
      - `call` is invoked in training mode, taking as input BOTH the source and
        target token ids, and returning the estimated logits for the target token
        ids.
      - `transduce` is invoked in inference mode, taking as input the source token
        ids ONLY, and outputting the token ids of the decoded target sequences
        using beam search.
    """

    def __init__(self,
                 vocab_size,
                 encoder_stack_size=6,
                 decoder_stack_size=6,
                 hidden_size=512,
                 num_heads=8,
                 filter_size=2048,
                 dropout_rate=0.1,
                 extra_decode_length=50,
                 beam_width=4,
                 alpha=0.6,
                 attention_mask_path = ''):
        """Constructor.

        Args:
          vocab_size: int scalar, num of subword tokens (including SOS/PAD and EOS)
            in the vocabulary.
          encoder_stack_size: int scalar, num of layers in encoder stack.
          decoder_stack_size: int scalar, num of layers in decoder stack.
          hidden_size: int scalar, the hidden size of continuous representation.
          num_heads: int scalar, num of attention heads.
          filter_size: int scalar, the depth of the intermediate dense layer of the
            feed-forward sublayer.
          dropout_rate: float scalar, dropout rate for the Dropout layers.
          extra_decode_length: int scalar, the max decode length would be the sum of
            `tgt_seq_len` and `extra_decode_length`.
          beam_width: int scalar, beam width for beam search.
          alpha: float scalar, the parameter for length normalization used in beam
            search.
        """
        super(TransformerModel_cls_1, self).__init__()
        self._vocab_size = vocab_size
        self._encoder_stack_size = encoder_stack_size
        self._decoder_stack_size = decoder_stack_size
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._dropout_rate = dropout_rate
        self._extra_decode_length = extra_decode_length
        self._beam_width = beam_width
        self._alpha = alpha

        self._embedding_logits_layer = EmbeddingLayer(vocab_size, hidden_size)
        self._encoder = Encoder(
            encoder_stack_size, hidden_size, num_heads, filter_size, dropout_rate)
        self._decoder = Decoder(
            decoder_stack_size, hidden_size, num_heads, filter_size, dropout_rate,attention_mask_path)

        self._encoder_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self._decoder_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.cls_token_gene = tf.Variable(initial_value=tf.random.normal([1, 1, 128]),name='cls1')
        self.cls_token_peak = tf.Variable(initial_value=tf.random.normal([1, 1, 128]),name='cls2')


    def call(self, src_token_ids, tgt_token_ids):
        """Takes as input the source and target token ids, and returns the estimated
        logits for the target sequences. Note this function should be called in
        training mode only.

        Args:
          src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
            of source sequences.
          tgt_token_ids: int tensor of shape [batch_size, tgt_seq_len], token ids
            of target sequences.

        Returns:
          logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size].
        """
        x1 = K.tanh(K.sum(src_token_ids, axis=-1))
        padding_mask = get_padding_mask(x1, SOS_ID)
        encoder_outputs = self._encode(src_token_ids, padding_mask,self.cls_token_peak, training=True)
        logits,decoder_outputs = self._decode(
            tgt_token_ids, encoder_outputs, padding_mask,self.cls_token_gene)

        return encoder_outputs, decoder_outputs, logits

    def _encode(self, src_token_ids, padding_mask,cls_token, training=True):
        """Converts source sequences token ids into continuous representation, and
        computes the Encoder-encoded sequences.

        Args:
          src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
            of source sequences.
          padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
            populated with either 0 (for tokens to keep) or 1 (for tokens to be
            masked).
          training: bool scalar, True if in training mode.

        Returns:
          encoder_outputs: float tensor of shape [batch_size, src_seq_len,
            hidden_size], the encoded source sequences to be used as reference.
        """
        src_seq_len = tf.shape(src_token_ids)[1]
        b,n,d = src_token_ids.shape

        # [batch_size, src_seq_len, hidden_size]
        src_token_embeddings = src_token_ids

        cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)

        src_token_embeddings = tf.concat([cls_tokens, src_token_embeddings], axis=1)

        # [src_seq_len, hidden_size]
        positional_encoding = get_positional_encoding(
            src_seq_len+1, self._hidden_size)
        src_token_embeddings += positional_encoding
        src_token_embeddings = self._encoder_dropout_layer(
            src_token_embeddings, training)

        # encoder_outputs = self._encoder(
        #     src_token_embeddings, padding_mask, training)

        return src_token_embeddings  # encoder_outputs

    def _decode(self, tgt_token_ids, encoder_outputs, padding_mask, cls_token):
        """Computes the estimated logits of target token ids, based on the encoded
        source sequences. Note this function should be called in training mode only.

        Args:
          tgt_token_ids: int tensor of shape [batch_size, tgt_seq_len] token ids of
            target sequences.
          encoder_outputs: float tensor of shape [batch_size, src_seq_len,
            hidden_size], the encoded source sequences to be used as reference.
          padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
            populated with either 0 (for tokens to keep) or 1 (for tokens to be
            masked).

        Returns:
          logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size].
        """
        tgt_seq_len = tf.shape(tgt_token_ids)[1]
        b, n, d = tgt_token_ids.shape
        # [batch_size, tgt_seq_len, hidden_size]
        tgt_token_embeddings = tgt_token_ids

        cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)

        tgt_token_embeddings = tf.concat([cls_tokens, tgt_token_embeddings], axis=1)

        # [tgt_seq_len, hidden_size]
        positional_encoding = get_positional_encoding(
            tgt_seq_len+1, self._hidden_size)
        tgt_token_embeddings += positional_encoding
        tgt_token_embeddings = self._decoder_dropout_layer(
            tgt_token_embeddings, training=True)

        look_ahead_mask = get_look_ahead_mask(tgt_seq_len+1)

        decoder_outputs = self._decoder(tgt_token_embeddings,
                                        encoder_outputs,
                                        look_ahead_mask,
                                        padding_mask,
                                        training=True)

        # logits = self._embedding_logits_layer(decoder_outputs, 'logits')
        return decoder_outputs, tgt_token_embeddings

    def _decode_nologits(self, tgt_token_ids, encoder_outputs, padding_mask, training=False):
        """Computes the estimated logits of target token ids, based on the encoded
        source sequences. Note this function should be called in training mode only.

        Args:
          tgt_token_ids: int tensor of shape [batch_size, tgt_seq_len] token ids of
            target sequences.
          encoder_outputs: float tensor of shape [batch_size, src_seq_len,
            hidden_size], the encoded source sequences to be used as reference.
          padding_mask: float tensor of shape [batch_size, 1, 1, src_seq_len],
            populated with either 0 (for tokens to keep) or 1 (for tokens to be
            masked).

        Returns:
          logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size].
        """
        tgt_seq_len = tf.shape(tgt_token_ids)[1]

        # [batch_size, tgt_seq_len, hidden_size]
        tgt_token_embeddings = self._embedding_logits_layer(
            tgt_token_ids, 'embedding')

        # [tgt_seq_len, hidden_size]
        positional_encoding = get_positional_encoding(
            tgt_seq_len, self._hidden_size)
        tgt_token_embeddings += positional_encoding
        tgt_token_embeddings = self._decoder_dropout_layer(
            tgt_token_embeddings, training=True)

        look_ahead_mask = get_look_ahead_mask(tgt_seq_len)

        decoder_outputs = self._decoder(tgt_token_embeddings,
                                        encoder_outputs,
                                        look_ahead_mask,
                                        padding_mask,
                                        training=training)

        # logits = self._embedding_logits_layer(decoder_outputs, 'logits')
        return decoder_outputs

    # def get_embeddings_source(self, src_token_ids):
    #     padding_mask = utils.get_padding_mask(src_token_ids, SOS_ID)
    #     encoder_outputs = self._encode(src_token_ids, padding_mask, training=False)
    #     return encoder_outputs

    def get_embeddings(self, src_token_ids, tgt_token_ids):
        padding_mask = get_padding_mask(src_token_ids, SOS_ID)
        encoder_outputs = self._encode(src_token_ids, padding_mask, training=False)
        # padding_mask_1 = utils.get_padding_mask(tgt_token_ids, SOS_ID)
        decoder_outputs = self._decode_nologits(tgt_token_ids, encoder_outputs, padding_mask, training=False)
        return encoder_outputs, decoder_outputs

    def transduce(self, src_token_ids):
        """Takes as input the source token ids only, and outputs the token ids of
        the decoded target sequences using beam search. Note this function should be
        called in inference mode only.

        Args:
          src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids
            of source sequences.

        Returns:
          decoded_ids: int tensor of shape [batch_size, decoded_seq_len], the token
            ids of the decoded target sequences using beam search.
          scores: float tensor of shape [batch_size], the scores (length-normalized
            log-probs) of the decoded target sequences.
          tgt_tgt_attention: a list of `decoder_stack_size` float tensor of shape
            [batch_size, num_heads, decoded_seq_len, decoded_seq_len],
            target-to-target attention weights.
          tgt_src_attention: a list of `decoder_stack_size` float tensor of shape
            [batch_size, num_heads, decoded_seq_len, src_seq_len], target-to-source
            attention weights.
          src_src_attention: a list of `encoder_stack_size` float tensor of shape
            [batch_size, num_heads, src_seq_len, src_seq_len], source-to-source
            attention weights.
        """
        batch_size, src_seq_len, embed_dim = tf.unstack(tf.shape(src_token_ids))
        max_decode_length = src_seq_len + self._extra_decode_length
        decoding_fn = self._build_decoding_fn(max_decode_length)
        decoding_cache = self._build_decoding_cache(src_token_ids, batch_size)
        sos_ids = tf.ones([batch_size], dtype='int32') * SOS_ID

        bs = beam_search.BeamSearch(decoding_fn,
                                    self._embedding_logits_layer._vocab_size,
                                    batch_size,
                                    self._beam_width,
                                    self._alpha,
                                    max_decode_length,
                                    EOS_ID)

        decoded_ids, scores, decoding_cache = bs.search(sos_ids, decoding_cache)
        print('scores shape:', scores.shape)

        tgt_tgt_attention = [
            decoding_cache['layer_%d' % i]['tgt_tgt_attention'].numpy()[:, 0]
            for i in range(self._decoder_stack_size)]
        tgt_src_attention = [
            decoding_cache['layer_%d' % i]['tgt_src_attention'].numpy()[:, 0]
            for i in range(self._decoder_stack_size)]

        decoded_ids = decoded_ids[:, 0, 1:]
        scores = scores[:, 0]

        src_src_attention = [
            self._encoder._stack[i]._mha._attention_weights.numpy()
            for i in range(self._encoder._stack_size)]

        return (decoded_ids, scores,
                tgt_tgt_attention, tgt_src_attention, src_src_attention)

    def _build_decoding_cache(self, src_token_ids, batch_size):
        """Builds a dictionary that caches previously computed key and value feature
        maps and attention weights of the growing decoded sequence.

        Args:
          src_token_ids: int tensor of shape [batch_size, src_seq_len], token ids of
            source sequences.
          batch_size: int scalar, num of sequences in a batch.

        Returns:
          decoding_cache: dict of entries
            'encoder_outputs': tensor of shape [batch_size, src_seq_len,
              hidden_size],
            'padding_mask': tensor of shape [batch_size, 1, 1, src_seq_len],

            and entries with keys 'layer_0',...,'layer_[decoder_num_layers - 1]'
            where the value associated with key 'layer_*' is a dict with entries
              'k': tensor of shape [batch_size, 0, num_heads, size_per_head],
              'v': tensor of shape [batch_size, 0, num_heads, size_per_head],
              'tgt_tgt_attention': tensor of shape [batch_size, num_heads,
                0, 0],
              'tgt_src_attention': tensor of shape [batch_size, num_heads,
                0, src_seq_len].
        """
        x1 = K.tanh(K.sum(src_token_ids, axis=-1))
        padding_mask = get_padding_mask(x1, SOS_ID)
        encoder_outputs = self._encode(src_token_ids, padding_mask, training=False)
        size_per_head = self._hidden_size // self._num_heads
        src_seq_len = padding_mask.shape[-1]

        decoding_cache = {'layer_%d' % layer:
            {'k':
                tf.zeros([
                    batch_size, 0, self._num_heads, size_per_head
                ], 'float32'),
                'v':
                    tf.zeros([
                        batch_size, 0, self._num_heads, size_per_head
                    ], 'float32'),
                'tgt_tgt_attention':
                    tf.zeros([
                        batch_size, self._num_heads, 0, 0], 'float32'),
                'tgt_src_attention':
                    tf.zeros([
                        batch_size, self._num_heads, 0, src_seq_len], 'float32')

            } for layer in range(self._decoder._stack_size)
        }
        decoding_cache['encoder_outputs'] = encoder_outputs
        decoding_cache['padding_mask'] = padding_mask
        return decoding_cache

    def _build_decoding_fn(self, max_decode_length):
        """Builds the decoding function that will be called in beam search.

        The function steps through the proposed token ids one at a time, and
        generates the logits of next token id over the vocabulary.

        Args:
          max_decode_length: int scalar, the decoded sequences would not exceed
            `max_decode_length`.

        Returns:
          decoding_fn: a callable that outputs the logits of the next decoded token
            ids.
        """
        # [max_decode_length, hidden_size]
        timing_signal = get_positional_encoding(
            max_decode_length, self._hidden_size)
        timing_signal = tf.cast(timing_signal, 'float32')

        def decoding_fn(decoder_input, cache, **kwargs):
            """Computes the logits of the next decoded token ids.

            Args:
              decoder_input: int tensor of shape [batch_size * beam_width, 1], the
                decoded tokens at index `i`.
              cache: dict of entries
                'encoder_outputs': tensor of shape
                  [batch_size * beam_width, src_seq_len, hidden_size],
                'padding_mask': tensor of shape
                  [batch_size * beam_width, 1, 1, src_seq_len],

                and entries with keys 'layer_0',...,'layer_[decoder_num_layers - 1]'
                where the value associated with key 'layer_*' is a dict with entries
                  'k': tensor of shape [batch_size * beam_width, seq_len, num_heads,
                    size_per_head],
                  'v': tensor of shape [batch_size * beam_width, seq_len, num_heads,
                    size_per_head],
                  'tgt_tgt_attention': tensor of shape [batch_size * beam_width,
                    num_heads, seq_len, seq_len],
                  'tgt_src_attention': tensor of shape [batch_size * beam_width,
                    num_heads, seq_len, src_seq_len].
                  Note `seq_len` is the running length of the growing decode sequence.
              kwargs: dict, storing the following additional keyword arguments.
                index -> int scalar tensor, the index of the `decoder_input` in the
                  decoded sequence.

            Returns:
              logits: float tensor of shape [batch_size * beam_width, vocab_size].
              cache: a dict with the same structure as the input `cache`, except that
                the shapes of the values of key `k`, `v`, `tgt_tgt_attention`,
                `tgt_src_attention` are
                [batch_size * beam_width, seq_len + 1, num_heads, size_per_head],
                [batch_size * beam_width, seq_len + 1, num_heads, size_per_head],
                [batch_size * beam_width, num_heads, seq_len + 1, seq_len + 1],
                [batch_size * beam_width, num_heads, seq_len + 1, src_seq_len].
            """
            index = kwargs['index']
            # [batch_size * beam_width, 1, hidden_size]
            decoder_input = self._embedding_logits_layer(decoder_input, 'embedding')
            decoder_input += timing_signal[index:index + 1]

            decoder_outputs = self._decoder(decoder_input,
                                            cache['encoder_outputs'],
                                            tf.zeros((1, 1, 1, index + 1),
                                                     dtype='float32'),
                                            cache['padding_mask'],
                                            training=False,
                                            cache=cache)

            logits = self._embedding_logits_layer(decoder_outputs, mode='logits')
            logits = tf.squeeze(logits, axis=1)
            return logits, cache

        return decoding_fn
