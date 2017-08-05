import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules
from onmt.modules import aeq
from onmt.modules.Gate import ContextGateFactory
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.nn.functional as F
import math


class Embeddings(nn.Module):
    def __init__(self, opt, dicts, feature_dicts=None):
        self.positional_encoding = opt.position_encoding
        if self.positional_encoding:
            self.pe = self.make_positional_encodings(opt.word_vec_size, 5000)
            if len(opt.gpus) > 0:
                self.pe.cuda()

        self.word_vec_size = opt.word_vec_size

        super(Embeddings, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=onmt.Constants.PAD)
        # Word embeddings.
        self.dropout = nn.Dropout(p=opt.dropout)
        self.feature_dicts = feature_dicts
        # Feature embeddings.
        if self.feature_dicts:
            self.feature_luts = nn.ModuleList([
                nn.Embedding(feature_dict.size(),
                             opt.feature_vec_size,
                             padding_idx=onmt.Constants.PAD)
                for feature_dict in feature_dicts])

            # MLP on features and words.
            self.activation = nn.ReLU()
            self.linear = onmt.modules.BottleLinear(
                opt.word_vec_size +
                len(feature_dicts) * opt.feature_vec_size,
                opt.word_vec_size)
        else:
            self.feature_luts = nn.ModuleList([])

    def make_positional_encodings(self, dim, max_len):
        pe = torch.FloatTensor(max_len, 1, dim).fill_(0)
        for i in range(dim):
            for j in range(max_len):
                k = float(j) / (10000.0 ** (2.0*i / float(dim)))
                pe[j, 0, i] = math.cos(k) if i % 2 == 1 else math.sin(k)
        return pe

    def load_pretrained_vectors(self, emb_file):
        if emb_file is not None:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, src_input):
        """
        Embed the words or utilize features and MLP.

        Args:
            src_input (LongTensor): len x batch x nfeat

        Return:
            emb (FloatTensor): len x batch x input_size
        """
        word = self.word_lut(src_input[:, :, 0])
        emb = word
        if self.feature_dicts:
            features = [feature_lut(src_input[:, :, j+1])
                        for j, feature_lut in enumerate(self.feature_luts)]

            # Apply one MLP layer.
            emb = self.activation(
                self.linear(torch.cat([word] + features, -1)))

        if self.positional_encoding:
            emb = emb + Variable(self.pe[:emb.size(0), :1, :emb.size(2)]
                                 .expand_as(emb))
            emb = self.dropout(emb)

        return emb


class Encoder(nn.Module):
    """
    Encoder recurrent neural network.
    """
    def __init__(self, opt, dicts, feature_dicts=None):
        """
        Args:
            opt: Model options.
            dicts (`Dict`): The src dictionary
            features_dicts (`[Dict]`): List of src feature dictionaries.
        """
        # Number of rnn layers.
        self.layers = opt.layers

        # Use a bidirectional model.
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0

        # Size of the encoder RNN.
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.embeddings = Embeddings(opt, dicts, feature_dicts)

        # The Encoder RNN.
        self.encoder_layer = opt.encoder_layer

        if self.encoder_layer == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerEncoder(self.hidden_size, opt)
                 for i in range(opt.layers)])
        else:
            self.rnn = getattr(nn, opt.rnn_type)(
                 input_size, self.hidden_size,
                 num_layers=opt.layers,
                 dropout=opt.dropout,
                 bidirectional=opt.brnn)

        self.topic2vec = None
        if opt.topic2vec:
            self.topic_matrix = nn.Parameter(torch.FloatTensor(opt.topic_vec_size, opt.topic_num).uniform_(-0.01, 0.01), requires_grad=True)
            self.topic2vec = Topic2Vec(opt)

    def forward(self, input, lengths=None, hidden=None):
        """
        Args:
            input (LongTensor): len x batch x nfeat
            lengths (LongTensor): batch
            hidden: Initial hidden state.

        Returns:
            hidden_t (FloatTensor): Pair of layers x batch x rnn_size - final
                                    Encoder state
            outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        # CHECKS
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            _, n_batch_ = lengths.size()
            aeq(n_batch, n_batch_)
        # END CHECKS

        emb = self.embeddings(input)

        # word_embedding + topic_embedding
        topic_vec = None
        topic_dist = None
        if self.topic2vec is not None:
            #Returns:
            #   topic_vec: batch, topic_vec_size, 1
            #   topic_dist: batch x topic_num x 1
            topic_vec, topic_dist = self.topic2vec(emb, self.topic_matrix)
            # print "topic_matrix", self.topic_matrix
            # print "topic_vec", topic_vec
            # print "emb", emb
            topic_vec_repeat = topic_vec.repeat(1, 1, emb.size(0)).permute(2, 0, 1).contiguous()
            # print "topic_vec_repeat", topic_vec_repeat
            emb = emb + topic_vec_repeat
            # print "emb", emb


        s_len, n_batch, vec_size = emb.size()

        if self.encoder_layer == "mean":
            # No RNN, just take mean as final state.
            mean = emb.mean(0) \
                   .expand(self.layers, n_batch, vec_size)
            return (mean, mean), emb, topic_vec

        elif self.encoder_layer == "transformer":
            # Self-attention tranformer.
            out = emb.transpose(0, 1).contiguous()
            for i in range(self.layers):
                out = self.transformer[i](out, input[:, :, 0].transpose(0, 1))
            return Variable(emb.data), out.transpose(0, 1).contiguous(), topic_vec
        else:
            # Standard RNN encoder.
            packed_emb = emb
            if lengths is not None:
                # Lengths data is wrapped inside a Variable.
                lengths = lengths.data.view(-1).tolist()
                packed_emb = pack(emb, lengths)
            outputs, hidden_t = self.rnn(packed_emb, hidden)
            if lengths:
                outputs = unpack(outputs)[0]
            return hidden_t, outputs, topic_vec


class Decoder(nn.Module):
    """
    Decoder + Attention recurrent neural network.
    """

    def __init__(self, opt, dicts):
        """
        Args:
            opt: model options
            dicts: Target `Dict` object
        """
        self.layers = opt.layers
        self.decoder_layer = opt.decoder_layer
        self._coverage = opt.coverage_attn
        self.hidden_size = opt.rnn_size
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.embeddings = Embeddings(opt, dicts, None)

        if self.decoder_layer == "transformer":
            self.transformer = nn.ModuleList(
                [onmt.modules.TransformerDecoder(self.hidden_size, opt)
                 for _ in range(opt.layers)])
        else:
            if opt.rnn_type == "LSTM":
                stackedCell = onmt.modules.StackedLSTM
            else:
                stackedCell = onmt.modules.StackedGRU
            self.rnn = stackedCell(opt.layers, input_size,
                                   opt.rnn_size, opt.dropout)
            self.context_gate = None
            if opt.context_gate is not None:
                self.context_gate = ContextGateFactory(
                    opt.context_gate, input_size,
                    opt.rnn_size, opt.rnn_size,
                    opt.rnn_size
                )

            # zeng: like context_gate which don't use the attention embedding from source, but the topic vec.
            # self.topic_gate = None
            # if opt.topic_gate is not None:
            # 	self.topic_gate = ContextGateFactory(
            # 		opt.topic_gate, input_size,
            # 		opt.rnn_size, opt.rnn_size,
            # 		opt.topic_vec_size)

        self.dropout = nn.Dropout(opt.dropout)

        # Std attention layer.
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size,
                                                 coverage=self._coverage,
                                                 attn_type=opt.attention_type)

        # Separate Copy Attention.
        self._copy = False
        if opt.copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                opt.rnn_size, attn_type=opt.attention_type)
            self._copy = True

        # self.topic2vec_decoder = opt.topic2vec_decoder
        # self.topic_type_decoder = opt.topic_type_decoder

    def forward(self, input, src, context, state, topic=None):
        """
        Forward through the decoder.

        Args:
            input (LongTensor):  (len x batch) -- Input tokens
            src (LongTensor)
            context:  (src_len x batch x rnn_size)  -- Memory bank
            state: an object initializing the decoder.

        Returns:
            outputs: (len x batch x rnn_size)
            final_states: an object of the same form as above
            attns: Dictionary of (src_len x batch)
        """
        # CHECKS
        t_len, n_batch = input.size()
        s_len, n_batch_, _ = src.size()
        s_len_, n_batch__, _ = context.size()
        aeq(n_batch, n_batch_, n_batch__)
        # aeq(s_len, s_len_)
        # END CHECKS
        if self.decoder_layer == "transformer":
            if state.previous_input:
                input = torch.cat([state.previous_input.squeeze(2), input], 0)

        emb = self.embeddings(input.unsqueeze(2))

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []

        # Setup the different types of attention.
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        if self.decoder_layer == "transformer":
            # Tranformer Decoder.
            assert isinstance(state, TransformerDecoderState)
            output = emb.transpose(0, 1).contiguous()
            src_context = context.transpose(0, 1).contiguous()
            for i in range(self.layers):
                output, attn \
                    = self.transformer[i](output, src_context,
                                          src[:, :, 0].transpose(0, 1),
                                          input.transpose(0, 1))
            outputs = output.transpose(0, 1).contiguous()
            if state.previous_input:
                outputs = outputs[state.previous_input.size(0):]
                attn = attn[:, state.previous_input.size(0):].squeeze()
                attn = torch.stack([attn])
            attns["std"] = attn
            if self._copy:
                attns["copy"] = attn
            state = TransformerDecoderState(input.unsqueeze(2))
        else:
            assert isinstance(state, RNNDecoderState)
            output = state.input_feed.squeeze(0)
            hidden = state.hidden
            # CHECKS
            n_batch_, _ = output.size()
            aeq(n_batch, n_batch_)
            # END CHECKS

            coverage = state.coverage.squeeze(0) \
                if state.coverage is not None else None

            # Standard RNN decoder.
            for i, emb_t in enumerate(emb.split(1)):
                emb_t = emb_t.squeeze(0)
                if self.input_feed:
                    emb_t = torch.cat([emb_t, output], 1)

                rnn_output, hidden = self.rnn(emb_t, hidden)
                attn_output, attn = self.attn(rnn_output,
                                              context.transpose(0, 1))
                if self.context_gate is not None:
                    output = self.context_gate(
                        emb_t, rnn_output, attn_output
                    )
                    output = self.dropout(output)
                else:
                    output = self.dropout(attn_output)
                outputs += [output]
                attns["std"] += [attn]

                # COVERAGE
                if self._coverage:
                    coverage = (coverage + attn) if coverage else attn
                    attns["coverage"] += [coverage]

                # COPY
                if self._copy:
                    _, copy_attn = self.copy_attn(output,
                                                  context.transpose(0, 1))
                    attns["copy"] += [copy_attn]
            state = RNNDecoderState(hidden, output.unsqueeze(0),
                                    coverage.unsqueeze(0)
                                    if coverage is not None else None)
            outputs = torch.stack(outputs)
            for k in attns:
                attns[k] = torch.stack(attns[k])

            # zeng: topic + deocder
            # outputs: len x batch x rnn_size
            # topic_vec: batch x topic_vec_size x 1
            # if self.topic2vec_decoder:
            #     topic_repeat = topic.repeat(1, 1, outputs.size(0))
            #     topic_repeat = topic_repeat.permute(2, 0, 1)
            #     if self.topic_type_decoder == "sum":
            #         assert outputs.size(2) == topic.size(1)
            #         outputs = torch.add(outputs, topic_repeat)
            #     elif self.topic_type_decoder == "concat":
            #         outputs = torch.cat((outputs, topic_repeat), 2)

        return outputs, state, attns


class Topic2Vec(nn.Module):
    """docstring for Topic2Vec"""
    def __init__(self, opt):
        super(Topic2Vec, self).__init__()
        self.topic_matrix = nn.Parameter(torch.FloatTensor(opt.topic_vec_size, opt.topic_num), requires_grad=True)
        
        self.mix_of_experts = nn.ModuleList([nn.Linear() for i in opt.experts_num])

        self.topic_linear = nn.Linear()

    def forward(self, input):
        topic_proportion = self.mix_of_experts(input)

        topic_vec = self.topic_matrix.mul(topic_proportion)

        return topic_vec, topic_proportion


class NMTModel(nn.Module):
    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def _fix_enc_hidden(self, h):
        """
        The encoder hidden is  (layers*directions) x batch x dim
        We need to convert it to layers x batch x (directions*dim)
        """
        if self.encoder.num_directions == 2:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def init_decoder_state(self, context, enc_hidden):
        if self.decoder.decoder_layer == "transformer":
            return TransformerDecoderState()
        elif isinstance(enc_hidden, tuple):
            dec = RNNDecoderState(tuple([self._fix_enc_hidden(enc_hidden[i])
                                         for i in range(len(enc_hidden))]))
        else:
            dec = RNNDecoderState(self._fix_enc_hidden(enc_hidden))
        dec.init_input_feed(context, self.decoder.hidden_size)
        return dec

    def forward(self, src, tgt, lengths, dec_state=None):
        """
        Args:
            src, tgt, lengths
            dec_state: A decoder state object

        Returns:
            outputs (FloatTensor): (len x batch x rnn_size) -- Decoder outputs.
            attns (FloatTensor): Dictionary of (src_len x batch)
            dec_hidden (FloatTensor): tuple (1 x batch x rnn_size)
                                      Init hidden state
        """
        src = src
        tgt = tgt[:-1]  # exclude last target from inputs
        
        # 1. encoder
        enc_hidden, context, topic_vec = self.encoder(src, lengths)

        # 2. decoder
        enc_state = self.init_decoder_state(context, enc_hidden)
        out, dec_state, attns = self.decoder(tgt, src, context,
                                             enc_state if dec_state is None
                                             else dec_state, topic_vec)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return out, attns, dec_state


class DecoderState(object):
    def detach(self):
        for h in self.all:
            if h is not None:
                h.detach_()

    def repeatBeam_(self, beamSize):
        self._resetAll([Variable(e.data.repeat(1, beamSize, 1))
                        for e in self.all])

    def beamUpdate_(self, idx, positions, beamSize):
        for e in self.all:
            a, br, d = e.size()
            sentStates = e.view(a, beamSize, br // beamSize, d)[:, :, idx]
            sentStates.data.copy_(
                sentStates.data.index_select(1, positions))


class RNNDecoderState(DecoderState):
    def __init__(self, rnnstate, input_feed=None, coverage=None):
        # all objects are X x batch x dim
        # or X x (beam * sent) for beam search
        if not isinstance(rnnstate, tuple):
            self.hidden = (rnnstate,)
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage
        self.all = self.hidden + (self.input_feed,)

    def init_input_feed(self, context, rnn_size):
        batch_size = context.size(1)
        h_size = (batch_size, rnn_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(),
                                   requires_grad=False).unsqueeze(0)
        self.all = self.hidden + (self.input_feed,)

    def _resetAll(self, all):
        vars = [Variable(a.data if isinstance(a, Variable) else a,
                         volatile=True) for a in all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]
        self.all = self.hidden + (self.input_feed,)


class TransformerDecoderState(DecoderState):
    def __init__(self, input=None):
        # all objects are X x batch x dim
        # or X x (beam * sent) for beam search
        self.previous_input = input
        self.all = (self.previous_input,)

    def _resetAll(self, all):
        vars = [(Variable(a.data if isinstance(a, Variable) else a,
                          volatile=True))
                for a in all]
        self.previous_input = vars[0]
        self.all = (self.previous_input,)

    def repeatBeam_(self, beamSize):
        pass

class Topic2Vec(nn.Module):

    def __init__(self, opt):
        super(Topic2Vec, self).__init__()
        self.topic_type = opt.topic_type
        if self.topic_type == "mean":
            self.linear = onmt.modules.BottleLinear(opt.topic_vec_size, opt.topic_num)
        elif self.topic_type == "cnn":
            self.windows = [2, 3, 4]
            self.convs = nn.ModuleList([nn.Conv2d(1, opt.kernel_num, (w, opt.topic_vec_size)) for w in self.windows])
            self.full_connected = nn.Linear(len(self.windows)*opt.kernel_num, opt.topic_num)
        elif self.topic_type == "moe":
        	self.experts = nn.ModuleList([onmt.modules.BottleLinear(opt.word_vec_size, opt.topic_vec_size) for i in xrange(opt.experts_num)])
        	self.gating = onmt.modules.BottleSoftmax()
        
        self.dropout = nn.Dropout(opt.dropout)
        self.softmax = nn.Softmax()
    
    """
    Topic2Vec:
        Args:
            input: len x batch x word_vec_size
            topic_matrix: topic_vec_size x topic_num
        Returns:
            topic_vec: batch, topic_vec_size, 1
            topic_dist: batch x topic_num x 1
    """
    def forward(self, input, topic_matrix):
        if self.topic_type == "mean":
            # print "input", input
            topic_weights = self.linear(input) # len x batch x topic_num
            # print "topic_weights", topic_weights.sum(0)
            topic_dist = self.softmax(topic_weights.sum(0).squeeze(0)).unsqueeze(2)  # batch x topic_num x 1
            # print "topic_dist", topic_dist
        # print "topic_matrix", topic_matrix
        elif self.topic_type == "cnn":
            x = [F.relu(conv(input.t().unsqueeze(1))).squeeze(3) for conv in self.convs]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
            x = torch.cat(x, 1)
            x = self.dropout(x)
            topic_dist = self.softmax(self.full_connected(x)).unsqueeze(2)
            # print "topic_dist", topic_dist
        elif self.topic_type == "moe":
        	word_topic_weights = torch.stack([linear(input) for linear in self.linearList])
        	gating = nn.BottleSoftmax(word_topic_weights).mul(word_topic_weights)

        	topic_vec = torch.sum(gating, 0)


        topic_matrix_repeat = topic_matrix.expand(topic_dist.size(0), topic_matrix.size(0), topic_matrix.size(1)) # batch x topic_vec_size x topic_num
        topic_vec = torch.bmm(topic_matrix_repeat, topic_dist) # batch x topic_vec_size x 1
        return topic_vec, topic_dist


class Memnn(nn.Module):
    """
    Memnn:
        Args:
            context: seq_len, batch, rnn_size
        Returns:
            topic_aware_vec: batch, rnn_size, 1
            topic_dist: batch, topic_num
    """

    def __init__(self, opt):
        super(Memnn, self).__init__()
        self.hops = opt.hops
        self.softmax = nn.Softmax()

    def forward(self, context, topic_matrix):
        output = torch.sum(context.permute(1, 2, 0), 2).contiguous()
        # print "output: ", output

        # batch_size, topic_num, topic_vec_size
        batch_topic = topic_matrix.expand(output.size(0), topic_matrix.size(0), topic_matrix.size(1)).transpose(1, 2)
        # print "batch_topic: ", batch_topic

        # topic_vec_size == rnn_size
        for i in range(self.hops):
            # input: batch_size * hidden
            topic_simi = torch.bmm(batch_topic,
                                   output)  # batch_size, topic_num, topic_vec_size * batch_size, rnn_size, 1
            # print "topic_simi: ", topic_simi

            topic_dist = self.softmax(topic_simi.squeeze(2))  # topic_dist: batch_size * topic_num
            # print "topic_dist", topic_dist

            # output: batch_size, topic_vec_size, topic_num * batch_size, topic_num, 1
            output = torch.bmm(batch_topic.permute(0, 2, 1), topic_dist.unsqueeze(2)) + output
            # print "output: ", output

        return output, topic_dist