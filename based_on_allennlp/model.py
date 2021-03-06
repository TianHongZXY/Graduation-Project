from allennlp_models.generation import ComposedSeq2Seq, SimpleSeq2Seq
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, LstmSeq2SeqEncoder, Seq2SeqEncoder
import torch
from allennlp_models.generation.modules.seq_decoders import AutoRegressiveSeqDecoder
from allennlp_models.generation.modules.decoder_nets import LstmCellDecoderNet


def create_seq2seqmodel(vocab, src_embedders, tgt_embedders, hidden_dim=100, num_layers=1,
                        max_decoding_steps=20, beam_size=1, use_bleu=True, device=0):
    encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(src_embedders.get_output_dim(), hidden_dim, batch_first=True))
    decoder_net = LstmCellDecoderNet(decoding_dim=encoder.get_output_dim(), target_embedding_dim=tgt_embedders.get_output_dim())
    decoder = AutoRegressiveSeqDecoder(vocab, decoder_net, max_decoding_steps, tgt_embedders, beam_size=beam_size)
    model = ComposedSeq2Seq(vocab, src_embedders, encoder, decoder)
    # model = SimpleSeq2Seq(vocab, src_embedders, encoder, max_decoding_steps, target_namespace="target_tokens",
    #                        target_embedding_dim=tgt_embedders.get_output_dim(), beam_size=beam_size,
    #                        use_bleu=use_bleu)
    model.to(device)
    return model
