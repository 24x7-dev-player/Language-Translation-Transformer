from models.input_embeddings import InputEmbeddings
from models.positional_encoding import PositionalEncoding
from models.multi_head_attention_block import MultiHeadAttentionBlock
from models.feed_forward_block import FeedForwardBlock
from models.encoder_block import EncoderBlock
from models.decoder_block import DecoderBlock
from models.encoder import Encoder
from models.decoder import Decoder
from models.projection_layer import ProjectionLayer
import torch.nn as nn
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("translation.log"),
                        logging.StreamHandler()
                    ])


# Creating the Transformer Architecture
class Transformer(nn.Module):
    
    # This takes in the encoder and decoder, as well the embeddings for the source and target language.
    # It also takes in the Positional Encoding for the source and target language, as well as the projection layer
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    # Encoder     
    def encode(self, src, src_mask):
        src = self.src_embed(src) # Applying source embeddings to the input source language
        src = self.src_pos(src) # Applying source positional encoding to the source embeddings
        return self.encoder(src, src_mask) # Returning the source embeddings plus a source mask to prevent attention to certain elements
    
    # Decoder
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt) # Applying target embeddings to the input target language (tgt)
        tgt = self.tgt_pos(tgt) # Applying target positional encoding to the target embeddings
        
        # Returning the target embeddings, the output of the encoder, and both source and target masks
        # The target mask ensures that the model won't 'see' future elements of the sequence
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    # Applying Projection Layer with the Softmax function to the Decoder output
    def project(self, x):
        return self.projection_layer(x)


# Building & Initializing Transformer
# Definin function and its parameter, including model dimension, number of encoder and decoder stacks, heads, etc.
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    
    # Creating Embedding 
    logging.info("Creating Embedding ... ")
    src_embed = InputEmbeddings(d_model, src_vocab_size) # Source language (Source Vocabulary to 512-dimensional vectors)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size) # Target language (Target Vocabulary to 512-dimensional vectors)
    logging.info("Embedding Successfully!")
    
    
    # Creating Positional Encoding layers
    logging.info("Creating Positional Encoding ... ")
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout) # Positional encoding for the source language embeddings
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout) # Positional encoding for the target language embeddings
    logging.info("Positional Encoding Successfully!")
    
    
    # Creating EncoderBlocks
    logging.info("Creating Encoder Block ... ")
    encoder_blocks = [] # Initial list of empty EncoderBlocks
    logging.info("Creating Multi Head Attention or Self Attention ... ")
    logging.info("Creating Feed Forward block ... ")
    logging.info("Combine layers into EncoderBlock with Residual Network ... ")
    logging.info("Layer Normalization Starting ... ")
    for _ in range(N): # Iterating 'N' times to create 'N' EncoderBlocks (N = 6)
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Self-Attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # FeedForward
        
        # Combine layers into an EncoderBlock
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block) # Appending EncoderBlock to the list of EncoderBlocks
        
    # Creating DecoderBlocks
    logging.info("Creating Encoder Block Successfully!")
    logging.info("Creating Decoder Block ... ")
    logging.info("Creating Multi Head Attention or Self Attention ... ")
    logging.info("Creating Feed Forward block ... ")
    logging.info("Combine layers into DecoderBlock with Residual Network ... ")
    logging.info("Layer Normalization Starting ... ")
    decoder_blocks = [] # Initial list of empty DecoderBlocks
    for _ in range(N): # Iterating 'N' times to create 'N' DecoderBlocks (N = 6)
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Self-Attention
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout) # Cross-Attention
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # FeedForward
        
        # Combining layers into a DecoderBlock
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block) # Appending DecoderBlock to the list of DecoderBlocks
        
    # Creating the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks lists
    logging.info("Creating Decoder Block Successfully!")
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    logging.info("Creating the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks Successfully")
    
    # Creating projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size) # Map the output of Decoder to the Target Vocabulary Space
    logging.info("Creating Projection layer Successfully!")
    
    # Creating the transformer by combining everything above
    logging.info("Creating the transformer by combining everything")
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    logging.info("Parameters Initialization ... ")
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer # Assembled and initialized Transformer. Ready to be trained and validated!

