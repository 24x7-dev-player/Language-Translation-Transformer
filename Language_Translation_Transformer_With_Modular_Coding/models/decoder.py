import torch.nn as nn
from models.layer_normalization import LayerNormalization

# Building Decoder
# A Decoder can have several Decoder Blocks
class Decoder(nn.Module):
    
    # The Decoder takes in instances of 'DecoderBlock'
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        
        # Storing the 'DecoderBlock's
        self.layers = layers
        self.norm = LayerNormalization() # Layer to normalize the output
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        # Iterating over each DecoderBlock stored in self.layers
        for layer in self.layers:
            # Applies each DecoderBlock to the input 'x' plus the encoder output and source and target masks
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x) # Returns normalized output