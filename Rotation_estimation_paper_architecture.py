import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import segmentation_models_pytorch as smp

class TransformerDecoderBlock(nn.Module):#cross attention implementation
    
    def __init__(self, d_model=128, nhead=4, dim_feedforward=512, dropout=0.1):#dim_feedforward =2048
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, query, key_value, query_mask=None, key_value_mask=None, 
                query_key_padding_mask=None, key_value_key_padding_mask=None):
        attn_output, _ = self.cross_attn(
            query, 
            key_value, 
            key_value, 
            attn_mask=query_mask, 
            key_padding_mask=query_key_padding_mask
        )
        query = query + self.dropout1(attn_output)
        query = self.norm1(query)


        ffn_output = self.ffn(query)
        query = query + ffn_output
        query = self.norm2(query)

        return query
    

class SiameseUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, encoder_depth=5):
        super(SiameseUNet, self).__init__()
        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1  
        )
        self.encoder_depth = encoder_depth

        self.projection = nn.Linear(512, 128)
    def forward(self, x):
        features = self.base_model.encoder(x)
        out = features[-1].permute(0, 2, 3, 1)
        out = out.flatten(1, 2)  # Flatten spatial dimensions
        projected = self.projection(out)  # Project to 128 channels
        return projected#features[-1].flatten(2)

        
class ExtremeRotationEstimator(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2):
        super(ExtremeRotationEstimator, self).__init__()
        # Single shared instance of SiameseUNet
        self.embedding_net = SiameseUNet(encoder_name='resnet34', encoder_weights='imagenet')

        # Transformer Encoder for cross-attention computation
        self.Decoder_0_L_layer__1 = TransformerDecoderBlock(d_model=128, nhead=4)
        self.Decoder_0_L_layer__2 = TransformerDecoderBlock(d_model=128, nhead=4)

        self.Decoder_0_R_layer__1 = TransformerDecoderBlock(d_model=128, nhead=4)
        self.Decoder_0_R_layer__2 = TransformerDecoderBlock(d_model=128, nhead=4)
        
        
        self.Decoder_1_layer__1 = TransformerDecoderBlock(d_model=128, nhead=4)
        self.Decoder_1_layer__2 = TransformerDecoderBlock(d_model=128, nhead=4)
        
        self.Decoder_2_layer__1 = TransformerDecoderBlock(d_model=128, nhead=4)
        self.Decoder_2_layer__2 = TransformerDecoderBlock(d_model=128, nhead=4)




        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # encoder with mask 
        # self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        #Projection layer to adjust `rotation_query` dimensions
        #apprach1
        #self.rotation_query_projection = nn.Linear(4, 128)
        #Approach 2
        self.rotation_query_projection = nn.Linear(4, embed_dim * 98)
        # Final MLP for quaternion output
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),#embed_dim
            nn.ReLU(),
            nn.Linear(64, 4)  # Output size 4 for quaternion representation
        )

    def forward(self, img1, img2,mask_input):
        emb1 = self.embedding_net(img1)
        emb2 = self.embedding_net(img2)

        #cross attention decoder 0
        decoder0_L_layer1 = self.Decoder_0_L_layer__1(emb2,emb1)
        decoder0_L_layer2 = self.Decoder_0_L_layer__2(decoder0_L_layer1,emb1)
        
        
        decoder0_R_layer1 = self.Decoder_0_R_layer__1(emb1,emb2)
        decoder0_R_layer2 = self.Decoder_0_R_layer__2(decoder0_R_layer1,emb2)
        

        combined_embeddings = torch.cat((decoder0_L_layer2, decoder0_R_layer2), dim=1)

        
        cross_attention_output = self.transformer_encoder(combined_embeddings,mask=mask_input)

        # First Transformer Decoder for rotation query

        #approach 1
        # rotation_query = self.rotation_query_projection(rotation_query)
        # rotation_query = rotation_query.unsqueeze(1)
        # rotation_query = rotation_query.expand(-1, cross_attention_output.size(1), -1)

        #approach 2
        rotation_query = self.rotation_query_projection(rotation_query)  # Shape: (batch_size, 98 * embed_dim)
        rotation_query = rotation_query.view(-1, 98, 128)
        
        
        decoder1_layer1= self.Decoder_1_layer__1(cross_attention_output,rotation_query)#encoder out as query and rotation as key and value
        decoder1_layer2 = self.Decoder_1_layer__2(decoder1_layer1,rotation_query)

        decoder2_layer1 = self.Decoder_2_layer__1(rotation_query,decoder1_layer2)#rotation as query and decoder1_layer2 as key and value
        decoder2_layer2 = self.Decoder_2_layer__2(rotation_query,decoder2_layer1)


        # MLP to predict quaternion
        quaternion = self.mlp(decoder2_layer2)#T_bar.mean(dim=1)

        return quaternion
