import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import segmentation_models_pytorch as smp




import torch
import torch.nn as nn

class QuaternionLoss(nn.Module):
    def __init__(self):
        super(QuaternionLoss, self).__init__()

    def forward(self, predicted, ground_truth):

        predicted_normalized = predicted / torch.norm(predicted, dim=1, keepdim=True)
        
        loss = torch.norm(ground_truth - predicted_normalized, dim=1)
        
        return loss.mean()


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
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2,sequence_length=1176):
        super(ExtremeRotationEstimator, self).__init__()
        # Single shared instance of SiameseUNet
        self.embedding_net = SiameseUNet(encoder_name='resnet34', encoder_weights='imagenet')
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length

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
        # self.rotation_query_projection = nn.Linear(3, embed_dim * sequence_length)
        self.rotation_query_projection = nn.Sequential(
            nn.Linear(3, embed_dim), 
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim) 
        )

        # Final MLP for quaternion output
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim*sequence_length, 64),#embed_dim
            nn.ReLU(),
            nn.Linear(64, 3)  # Output size 4 for quaternion representation
        )

    def create_cross_attention_mask(self,sequence_length):
        mask = torch.zeros((sequence_length, sequence_length))
        half = sequence_length // 2
        mask[:half, :half] = float('-inf')  
        mask[half:, half:] = float('-inf')  

        return mask


    def forward(self, img1, img2,rotation_query,mask_input=None):
        emb1 = self.embedding_net(img1)
        emb2 = self.embedding_net(img2)


        #cross attention decoder 0
        decoder0_L_layer1 = self.Decoder_0_L_layer__1(emb2,emb1)
        decoder0_L_layer2 = self.Decoder_0_L_layer__2(decoder0_L_layer1,emb1)
        
        
        decoder0_R_layer1 = self.Decoder_0_R_layer__1(emb1,emb2)
        decoder0_R_layer2 = self.Decoder_0_R_layer__2(decoder0_R_layer1,emb2)
        

        combined_embeddings = torch.cat((decoder0_L_layer2, decoder0_R_layer2), dim=1)

        if mask_input is None:
            sequence_length = combined_embeddings.size(1)  # 2 * H * W
            mask_input = self.create_cross_attention_mask(sequence_length).to(combined_embeddings.device)



        combined_embeddings = combined_embeddings.permute(1, 0, 2)
        
        cross_attention_output = self.transformer_encoder(combined_embeddings,mask=mask_input)# combined_embeddings.permute(1, 0, 2)

        cross_attention_output = cross_attention_output.permute(1, 0, 2)
        # First Transformer Decoder for rotation query

        #approach 1
        # rotation_query = self.rotation_query_projection(rotation_query)
        # rotation_query = rotation_query.unsqueeze(1)
        # rotation_query = rotation_query.expand(-1, cross_attention_output.size(1), -1)

        #approach 2


        rotation_query_projected = self.rotation_query_projection(rotation_query)  # Shape: (batch_size, 98 * embed_dim)
        # rotation_query_projected = rotation_query_projected.view(-1, self.sequence_length, self.embed_dim)
        rotation_query_projected = rotation_query_projected.unsqueeze(1).expand(-1, self.sequence_length, -1)


        # print(f"Shape of cross_attention_output: {cross_attention_output.shape}")  # Expected: (batch_size, 1176, 128)
        # print(f"Shape of rotation_query_projected: {rotation_query_projected.shape}")  



        
        
        decoder1_layer1= self.Decoder_1_layer__1(cross_attention_output,rotation_query_projected)#encoder out as query and rotation as key and value
        decoder1_layer2 = self.Decoder_1_layer__2(decoder1_layer1,rotation_query_projected)

        decoder2_layer1 = self.Decoder_2_layer__1(rotation_query_projected,decoder1_layer2)#rotation as query and decoder1_layer2 as key and value
        decoder2_layer2 = self.Decoder_2_layer__2(rotation_query_projected,decoder2_layer1)

        mlp_input = decoder2_layer2.view(decoder2_layer2.size(0), -1)

        # MLP to predict quaternion
        quaternion = self.mlp(mlp_input)#T_bar.mean(dim=1)


        return quaternion

