import math
import os.path

import torch

import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta

from models.encoder.wav2vec import Wav2Vec2Model
from models.encoder.wavlm import WavLMModel

from models.builder import MODELS

from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config

@MODELS.register_module("Audio2Expression")
class Audio2Expression(nn.Module):
    def __init__(self,
                 device: torch.device = None,
                 pretrained_encoder_type: str = 'wav2vec',
                 pretrained_encoder_path: str = '',
                 wav2vec2_config_path: str = '',
                 num_identity_classes: int = 0,
                 identity_feat_dim: int = 64,
                 hidden_dim: int = 512,
                 expression_dim: int = 52,
                 norm_type: str = 'ln',
                 decoder_depth: int = 3,
                 use_transformer: bool = False,
                 num_attention_heads: int = 8,
                 num_transformer_layers: int = 6,
                 ):
        super().__init__()

        self.device = device

        # Initialize audio feature encoder
        if pretrained_encoder_type == 'wav2vec':
            if os.path.exists(pretrained_encoder_path):
                self.audio_encoder = Wav2Vec2Model.from_pretrained(pretrained_encoder_path)
            else:
                config = Wav2Vec2Config.from_pretrained(wav2vec2_config_path)
                self.audio_encoder = Wav2Vec2Model(config)
            encoder_output_dim = 768
        elif pretrained_encoder_type == 'wavlm':
            self.audio_encoder = WavLMModel.from_pretrained(pretrained_encoder_path)
            encoder_output_dim = 768
        else:
            raise NotImplementedError(f"Encoder type {pretrained_encoder_type} not supported")

        self.audio_encoder.feature_extractor._freeze_parameters()
        self.feature_projection = nn.Linear(encoder_output_dim, hidden_dim)

        self.identity_encoder = AudioIdentityEncoder(
            hidden_dim,
            num_identity_classes,
            identity_feat_dim,
            use_transformer,
            num_attention_heads,
            num_transformer_layers
        )

        self.decoder = nn.ModuleList([
            nn.Sequential(*[
                ConvNormRelu(hidden_dim, hidden_dim, norm=norm_type)
                for _ in range(decoder_depth)
            ])
        ])

        self.output_proj = nn.Linear(hidden_dim, expression_dim)

    def freeze_encoder_parameters(self, do_freeze=False):

        for name, param in self.audio_encoder.named_parameters():
            if('feature_extractor' in name):
                param.requires_grad = False
            else:
                param.requires_grad = (not do_freeze)

    def forward(self, input_dict):

        if 'time_steps' not in input_dict:
            audio_length = input_dict['input_audio_array'].shape[1]
            time_steps = math.ceil(audio_length / 16000 * 30)
        else:
            time_steps = input_dict['time_steps']

        # Process audio through encoder
        audio_input = input_dict['input_audio_array'].flatten(start_dim=1)
        hidden_states = self.audio_encoder(audio_input, frame_num=time_steps).last_hidden_state

        # Project features to hidden dimension
        audio_features = self.feature_projection(hidden_states).transpose(1, 2)

        # Process identity-conditioned features
        audio_features = self.identity_encoder(audio_features, identity=input_dict['id_idx'])

        # Refine features through decoder
        audio_features = self.decoder[0](audio_features)

        # Generate output parameters
        audio_features = audio_features.permute(0, 2, 1)
        expression_params = self.output_proj(audio_features)

        return torch.sigmoid(expression_params)


class AudioIdentityEncoder(nn.Module):
    def __init__(self,
                 hidden_dim,
                 num_identity_classes=0,
                 identity_feat_dim=64,
                 use_transformer=False,
                 num_attention_heads = 8,
                 num_transformer_layers = 6,
                 dropout_ratio=0.1,
                 ):
        super().__init__()

        in_dim = hidden_dim + identity_feat_dim
        self.id_mlp = nn.Conv1d(num_identity_classes, identity_feat_dim, 1, 1)
        self.first_net = SeqTranslator1D(in_dim, hidden_dim,
                                         min_layers_num=3,
                                         residual=True,
                                         norm='ln'
                                         )
        self.grus = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout_ratio)

        self.use_transformer = use_transformer
        if(self.use_transformer):
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_attention_heads, dim_feedforward= 2 * hidden_dim, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

    def forward(self,
                audio_features: torch.Tensor,
                identity: torch.Tensor = None,
                time_steps: int = None) -> tuple:

        audio_features = self.dropout(audio_features)
        identity = identity.reshape(identity.shape[0], -1, 1).repeat(1, 1, audio_features.shape[2]).to(torch.float32)
        identity = self.id_mlp(identity)
        audio_features = torch.cat([audio_features, identity], dim=1)

        x = self.first_net(audio_features)

        if time_steps is not None:
            x = F.interpolate(x, size=time_steps, align_corners=False, mode='linear')

        if(self.use_transformer):
            x = x.permute(0, 2, 1)
            x = self.transformer_encoder(x)
            x = x.permute(0, 2, 1)

        return x

class ConvNormRelu(nn.Module):
    '''
    (B,C_in,H,W) -> (B, C_out, H, W)
    there exist some kernel size that makes the result is not H/s
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 type='1d',
                 leaky=False,
                 downsample=False,
                 kernel_size=None,
                 stride=None,
                 padding=None,
                 p=0,
                 groups=1,
                 residual=False,
                 norm='bn'):
        '''
        conv-bn-relu
        '''
        super(ConvNormRelu, self).__init__()
        self.residual = residual
        self.norm_type = norm
        # kernel_size = k
        # stride = s

        if kernel_size is None and stride is None:
            if not downsample:
                kernel_size = 3
                stride = 1
            else:
                kernel_size = 4
                stride = 2

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(stride, tuple):
                padding = tuple(int((kernel_size - st) / 2) for st in stride)
            elif isinstance(kernel_size, tuple) and isinstance(stride, int):
                padding = tuple(int((ks - stride) / 2) for ks in kernel_size)
            elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
                padding = tuple(int((ks - st) / 2) for ks, st in zip(kernel_size, stride))
            else:
                padding = int((kernel_size - stride) / 2)

        if self.residual:
            if downsample:
                if type == '1d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )
                elif type == '2d':
                    self.residual_layer = nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )
            else:
                if in_channels == out_channels:
                    self.residual_layer = nn.Identity()
                else:
                    if type == '1d':
                        self.residual_layer = nn.Sequential(
                            nn.Conv1d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        )
                    elif type == '2d':
                        self.residual_layer = nn.Sequential(
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding
                            )
                        )

        in_channels = in_channels * groups
        out_channels = out_channels * groups
        if type == '1d':
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups)
            self.norm = nn.BatchNorm1d(out_channels)
            self.dropout = nn.Dropout(p=p)
        elif type == '2d':
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups)
            self.norm = nn.BatchNorm2d(out_channels)
            self.dropout = nn.Dropout2d(p=p)
        if norm == 'gn':
            self.norm = nn.GroupNorm(2, out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        if self.norm_type == 'ln':
            out = self.dropout(self.conv(x))
            out = self.norm(out.transpose(1,2)).transpose(1,2)
        else:
            out = self.norm(self.dropout(self.conv(x)))
        if self.residual:
            residual = self.residual_layer(x)
            out += residual
        return self.relu(out)

""" from https://github.com/ai4r/Gesture-Generation-from-Trimodal-Context.git """
class SeqTranslator1D(nn.Module):
    '''
    (B, C, T)->(B, C_out, T)
    '''
    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=None,
                 stride=None,
                 min_layers_num=None,
                 residual=True,
                 norm='bn'
                 ):
        super(SeqTranslator1D, self).__init__()

        conv_layers = nn.ModuleList([])
        conv_layers.append(ConvNormRelu(
            in_channels=C_in,
            out_channels=C_out,
            type='1d',
            kernel_size=kernel_size,
            stride=stride,
            residual=residual,
            norm=norm
        ))
        self.num_layers = 1
        if min_layers_num is not None and self.num_layers < min_layers_num:
            while self.num_layers < min_layers_num:
                conv_layers.append(ConvNormRelu(
                    in_channels=C_out,
                    out_channels=C_out,
                    type='1d',
                    kernel_size=kernel_size,
                    stride=stride,
                    residual=residual,
                    norm=norm
                ))
                self.num_layers += 1
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.conv_layers(x)


def audio_chunking(audio: torch.Tensor, frame_rate: int = 30, chunk_size: int = 16000):
    """
    :param audio: 1 x T tensor containing a 16kHz audio signal
    :param frame_rate: frame rate for video (we need one audio chunk per video frame)
    :param chunk_size: number of audio samples per chunk
    :return: num_chunks x chunk_size tensor containing sliced audio
    """
    samples_per_frame = 16000 // frame_rate
    padding = (chunk_size - samples_per_frame) // 2
    audio = torch.nn.functional.pad(audio.unsqueeze(0), pad=[padding, padding]).squeeze(0)
    anchor_points = list(range(chunk_size//2, audio.shape[-1]-chunk_size//2, samples_per_frame))
    audio = torch.cat([audio[:, i-chunk_size//2:i+chunk_size//2] for i in anchor_points], dim=0)
    return audio

""" https://github.com/facebookresearch/meshtalk """
class MeshtalkEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, model_name: str = 'audio_encoder'):
        """
        :param latent_dim: size of the latent audio embedding
        :param model_name: name of the model, used to load and save the model
        """
        super().__init__()

        self.melspec = ta.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=2048, win_length=800, hop_length=160, n_mels=80
        )

        conv_len = 5
        self.convert_dimensions = torch.nn.Conv1d(80, 128, kernel_size=conv_len)
        self.weights_init(self.convert_dimensions)
        self.receptive_field = conv_len

        convs = []
        for i in range(6):
            dilation = 2 * (i % 3 + 1)
            self.receptive_field += (conv_len - 1) * dilation
            convs += [torch.nn.Conv1d(128, 128, kernel_size=conv_len, dilation=dilation)]
            self.weights_init(convs[-1])
        self.convs = torch.nn.ModuleList(convs)
        self.code = torch.nn.Linear(128, latent_dim)

        self.apply(lambda x: self.weights_init(x))

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            try:
                torch.nn.init.constant_(m.bias, .01)
            except:
                pass

    def forward(self, audio: torch.Tensor):
        """
        :param audio: B x T x 16000 Tensor containing 1 sec of audio centered around the current time frame
        :return: code: B x T x latent_dim Tensor containing a latent audio code/embedding
        """
        B, T = audio.shape[0], audio.shape[1]
        x = self.melspec(audio).squeeze(1)
        x = torch.log(x.clamp(min=1e-10, max=None))
        if T == 1:
            x = x.unsqueeze(1)

        # Convert to the right dimensionality
        x = x.view(-1, x.shape[2], x.shape[3])
        x = F.leaky_relu(self.convert_dimensions(x), .2)

        # Process stacks
        for conv in self.convs:
            x_ = F.leaky_relu(conv(x), .2)
            if self.training:
                x_ = F.dropout(x_, .2)
            l = (x.shape[2] - x_.shape[2]) // 2
            x = (x[:, :, l:-l] + x_) / 2

        x = torch.mean(x, dim=-1)
        x = x.view(B, T, x.shape[-1])
        x = self.code(x)

        return {"code": x}

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=15, max_seq_len=64):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1) # (1, repeat_num, period, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # print(self.pe.shape, x.shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GeneratorTransformer(nn.Module):
    def __init__(self,
                 n_poses,
                 each_dim: list,
                 dim_list: list,
                 training=True,
                 device=None,
                 identity=False,
                 num_classes=0,
                 ):
        super().__init__()

        self.training = training
        self.device = device
        self.gen_length = n_poses

        norm = 'ln'
        in_dim = 256
        out_dim = 256

        self.encoder_choice = 'faceformer'

        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")  # "vitouphy/wav2vec2-xls-r-300m-phoneme""facebook/wav2vec2-base-960h"
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, in_dim)

        self.audio_middle = AudioEncoder(in_dim, out_dim, False, num_classes)

        self.dim_list = dim_list

        self.decoder = nn.ModuleList()
        self.final_out = nn.ModuleList()

        self.hidden_size = 768
        self.transformer_de_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=4,
            dim_feedforward=self.hidden_size*2,
            batch_first=True
            )
        self.face_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=4)
        self.feature2face = nn.Linear(256, self.hidden_size)

        self.position_embeddings = PeriodicPositionalEncoding(self.hidden_size, period=64, max_seq_len=64)
        self.id_maping = nn.Linear(12,self.hidden_size)


        self.decoder.append(self.face_decoder)
        self.final_out.append(nn.Linear(self.hidden_size, 32))

    def forward(self, in_spec, gt_poses=None, id=None, pre_state=None, time_steps=None):
        if gt_poses is None:
            time_steps = 64
        else:
            time_steps = gt_poses.shape[1]

        # vector, hidden_state = self.audio_encoder(in_spec, pre_state, time_steps=time_steps)
        if self.encoder_choice == 'meshtalk':
            in_spec = audio_chunking(in_spec.squeeze(-1), frame_rate=30, chunk_size=16000)
            feature = self.audio_encoder(in_spec.unsqueeze(0))["code"].transpose(1, 2)
        elif self.encoder_choice == 'faceformer':
            hidden_states = self.audio_encoder(in_spec.reshape(in_spec.shape[0], -1), frame_num=time_steps).last_hidden_state
            feature = self.audio_feature_map(hidden_states).transpose(1, 2)
        else:
            feature, hidden_state = self.audio_encoder(in_spec, pre_state, time_steps=time_steps)

        feature, _ = self.audio_middle(feature, id=None)
        feature = self.feature2face(feature.permute(0,2,1))

        id = id.unsqueeze(1).repeat(1,64,1).to(torch.float32)
        id_feature = self.id_maping(id)
        id_feature = self.position_embeddings(id_feature)

        for i in range(self.decoder.__len__()):
            mid = self.decoder[i](tgt=id_feature, memory=feature)
            out = self.final_out[i](mid)

        return out, None

def linear_interpolation(features, output_len: int):
    features = features.transpose(1, 2)
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)

def init_biased_mask(n_head, max_seq_len, period):

    def get_slopes(n):

        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.div(
        torch.arange(start=0, end=max_seq_len,
                     step=period).unsqueeze(1).repeat(1, period).view(-1),
        period,
        rounding_mode='floor')
    bias = -torch.flip(bias, dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i + 1] = bias[-(i + 1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len,
                                  max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


# Alignment Bias
def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, i] = 0
    return (mask == 1).to(device=device)


# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=3000):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, period, d_model)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self):
        super(BaseModel, self).__init__()
        # self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *x):
        """Forward pass logic.

        :return: Model output
        """
        raise NotImplementedError

    def freeze_model(self, do_freeze: bool = True):
        for param in self.parameters():
            param.requires_grad = (not do_freeze)

    def summary(self, logger, writer=None):
        """Model summary."""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size())
                      for p in model_parameters]) / 1e6  # Unit is Mega
        logger.info('===>Trainable parameters: %.3f M' % params)
        if writer is not None:
            writer.add_text('Model Summary',
                            'Trainable parameters: %.3f M' % params)


"""https://github.com/X-niper/UniTalker"""
class UniTalkerDecoderTransformer(BaseModel):

    def __init__(self, out_dim, identity_num, period=30, interpolate_pos=1) -> None:
        super().__init__()
        self.learnable_style_emb = nn.Embedding(identity_num, out_dim)
        self.PPE = PeriodicPositionalEncoding(
            out_dim, period=period, max_seq_len=3000)
        self.biased_mask = init_biased_mask(
            n_head=4, max_seq_len=3000, period=period)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=out_dim,
            nhead=4,
            dim_feedforward=2 * out_dim,
            batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=1)
        self.interpolate_pos = interpolate_pos

    def forward(self, hidden_states: torch.Tensor, style_idx: torch.Tensor,
                frame_num: int):
        style_idx = torch.argmax(style_idx, dim=1)
        obj_embedding = self.learnable_style_emb(style_idx)
        obj_embedding = obj_embedding.unsqueeze(1).repeat(1, frame_num, 1)
        style_input = self.PPE(obj_embedding)
        tgt_mask = self.biased_mask.repeat(style_idx.shape[0], 1, 1)[:, :style_input.shape[1], :style_input.
                                    shape[1]].clone().detach().to(
                                        device=style_input.device)
        memory_mask = enc_dec_mask(hidden_states.device, style_input.shape[1],
                                   frame_num)
        feat_out = self.transformer_decoder(
            style_input,
            hidden_states,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask)
        if self.interpolate_pos == 2:
            feat_out = linear_interpolation(feat_out, output_len=frame_num)
        return feat_out