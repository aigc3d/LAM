import numpy as np
import torch
from transformers import WavLMModel
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput
from typing import Optional, Tuple, Union
import torch.nn.functional as F

def linear_interpolation(features, output_len: int):
    features = features.transpose(1, 2)
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)

# the implementation of Wav2Vec2Model is borrowed from https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2Model # noqa: E501
# initialize our encoder with the pre-trained wav2vec 2.0 weights.


class WavLMModel(WavLMModel):
    def __init__(self, config):
        super().__init__(config)

    def _freeze_wav2vec2_parameters(self, do_freeze: bool = True):
        for param in self.parameters():
            param.requires_grad = (not do_freeze)

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        frame_num=None,
        interpolate_pos: int = 0,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if interpolate_pos == 0:
            extract_features = linear_interpolation(
                extract_features, output_len=frame_num)
            
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if interpolate_pos == 1:
            hidden_states = linear_interpolation(
                hidden_states, output_len=frame_num)
            
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )