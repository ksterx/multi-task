from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from torch.nn import CrossEntropyLoss


@dataclass
class CausalLMAndBranchOutputWithPast:
    loss: Optional[torch.FloatTensor] = None
    loss_head1: Optional[torch.FloatTensor] = None
    loss_head2: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    head2_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    branch_logits: torch.FloatTensor = None


class Gemma2ForMultiHeadCausalLM(Gemma2ForCausalLM):
    head2_loss_weight: float = 2.0
    nth_token: int = 10

    def __init__(
        self,
        config,
    ):
        super().__init__(config)

        self.lm_head2 = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.LayerNorm(config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(config.hidden_size, config.vocab_size),
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,  # 自分で loss を計算するので一旦 None にしておく
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

        # 1. Head1 logits
        head1_logits = self.lm_head(hidden_states)

        # 2. Head2 logits
        head2_logits = self.lm_head2(hidden_states)

        # 3. Total loss
        total_loss = None
        head1_loss = None
        head2_loss = None

        if labels is not None:
            # 1. Head1 loss (next token prediction)
            shift_logits = head1_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            head1_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # 2. Head2 loss (next nth token prediction)
            shift_logits = head2_logits[..., : -self.nth_token, :].contiguous()
            shift_labels = labels[..., self.nth_token :].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            head2_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        # 3. Total loss
        if head1_loss is not None and head2_loss is not None:
            total_loss = head1_loss + self.head2_loss_weight * head2_loss
        elif head1_loss is not None:
            total_loss = head1_loss
        elif head2_loss is not None:
            total_loss = head2_loss

        # Output
        return CausalLMAndBranchOutputWithPast(
            loss=total_loss,
            loss_head1=head1_loss,
            loss_head2=head2_loss,
            logits=head1_logits,
            head2_logits=head2_logits,
        )


# class Gemma3WithBranch(Gemma3ForCausalLM):
#     def __init__(
#         self, config, branch_hidden_size=256, branch_output_size=3, dropout_rate=0.2
#     ):
#         super().__init__(config)

#         # 1) まず全パラメータを freeze
#         for param in self.parameters():
#             param.requires_grad = False

#         # 2) branch を定義 - より深いネットワークと dropout を追加
#         hidden_size = self.model.config.hidden_size
#         self.branch = torch.nn.Sequential(
#             torch.nn.Linear(hidden_size, branch_hidden_size),
#             torch.nn.LayerNorm(branch_hidden_size),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(dropout_rate),
#             torch.nn.Linear(branch_hidden_size, branch_hidden_size // 2),
#             torch.nn.LayerNorm(branch_hidden_size // 2),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(dropout_rate),
#             torch.nn.Linear(branch_hidden_size // 2, branch_output_size),
#         )

#         # 3) branch のみ学習を有効化
#         for param in self.branch.parameters():
#             param.requires_grad = True

#         # 必要に応じて、branch_output_sizeを config に持たせておくとよい
#         self.config.branch_output_size = branch_output_size

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values=None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         cache_position: Optional[torch.LongTensor] = None,
#         logits_to_keep: Union[int, torch.Tensor] = 0,
#         branch_labels=None,
#         **loss_kwargs,
#     ):
#         output_attentions = (
#             output_attentions
#             if output_attentions is not None
#             else self.config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states
#             if output_hidden_states is not None
#             else self.config.output_hidden_states
#         )
#         return_dict = (
#             return_dict if return_dict is not None else self.config.use_return_dict
#         )

#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             cache_position=cache_position,
#             **loss_kwargs,
#         )

#         # hidden_states: (batch_size, seq_len, hidden_size)
#         hidden_states = outputs[0]

#         # 通常の言語モデルのログits計算
#         slice_indices = (
#             slice(-logits_to_keep, None)
#             if isinstance(logits_to_keep, int)
#             else logits_to_keep
#         )
#         logits = self.lm_head(hidden_states[:, slice_indices, :])
#         if self.config.final_logit_softcapping is not None:
#             logits = logits / self.config.final_logit_softcapping
#             logits = torch.tanh(logits)
#             logits = logits * self.config.final_logit_softcapping

#         # branch用の出力（最後のトークンの hidden state を利用する例）
#         # branch_input = torch.mean(hidden_states, dim=1)
#         branch_input = hidden_states
#         branch_logits = self.branch(branch_input)

#         # ---- 損失計算 ----
#         loss = None
#         loss_backbone = None
#         loss_branch = None

#         if labels is not None:
#             # 親クラスの self.loss_function は
#             # (logits, labels, vocab_size, **loss_kwargs) の形で呼ぶという想定
#             loss_backbone = self.loss_function(
#                 logits, labels, self.vocab_size, **loss_kwargs
#             )

#         if branch_labels is not None:
#             bsz, seq_len, n_class = branch_logits.size()
#             branch_logits_reshaped = branch_logits.view(bsz * seq_len, n_class)
#             branch_labels_reshaped = branch_labels.view(bsz * seq_len)

#             # 無効箇所 (PAD) を除外するなら ignore_index を設定
#             loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
#             loss_branch = loss_fn(branch_logits_reshaped, branch_labels_reshaped)

#         if loss_backbone is not None and loss_branch is not None:
#             # branchの損失に5倍の重みを付ける
#             loss = loss_backbone + 5.0 * loss_branch
#         elif loss_backbone is not None:
#             loss = loss_backbone
#         elif loss_branch is not None:
#             loss = loss_branch

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         # CausalLMAndBranchOutputWithPast は branch_logits を保持可能
#         return CausalLMAndBranchOutputWithPast(
#             loss=loss,
#             loss_backbone=loss_backbone,
#             loss_branch=loss_branch,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#             branch_logits=branch_logits,
#         )
