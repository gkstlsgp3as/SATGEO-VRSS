import torch.nn as nn
import torch
from collections import namedtuple
from typing import List, Any

# from app.algorithm.s07.utils.data_preprocessing import read_platform


############################### global Attention ######################################
import torch
import torch.nn as nn
from typing import Tuple

# Encoder 정의
class SequenceEncoder(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 128) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): 입력 데이터 (batch_size, seq_len, input_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - outputs: LSTM 출력 (batch_size, seq_len, hidden_size)
                - hidden: 최종 hidden state (num_layers, batch_size, hidden_size)
                - cell: 최종 cell state (num_layers, batch_size, hidden_size)
        """
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


# Attention 정의 (Bahdanau Attention)
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        nn.init.normal_(self.v, mean=0, std=0.1)

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden (torch.Tensor): 디코더의 hidden state (batch_size, hidden_size)
            encoder_outputs (torch.Tensor): 인코더 출력 (batch_size, seq_len, hidden_size)

        Returns:
            torch.Tensor: context vector (batch_size, 1, hidden_size)
        """
        batch_size, seq_len, _ = encoder_outputs.size()

        # hidden 확장
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Energy 계산
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # Attention 가중치 계산
        v = self.v.unsqueeze(0).expand(batch_size, -1).unsqueeze(2)
        attention_weights = torch.bmm(energy, v).squeeze(-1)
        attention_weights = torch.softmax(attention_weights, dim=1).unsqueeze(1)

        # Context 벡터 계산
        context = torch.bmm(attention_weights, encoder_outputs)
        return context


# Decoder 정의
class SequenceDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size + input_size, hidden_size, batch_first=True)
        self.attention = BahdanauAttention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): 디코더 입력 (batch_size, 1, input_size)
            hidden (torch.Tensor): LSTM hidden state (num_layers, batch_size, hidden_size)
            cell (torch.Tensor): LSTM cell state (num_layers, batch_size, hidden_size)
            encoder_outputs (torch.Tensor): 인코더 출력 (batch_size, seq_len, hidden_size)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - predictions: 출력값 (batch_size, 1, output_size)
                - hidden: 업데이트된 hidden state
                - cell: 업데이트된 cell state
        """
        context = self.attention(hidden[-1], encoder_outputs)
        x = torch.cat((x, context), dim=2)
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        predictions = self.fc(torch.cat((outputs, context), dim=2))
        return predictions, hidden, cell


# Seq2Seq 모델 정의
class Seq2SeqModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 5) -> None:
        super().__init__()
        self.encoder = SequenceEncoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = SequenceDecoder(input_size=output_size, hidden_size=hidden_size, output_size=output_size)

    def forward(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
        output_size: int
    ) -> torch.Tensor:
        """
        Args:
            encoder_inputs (torch.Tensor): 인코더 입력 (batch_size, seq_len, input_size)
            decoder_inputs (torch.Tensor): 디코더 입력 (batch_size, seq_len, output_size)
            output_size (int): 디코더 출력 크기

        Returns:
            torch.Tensor: 모델 출력 (batch_size, seq_len, output_size)
        """
        encoder_outputs, hidden, cell = self.encoder(encoder_inputs)
        outputs = []
        decoder_input = decoder_inputs[:, 0, :output_size].unsqueeze(1)

        for t in range(decoder_inputs.size(1)):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs.append(output)
            teacher_forcing_ratio = 0.5

            if torch.rand(1).item() < teacher_forcing_ratio and t != decoder_inputs.size(1) - 1:
                decoder_input = decoder_inputs[:, t + 1, :output_size].unsqueeze(1)
            else:
                decoder_input = output.detach()

        return torch.cat(outputs, dim=1)

    def inference(
        self,
        encoder_inputs: torch.Tensor,
        decoder_inputs: torch.Tensor,
        decoder_length: int,
        output_size: int
    ) -> torch.Tensor:
        """
        Inference 시퀀스 예측

        Args:
            encoder_inputs (torch.Tensor): 인코더 입력 (batch_size, seq_len, input_size)
            decoder_inputs (torch.Tensor): 초기 디코더 입력 (batch_size, 1, output_size)
            decoder_length (int): 예측 길이
            output_size (int): 디코더 출력 크기

        Returns:
            torch.Tensor: 예측 결과 (batch_size, decoder_length, output_size)
        """
        encoder_outputs, hidden, cell = self.encoder(encoder_inputs)
        outputs = []
        decoder_input = decoder_inputs[:, :, :output_size]

        for _ in range(decoder_length):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs.append(output)
            decoder_input = output.detach()

        return torch.cat(outputs, dim=1)[:, :, :output_size]
