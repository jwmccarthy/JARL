from abc import ABC, abstractmethod
from typing import Self

import torch as th
import torch.nn as nn


NativeState = th.Tensor | tuple[th.Tensor, th.Tensor]


class Recurrent(nn.Module, ABC):
    rnn_type: type[nn.RNNBase]

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        init_func=None,
    ) -> None:
        super().__init__()

        if hidden_size < 1 or num_layers < 1:
            raise ValueError("recurrent dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if num_layers == 1 and dropout:
            raise ValueError("dropout requires more than one recurrent layer")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bias = bias
        self.init_func = init_func
        self.feats = hidden_size
        self.built = False

    def build(self, in_dim: int) -> Self:
        if in_dim < 1:
            raise ValueError("input dimension must be positive")

        self.rnn = self.rnn_type(
            input_size=in_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bias=self.bias,
        )
        self.input_size = in_dim
        self.built = True
        if self.init_func:
            initialized = self.init_func(self.rnn)
            self.rnn = self.rnn if initialized is None else initialized
        else:
            self._initialize()

        return self

    def initial_state(
        self,
        batch_size: int,
        device: str | th.device | None = None,
        dtype: th.dtype | None = None,
    ) -> th.Tensor:
        self._require_built()
        if batch_size < 1:
            raise ValueError("batch size must be positive")

        parameter = next(self.rnn.parameters())
        device = parameter.device if device is None else th.device(device)
        dtype = parameter.dtype if dtype is None else dtype

        return th.zeros(
            self._state_shape(batch_size),
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        value: th.Tensor,
        state: th.Tensor | None = None,
        reset: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        self._require_built()
        sequence, single_step = self._as_sequence(value)
        state = self._prepare_state(state, sequence)
        reset = self._prepare_reset(reset, sequence, single_step)
        native_state = self._to_native(state)

        if reset is None or not reset.any().item():
            output, native_state = self.rnn(sequence, native_state)
        else:
            output, native_state = self._unroll(sequence, native_state, reset)

        output = output.squeeze(0) if single_step else output
        return output, self._from_native(native_state)

    def _as_sequence(self, value: th.Tensor) -> tuple[th.Tensor, bool]:
        if value.ndim not in (2, 3):
            raise ValueError("recurrent input must be [batch, features] or [time, batch, features]")
        if value.shape[-1] != self.input_size:
            raise ValueError(
                f"input has {value.shape[-1]} features, expected {self.input_size}"
            )

        single_step = value.ndim == 2
        return (value.unsqueeze(0) if single_step else value), single_step

    def _prepare_state(
        self,
        state: th.Tensor | None,
        sequence: th.Tensor,
    ) -> th.Tensor:
        batch_size = sequence.shape[1]
        expected = self._state_shape(batch_size)

        if state is None:
            return self.initial_state(
                batch_size,
                device=sequence.device,
                dtype=sequence.dtype,
            )
        if state.shape != expected:
            raise ValueError(
                f"state has shape {tuple(state.shape)}, expected {expected}"
            )
        if state.device != sequence.device:
            raise ValueError("state must match input device")
        if state.dtype != sequence.dtype:
            state = state.to(dtype=sequence.dtype)

        return state

    def _prepare_reset(
        self,
        reset: th.Tensor | None,
        sequence: th.Tensor,
        single_step: bool,
    ) -> th.Tensor | None:
        if reset is None:
            return None

        expected = (sequence.shape[1],) if single_step else sequence.shape[:2]
        if reset.shape != expected:
            raise ValueError(
                f"reset has shape {tuple(reset.shape)}, expected {tuple(expected)}"
            )

        reset = reset.to(device=sequence.device, dtype=th.bool)
        return reset.unsqueeze(0) if single_step else reset

    def _unroll(
        self,
        sequence: th.Tensor,
        state: NativeState,
        reset: th.Tensor,
    ) -> tuple[th.Tensor, NativeState]:
        outputs = []

        for index in range(len(sequence)):
            state = self._reset_state(state, reset[index])
            output, state = self.rnn(sequence[index : index + 1], state)
            outputs.append(output)

        return th.cat(outputs), state

    def _reset_state(self, state: NativeState, reset: th.Tensor) -> NativeState:
        keep = (~reset)[None, :, None]

        if isinstance(state, tuple):
            return state[0] * keep, state[1] * keep
        return state * keep

    def _initialize(self) -> None:
        gates = 3 if self.rnn_type is nn.GRU else 4

        for name, parameter in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.zeros_(parameter)
                continue

            for gate in parameter.chunk(gates, dim=0):
                if "weight_hh" in name:
                    nn.init.orthogonal_(gate)
                else:
                    nn.init.xavier_uniform_(gate)

    def _require_built(self) -> None:
        if not self.built:
            raise RuntimeError("recurrent module must be built before use")

    @abstractmethod
    def _state_shape(self, batch_size: int) -> tuple[int, ...]:
        ...

    @abstractmethod
    def _to_native(self, state: th.Tensor) -> NativeState:
        ...

    @abstractmethod
    def _from_native(self, state: NativeState) -> th.Tensor:
        ...


class GRU(Recurrent):
    rnn_type = nn.GRU

    def _state_shape(self, batch_size: int) -> tuple[int, ...]:
        return batch_size, self.num_layers, self.hidden_size

    def _to_native(self, state: th.Tensor) -> th.Tensor:
        return state.transpose(0, 1).contiguous()

    def _from_native(self, state: NativeState) -> th.Tensor:
        assert isinstance(state, th.Tensor)
        return state.transpose(0, 1).contiguous()


class LSTM(Recurrent):
    rnn_type = nn.LSTM

    def _state_shape(self, batch_size: int) -> tuple[int, ...]:
        return batch_size, 2, self.num_layers, self.hidden_size

    def _to_native(self, state: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        hidden = state[:, 0].transpose(0, 1).contiguous()
        cell = state[:, 1].transpose(0, 1).contiguous()
        return hidden, cell

    def _from_native(self, state: NativeState) -> th.Tensor:
        assert isinstance(state, tuple)
        hidden, cell = state
        return th.stack(
            (
                hidden.transpose(0, 1),
                cell.transpose(0, 1),
            ), dim=1
        ).contiguous()
