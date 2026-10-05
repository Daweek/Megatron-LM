# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared runtime state for Megatron-FSDP modules."""

from typing import cast

import torch
from torch import nn

from .indexed_order import IndexedOrder
from .module import FsdpModule


class FsdpContext:
    """Runtime stream and prefetch state shared by FSDP roots constructed together."""

    allgather_stream: torch.cuda.Stream
    reduce_scatter_stream: torch.cuda.Stream
    # HFSDP/HSDP need explicit last-microbatch state. First-microbatch state is
    # unnecessary because each parameter group tracks whether model_weight is stale
    # after syncing from main_weight.
    is_last_microbatch: bool
    use_symmetric_memory: bool
    unify_communication_stream: bool
    # Static orders used to drive all-gather prefetch. We may want to switch to
    # capturing runtime order if static module order proves too fragile. Each
    # FsdpModule tracks its own materialized state via ``FsdpModule._unshard_event``.
    forward_order: IndexedOrder[FsdpModule]
    backward_order: IndexedOrder[FsdpModule]
    # The optimizer runs on the current stream and must wait for reductions on
    # this context's reduce-scatter stream. Each context owns its own stream, so
    # independent roots sharing a context need only one completion callback.
    _post_backward_hook_registered: bool

    def __init__(
        self,
        device: torch.device,
        use_symmetric_memory: bool = False,
        unify_communication_stream: bool = False,
        parameter_to_owner: dict[nn.Parameter, int] | None = None,
    ) -> None:
        """Create rank-local runtime state for FSDP modules on ``device``.

        Args:
            device: Device on which this context schedules communication.
            use_symmetric_memory: Whether modules constructed in this context allocate
                communication staging buffers from PyTorch's NCCL symmetric-memory pool.
            unify_communication_stream: Whether all-gathers and reduce-scatters share one
                communication stream to reduce peak transient memory.
            parameter_to_owner: Construction-time TensorAtomic owner assignments. See
                ``fully_shard_context``.
        """
        self.is_last_microbatch = True
        self.use_symmetric_memory = use_symmetric_memory
        self.unify_communication_stream = unify_communication_stream
        self.forward_order = IndexedOrder()
        self.backward_order = IndexedOrder()
        self._post_backward_hook_registered = False
        # Construction-only; empty after finalization.
        self._registered_modules: list[FsdpModule] = []
        self.parameter_to_owner = parameter_to_owner
        self._is_finalized = False
        self.allgather_stream = torch.cuda.Stream(device)
        if unify_communication_stream:
            # A unified stream lets an all-gather reuse the storage released by a
            # preceding reduce-scatter.
            self.reduce_scatter_stream = self.allgather_stream
        else:
            self.reduce_scatter_stream = torch.cuda.Stream(device)

    def register_module(self, module: FsdpModule) -> None:
        """Register a module constructed in this context."""
        if self._is_finalized:
            raise RuntimeError("Cannot register an FSDP module after its context is finalized.")
        self._registered_modules.append(module)

    def finalize(self) -> None:
        """Finalize roots, names, and cross-root prefetch orders."""
        if self._is_finalized:
            raise RuntimeError("FSDP context is already finalized.")

        children: set[FsdpModule] = set()
        for module in self._registered_modules:
            _collect_fsdp_children(cast(nn.Module, module), children)
        # FsdpModules that are not descendants of any other FsdpModule.
        roots = [module for module in self._registered_modules if module not in children]

        for root in roots:
            root._is_root = True
            for name, module in cast(nn.Module, root).named_modules():
                if not isinstance(module, FsdpModule):
                    continue
                module._name = name
                self.forward_order.append(module)

        for root in reversed(roots):
            _collect_backward_order(cast(nn.Module, root), self.backward_order)

        self._registered_modules.clear()
        self.parameter_to_owner = None
        self._is_finalized = True

    def ensure_finalized(self) -> None:
        """Raise if construction has not completed for this context."""
        if not self._is_finalized:
            raise RuntimeError(
                "FSDP context is not finalized. Exit fully_shard_context before running forward."
            )

    def current_stream(self) -> torch.cuda.Stream:
        """Current stream on this context's device."""
        return torch.cuda.current_stream(self.allgather_stream.device)

    def post_backward(self) -> None:
        """Order current-stream consumers after this context's gradient reductions."""
        self.current_stream().wait_stream(self.reduce_scatter_stream)
        self._post_backward_hook_registered = False

    def register_post_backward_hook(self) -> None:
        """Register one context-level final callback for the current backward.

        Multiple FSDP roots can share this context. Waiting for the
        reduce-scatter stream in each root's ``post_backward()`` would prevent
        one root's backward compute from overlapping another root's gradient
        reductions. Wait once at context-level autograd completion instead.
        """

        if self._post_backward_hook_registered:
            return
        self._post_backward_hook_registered = True

        # TODO(wujingyue): Switch to torch.autograd.graph.queue_callback() when Megatron-LM
        # requires a PyTorch version that includes it:
        # https://github.com/pytorch/pytorch/pull/193958
        torch.autograd.Variable._execution_engine.queue_callback(self.post_backward)


def _collect_backward_order(module: nn.Module, order: IndexedOrder[FsdpModule]) -> None:
    """Collect one root's static backward prefetch order."""
    if isinstance(module, FsdpModule):
        order.append(module)

    for child in reversed(list(module.children())):
        _collect_backward_order(child, order)


def _collect_fsdp_children(module: nn.Module, children: set[FsdpModule]) -> None:
    """Collect the nearest FSDP descendants of ``module``."""
    for child in module.children():
        if isinstance(child, FsdpModule):
            children.add(child)
        else:
            _collect_fsdp_children(child, children)
