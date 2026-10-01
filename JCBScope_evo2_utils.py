"""
JCBScope_evo2_utils.py
======================
JacobianScopes utilities adapted for Evo 2 (StripedHyena 2 architecture).

Original transformer-based version: https://github.com/AntonioLiu97/JacobianScopes

Key architectural differences from the LLaMA/Pythia version
------------------------------------------------------------
* Evo 2 is loaded via the `evo2` package (backed by Vortex), NOT from HuggingFace
  AutoModelForCausalLM.  The top-level wrapper is `Evo2('evo2_7b')`.
* `evo2_wrapper.model`  – the Vortex StripedHyena2 backbone (PyTorch nn.Module)
* `evo2_wrapper.tokenizer` – a CharLevelTokenizer (no HF tokenizer)
* Embedding layer lives at  `backbone.embedding_layer`  (nn.Embedding)
* Language-model head lives at  `backbone.unembed`       (nn.Linear, weight-tied
  with the embedding when `tie_embeddings=True` in config)
* No attention_mask argument – StripedHyena2 is fully convolutional / recurrent;
  all tokens are processed unconditionally.
* Vortex's backbone forward signature is:
      hidden, _ = backbone(input_ids)   # input_ids: [B, L] int tensor
  It does NOT accept `inputs_embeds` directly.
* To inject a modified embedding (required by JacobianScopes), we hook the
  embedding lookup with a one-shot forward hook that replaces the output of
  `backbone.embedding_layer` with our custom tensor.
* Output token ids are integers in [0, 512) using Evo 2's byte-level vocab.
  `evo2_wrapper.tokenizer.tokenize(seq)` → list[int]
  `evo2_wrapper.tokenizer.detokenize(ids)` → str

Install requirements
--------------------
    # Full install (40B / 1B require FP8 / Hopper):
    conda install -c nvidia cuda-nvcc cuda-cudart-dev
    conda install -c conda-forge transformer-engine-torch=2.3.0
    pip install flash-attn==2.8.0.post2 --no-build-isolation
    pip install evo2

    # Light install (7B only, no Transformer Engine):
    pip install evo2[light]

TransformerEngine / FP8 note
-----------------------------
This module sets ``NVTE_DISABLE_NVRTC=1`` at import time.  TransformerEngine's
FP8 linear layers attempt to JIT-compile CUDA kernels via NVRTC during any
autograd-enabled forward pass, which requires ``cuda_runtime.h`` to be on
``NVTE_CUDA_INCLUDE_DIR``.  On systems where that header is unavailable (common
in Singularity/Apptainer containers), the compilation fails with:
    "Could not find cuda_runtime.h …"
Setting ``NVTE_DISABLE_NVRTC=1`` tells TE to skip the NVRTC path and use its
pre-compiled kernels instead, restoring backward-pass functionality.
"""

import os
# Disable TransformerEngine NVRTC JIT before any TE submodule is imported.
# This must come before `import evo2` or `from transformer_engine import …`.
os.environ.setdefault("NVTE_DISABLE_NVRTC", "1")

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# TransformerEngine NVRTC guard
# ---------------------------------------------------------------------------

def _disable_te_nvrtc():
    """
    Disable TransformerEngine's NVRTC JIT path at runtime, even if TE was
    already imported before this module (e.g. via evo2's own imports).

    TE checks ``os.environ["NVTE_DISABLE_NVRTC"]`` lazily at the point where
    it tries to compile a kernel, so setting it here is still effective for
    all subsequent forward/backward passes even if TE is already loaded.
    """
    os.environ["NVTE_DISABLE_NVRTC"] = "1"
    try:
        # Also patch the internal flag if TE exposes it
        import transformer_engine.pytorch.cpp_extensions as _te_ext
        if hasattr(_te_ext, "_nvrtc_enabled"):
            _te_ext._nvrtc_enabled = False
    except Exception:
        pass  # TE not installed or attribute doesn't exist — no-op

_disable_te_nvrtc()


def get_backbone(evo2_wrapper):
    """
    Return the raw Vortex StripedHyena2 backbone from an Evo2 wrapper object.

    Parameters
    ----------
    evo2_wrapper : evo2.Evo2
        Loaded via e.g. ``Evo2('evo2_7b')``.

    Returns
    -------
    torch.nn.Module
        The underlying Vortex model (``evo2_wrapper.model``).
    """
    return evo2_wrapper.model


def get_embedding_layer(backbone):
    """
    Return the token-embedding nn.Embedding from the Vortex backbone.

    In Vortex the attribute is ``backbone.embedding_layer``.
    """
    if hasattr(backbone, "embedding_layer"):
        return backbone.embedding_layer
    raise AttributeError(
        "Could not find the embedding layer on the Vortex backbone. "
        "Expected 'backbone.embedding_layer'. "
        f"Available top-level attributes: {list(backbone._modules.keys())}"
    )


def get_unembed_weight(backbone):
    """
    Return the unembedding weight matrix ``[vocab_size, d_model]`` from the
    Vortex backbone, handling both possible representations:

    * ``backbone.unembed`` is an ``nn.Linear``  → return its ``.weight``
    * ``backbone.unembed`` is a Vortex ``Lambda`` wrapper (observed in
      evo2_1b_base) where the actual weight is tied to the embedding table
      → fall back to ``backbone.embedding_layer.weight``

    Always returns a ``[V, d]`` float tensor (the weight, not a module).
    """
    if hasattr(backbone, "unembed"):
        u = backbone.unembed
        # Case 1: plain nn.Linear
        if isinstance(u, nn.Linear):
            return u.weight                           # [V, d]
        # Case 2: Lambda / custom wrapper that exposes .weight
        if hasattr(u, "weight"):
            return u.weight
    # Fallback: tied embeddings – use embedding_layer.weight directly
    if hasattr(backbone, "embedding_layer"):
        return backbone.embedding_layer.weight        # [V, d]
    raise AttributeError(
        "Could not resolve unembed weight from backbone. "
        f"Top-level submodules: {list(backbone._modules.keys())}"
    )


# Keep a thin shim so old call-sites (get_unembed) still work;
# callers that previously did  unembed.weight.T  should now call
# get_unembed_weight(backbone).T  directly.
def get_unembed(backbone):
    """Deprecated shim – returns the weight tensor directly via get_unembed_weight."""
    return get_unembed_weight(backbone)


# ---------------------------------------------------------------------------
# Inference-tensor rehabilitation
# ---------------------------------------------------------------------------

def rehabilitate_inference_tensors(backbone):
    """
    Clone any "inference tensors" in backbone parameters and buffers back into
    normal autograd-capable tensors, in-place.

    Background
    ----------
    PyTorch marks any tensor *created* inside a ``torch.no_grad()`` block as an
    **inference tensor** (``tensor.is_inference() == True``).  These tensors
    can never appear in an autograd graph — even outside ``no_grad`` — because
    they were never assigned a grad_fn slot.

    Vortex's Hyena filter layers store learnable poles/residues as ``nn.Parameter``
    objects (e.g. ``residues``, ``log_poles``).  If a ``no_grad`` forward pass
    is run first (e.g. during the inference checks earlier in the debug script),
    PyTorch may materialise intermediate parameter views as inference tensors
    that get cached.  When a subsequent autograd forward pass tries to use them
    in a computation that needs to be saved for backward, it raises:

        RuntimeError: Inference tensors cannot be saved for backward.

    The fix recommended by PyTorch is ``.clone()``, which produces a fresh
    normal tensor that can participate in autograd.  We do this for every
    parameter and buffer in the backbone, replacing them in-place via
    ``param.data = param.data.clone()``.

    Call this once before any backward-pass forward run.

    Parameters
    ----------
    backbone : torch.nn.Module
        The Vortex StripedHyena backbone.
    """
    n_fixed = 0
    for name, param in backbone.named_parameters():
        if param.data.is_inference():
            param.data = param.data.clone()
            n_fixed += 1
    for name, buf in backbone.named_buffers():
        if buf is not None and buf.is_inference():
            # buffers are not nn.Parameter so we set via the module directly
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                mod = backbone.get_submodule(parts[0])
                mod.register_buffer(parts[1], buf.clone())
            n_fixed += 1
    return n_fixed


# ---------------------------------------------------------------------------
# Embedding lookup (device-safe)
# ---------------------------------------------------------------------------

def embedding_lookup(indices, embedding_layer):
    """
    Look up embeddings at given token indices, with device-safe handling.

    Uses ``F.embedding`` directly to avoid any Accelerate hook device mismatches
    when the model is sharded with ``device_map='auto'``.

    Parameters
    ----------
    indices : torch.Tensor
        Integer tensor of token ids.
    embedding_layer : nn.Embedding
        The embedding module.

    Returns
    -------
    torch.Tensor
        Float embedding vectors.
    """
    w_device = embedding_layer.weight.device
    if indices.device != w_device:
        indices = indices.to(w_device)
    return F.embedding(indices, embedding_layer.weight)


# ---------------------------------------------------------------------------
# Embedding injection via hook
# ---------------------------------------------------------------------------

class _EmbedOverrideHook:
    """
    One-shot forward hook that replaces the output of ``backbone.embedding_layer``
    with a pre-built embed tensor, then removes itself.

    This is the mechanism by which JacobianScopes injects a modified embedding
    (residual perturbation, presence mask, alpha scaling) into Evo 2's forward
    pass without needing Vortex to expose an ``inputs_embeds`` argument.
    """

    def __init__(self, embed_override: torch.Tensor):
        """
        Parameters
        ----------
        embed_override : torch.Tensor
            Shape ``[B, L, d_model]``.  Will be returned instead of the
            normal embedding lookup output.
        """
        self._override = embed_override
        self._handle = None

    def register(self, embedding_layer: nn.Embedding):
        self._handle = embedding_layer.register_forward_hook(self._hook_fn)
        return self

    def _hook_fn(self, module, input, output):
        # Remove the hook immediately (one-shot)
        self._handle.remove()
        return self._override.to(output.device, output.dtype)


# ---------------------------------------------------------------------------
# Core customised forward pass factory
# ---------------------------------------------------------------------------

def customize_forward_pass(
    backbone,
    residual,
    presence,
    input_ids,
    grad_idx,
):
    """
    Build a differentiable forward-pass function for Evo 2 / StripedHyena2,
    analogous to the transformer version in JCBScope_utils.py.

    Parameters
    ----------
    backbone : torch.nn.Module
        The Vortex backbone returned by ``get_backbone(evo2_wrapper)``.
    residual : torch.Tensor
        Shape ``[d_model]``.  The perturbation vector added at ``grad_idx``.
    presence : torch.Tensor
        Shape ``[1, L, 1]`` or ``[L]`` boolean / float mask applied to all
        token embeddings (1 = keep, 0 = ablate).
    input_ids : torch.Tensor
        Shape ``[1, L]`` int32 tensor of token ids.
    grad_idx : int
        The sequence position at which ``residual`` is injected.

    Returns
    -------
    forward_pass : callable
        A closure that executes one forward pass; see its docstring for the
        full signature.

    Notes
    -----
    * No ``attention_mask`` is needed – StripedHyena2 processes all positions
      unconditionally (no softmax attention mask).
    * The backbone forward returns ``(logits, _)``; ``logits`` has shape
      ``[B, L, vocab_size]``.  For single-batch use (``B=1``) this matches
      the transformer version's ``logits[0]`` of shape ``[L, vocab_size]``.
    * Tied-embedding logit correction (``tie_input_output_embed=True``) works
      identically: we replace the diagonal logit at each position with the dot
      product of the hidden state and the *next* token's embedding vector.
    """
    embedding_layer = get_embedding_layer(backbone)
    # unembed_w is the raw [V, d] weight matrix (handles Lambda wrapper & tied weights)
    unembed_w = get_unembed_weight(backbone)

    embed_device = embedding_layer.weight.device
    residual = residual.to(embed_device)
    presence = presence.to(embed_device)

    # Pre-compute base embeddings once (no grad, saved across calls)
    with torch.no_grad():
        input_ids_dev = input_ids.to(embed_device)
        # Use F.embedding to bypass any Accelerate device hooks
        base_embeds = F.embedding(input_ids_dev, embedding_layer.weight)
        # base_embeds: [1, L, d]

    # ------------------------------------------------------------------
    # Helper: build full embedding tensor with residual injected
    # ------------------------------------------------------------------
    def _build_inputs(
        residual_override=None,
        residual_batch=None,
    ):
        """
        Build the ``[B, L, d]`` input embedding tensor.

        Parameters
        ----------
        residual_override : torch.Tensor or None
            ``[d]`` – used for single-sample override of the stored residual.
        residual_batch : torch.Tensor or None
            ``[B, d]`` – used for batched finite-difference JVP.

        Returns
        -------
        torch.Tensor
            ``[B, L, d]`` embedding tensor.
        """
        if residual_batch is not None:
            B = residual_batch.shape[0]
            embeds = base_embeds.expand(B, -1, -1).clone()
            embeds[:, grad_idx, :] += residual_batch
            return embeds

        r = residual if residual_override is None else residual_override
        embeds = base_embeds.clone()
        embeds[0, grad_idx, :] += r
        return embeds

    # ------------------------------------------------------------------
    # Helper: compute logits from hidden states
    # ------------------------------------------------------------------
    def _compute_logits_tied(hidden, input_embeds):
        """
        Tied-embedding logit correction.

        Replaces each diagonal logit ``logits[t, next_token_id]`` with the dot
        product ``hidden[t] · embed(next_token)``, so that gradients flow
        through the input embedding at the *prediction* position as well.

        Parameters
        ----------
        hidden : torch.Tensor
            Shape ``[L, d]``.
        input_embeds : torch.Tensor
            Shape ``[1, L, d]`` (single batch).

        Returns
        -------
        torch.Tensor
            Shape ``[L, vocab_size]``.
        """
        vocab_embed = unembed_w                                    # [V, d]
        L = input_embeds.size(1)
        # Standard logit matrix
        logits = hidden.to(vocab_embed.device) @ vocab_embed.T    # [L, V]
        # Replace next-token logits with dot product through input embed
        hidden_prev = hidden[:-1]                                  # [L-1, d]
        embeds_next = input_embeds[0, 1:].to(hidden.device)       # [L-1, d]
        target_logits = (hidden_prev * embeds_next).sum(dim=-1)   # [L-1]
        targets = input_ids[0, 1:].to(logits.device)
        logits[torch.arange(L - 1, device=logits.device), targets.long()] = \
            target_logits.to(logits.device)
        return logits

    # ------------------------------------------------------------------
    # Main forward pass closure
    # ------------------------------------------------------------------
    def forward_pass(
        loss_position="all",
        hidden_norm_as_loss=False,
        unnormalized_logits=False,
        projection_probe=None,
        tie_input_output_embed=False,
        return_input_embeds=False,
        alpha=1.0,
        residual_override=None,
        residual_batch=None,
        return_hidden=False,
    ):
        """
        Execute one forward pass through Evo 2 with the injected residual /
        presence mask, mirroring the API of the transformer JCBScope version.

        Parameters
        ----------
        loss_position : int | torch.Tensor | "all"
            Which token position to compute the loss at.
            "all"  → cross-entropy averaged over the full sequence.
            int    → cross-entropy at that single next-token prediction.
        hidden_norm_as_loss : bool
            If True, use the normalised hidden state's dot-product as the loss
            (temperature scope / Hutchinson diagonal variant).
        unnormalized_logits : bool
            If True, use raw (un-softmax'd) log-likelihood instead of CE.
        projection_probe : torch.Tensor or None
            ``[d]`` unit vector for Hutchinson random-projection trace
            estimation.  If given, the loss is the dot product of the hidden
            state at ``loss_position`` with this vector.
        tie_input_output_embed : bool
            If True, apply the tied-embedding logit correction (see
            ``_compute_logits_tied``).
        return_input_embeds : bool
            If True, also return the input embeddings used in this call.
        alpha : float
            Scalar applied to all input embeddings before the forward pass.
        residual_override : torch.Tensor or None
            Single-sample residual override (replaces ``residual``).
        residual_batch : torch.Tensor or None
            Batched residuals ``[B, d]`` for finite-difference JVP.
        return_hidden : bool
            If True and ``loss_position != "all"``, return the hidden state at
            ``loss_position`` instead of the loss/logit tuple.

        Returns
        -------
        (loss, logits) : default
        (loss, logits, loss_full) : when loss_position == "all"
        (loss, logits, input_embeds) : when return_input_embeds is True
        hidden : torch.Tensor : when return_hidden is True and loss_position != "all"
        """
        # ---- build embedding tensor ----------------------------------------
        input_embeds = _build_inputs(
            residual_override=residual_override,
            residual_batch=residual_batch,
        )
        batched = residual_batch is not None

        # ---- rehabilitate inference tensors if autograd is active -----------
        # Vortex Hyena filter layers cache poles/residues that may have been
        # created under a previous torch.no_grad() context, marking them as
        # inference tensors.  When autograd is subsequently enabled (e.g. for
        # Jacobian computation), PyTorch raises:
        #   "Inference tensors cannot be saved for backward."
        # Cloning all such parameters/buffers before the forward pass fixes this.
        # We only pay this cost when torch.is_grad_enabled() is True.
        if torch.is_grad_enabled():
            rehabilitate_inference_tensors(backbone)

        # ---- apply presence mask and alpha -----------------------------------
        if batched:
            # presence: [L] → [1, L, 1] for broadcast against [B, L, d]
            p = presence.view(1, -1, 1)
            input_embeds = input_embeds * p * alpha
        else:
            # presence: [L] → [L, 1] for broadcast against [L, d]
            p = presence.view(-1, 1)
            input_embeds[0] = input_embeds[0] * p * alpha

        # ---- inject embeddings via one-shot hook ----------------------------
        #
        # Vortex's backbone forward takes ``input_ids`` and runs the embedding
        # lookup internally.  We intercept the output of ``embedding_layer``
        # and replace it with our custom ``input_embeds`` tensor.
        #
        # We also hook ``backbone.norm`` (the final LayerNorm before unembed)
        # to capture hidden states cheaply without re-running the model or
        # relying on non-existent ``return_logits`` kwargs.
        #
        # Both hooks are one-shot and remove themselves after firing.

        _hidden_capture = {}

        def _norm_capture_hook(module, input, output):
            # output of backbone.norm: [B, L, d]
            _hidden_capture["hidden"] = output
            # do NOT remove yet – let unembed consume the same output

        # Register the norm capture hook (persistent until we remove it)
        norm_module = backbone.norm  # always present in Vortex StripedHyena
        norm_handle = norm_module.register_forward_hook(_norm_capture_hook)

        # Register the embed override hook (one-shot, self-removing)
        _EmbedOverrideHook(input_embeds).register(embedding_layer)

        # Provide a dummy input_ids of the right shape (values don't matter
        # because the embed hook replaces the lookup output before it's used).
        dummy_ids = input_ids.to(embed_device)
        if batched:
            B = input_embeds.shape[0]
            dummy_ids = dummy_ids.expand(B, -1)

        # ---- run backbone (standard forward, returns logits) ----------------
        # StripedHyena.forward returns (logits, inference_params_dict_out)
        # logits shape: [B, L, V]
        fwd_out = backbone(dummy_ids)
        norm_handle.remove()   # clean up the norm capture hook

        if isinstance(fwd_out, (tuple, list)):
            logits_full = fwd_out[0]
        else:
            logits_full = fwd_out

        # ---- hidden states --------------------------------------------------
        hidden = _hidden_capture.get("hidden", None)
        # hidden: [B, L, d]  (None if norm hook didn't fire, e.g. diff arch)

        # ---- compute logits -------------------------------------------------
        # Use backbone logits directly (already computed via unembed_w inside
        # Vortex), falling back to manual matmul only when needed.
        if not batched:
            if tie_input_output_embed and hidden is not None:
                logits = _compute_logits_tied(hidden[0], input_embeds)
            else:
                logits = logits_full[0]     # [L, V]
        else:
            logits = logits_full            # [B, L, V]

        # ---- targets (must be int64 for CrossEntropyLoss) -------------------
        targets = input_ids[0, 1:].to(device=logits.device, dtype=torch.long)

        # ---- optional early return of hidden --------------------------------
        if return_hidden and loss_position != "all":
            if hidden is None:
                raise RuntimeError(
                    "return_hidden=True requires hidden states, but backbone.norm "
                    "hook did not capture them. Check that backbone.norm exists."
                )
            if not torch.is_tensor(loss_position):
                lp = torch.tensor(loss_position, device=hidden.device)
            else:
                lp = loss_position.to(hidden.device)
            if not batched:
                return hidden[0, lp, :]
            else:
                return hidden[:, lp, :]

        # ---- loss_position == "all" ----------------------------------------
        if loss_position == "all":
            if unnormalized_logits:
                target_logits = logits[torch.arange(len(targets)), targets]
                loss_full = -target_logits
                loss = loss_full.mean()
            else:
                loss = nn.CrossEntropyLoss()(logits[:-1], targets)
                loss_full = nn.CrossEntropyLoss(reduction="none")(
                    logits[:-1], targets
                ).detach()
            if return_input_embeds:
                return loss, logits, input_embeds.detach()
            return loss, logits, loss_full

        # ---- scalar / vector loss_position ----------------------------------
        if not torch.is_tensor(loss_position):
            loss_position = torch.tensor(loss_position)

        # Hutchinson random projection
        if projection_probe is not None:
            if hidden is None:
                raise RuntimeError(
                    "projection_probe requires hidden states (backbone.norm hook)."
                )
            projection_probe = projection_probe / projection_probe.norm(
                dim=-1, keepdim=True
            )
            loss_position = loss_position.to(hidden.device)
            loss = (
                hidden[0, loss_position, :] * projection_probe.to(hidden.device)
            ).sum(dim=-1)
            if return_input_embeds:
                return loss, logits, input_embeds.detach()
            return loss, logits

        # Temperature scope (hidden-norm loss)
        if hidden_norm_as_loss:
            if hidden is None:
                raise RuntimeError(
                    "hidden_norm_as_loss requires hidden states (backbone.norm hook)."
                )
            hidden_act = hidden[:, loss_position, :].detach()
            hidden_act = hidden_act / hidden_act.norm(dim=-1, keepdim=True)
            loss_position = loss_position.to(hidden.device)
            loss = (hidden[0, loss_position, :] * hidden_act).sum(dim=-1)
            if return_input_embeds:
                return loss, logits, input_embeds.detach()
            return loss, logits

        # Standard CE / unnormalized at chosen position
        target_chosen = targets[loss_position].unsqueeze(0).to(dtype=torch.long)
        loss_position = loss_position.to(logits.device)
        target_chosen = target_chosen.to(logits.device)

        if unnormalized_logits:
            loss = -logits[loss_position, target_chosen]
        else:
            assert isinstance(loss_position, int) or (
                torch.is_tensor(loss_position) and loss_position.dim() == 0
            ), "loss_position must be int, 0-D tensor, or 'all'"
            logits_chosen = logits[loss_position, :].unsqueeze(0)  # [1, V]
            loss = nn.CrossEntropyLoss()(logits_chosen, target_chosen)

        if return_input_embeds:
            return loss, logits, input_embeds.detach()
        return loss, logits

    return forward_pass


# ---------------------------------------------------------------------------
# Convenience: tokenise a DNA string to an input_ids tensor
# ---------------------------------------------------------------------------

def tokenize_sequence(sequence: str, evo2_wrapper, device="cuda"):
    """
    Tokenise a raw DNA string with the Evo 2 CharLevelTokenizer.

    Parameters
    ----------
    sequence : str
        Raw DNA string (e.g. ``'ACGTACGT'``).
    evo2_wrapper : evo2.Evo2
        Loaded Evo 2 model wrapper.
    device : str or torch.device

    Returns
    -------
    input_ids : torch.Tensor
        Shape ``[1, L]``, dtype ``torch.int32``.
    """
    ids = evo2_wrapper.tokenizer.tokenize(sequence)
    # tokenizer may return np.uint8 values; cast to plain int for torch.tensor
    ids = [int(i) for i in ids]
    return torch.tensor(ids, dtype=torch.int32).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal smoke-test.  Requires a GPU and a working evo2 installation.

    Usage
    -----
        python JCBScope_evo2_utils.py
    """
    from evo2 import Evo2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEQUENCE = "ATCGATCGATCGATCGATCG"

    print("Loading Evo 2 7B …")
    evo2_wrapper = Evo2("evo2_7b")
    backbone = get_backbone(evo2_wrapper)
    backbone.eval()

    input_ids = tokenize_sequence(SEQUENCE, evo2_wrapper, device=DEVICE)
    L = input_ids.shape[1]
    embedding_layer = get_embedding_layer(backbone)
    d_model = embedding_layer.weight.shape[1]

    print(f"Sequence length : {L}")
    print(f"d_model         : {d_model}")

    # Zero residual, no masking
    residual = torch.zeros(d_model).to(DEVICE)
    presence = torch.ones(L, dtype=torch.float32).to(DEVICE)

    forward_pass = customize_forward_pass(
        backbone=backbone,
        residual=residual,
        presence=presence,
        input_ids=input_ids,
        grad_idx=5,
    )

    # Full-sequence CE loss
    loss, logits, loss_full = forward_pass(loss_position="all")
    print(f"Full-sequence CE loss : {loss.item():.4f}")
    print(f"Per-token losses      : {loss_full.cpu().numpy()}")

    # Gradient of loss at position 10 w.r.t. the residual perturbation
    residual_grad = torch.zeros(d_model, requires_grad=True, device=DEVICE)
    fwd = customize_forward_pass(
        backbone=backbone,
        residual=residual_grad,
        presence=presence,
        input_ids=input_ids,
        grad_idx=5,
    )
    loss10, _ = fwd(loss_position=10)
    loss10.backward()
    print(f"Gradient norm at pos 10 : {residual_grad.grad.norm().item():.6f}")