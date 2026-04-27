"""Phase 2 nn — Embedding, LayerNorm, MultiHeadAttention, TransformerBlock, AdamW."""

import numpy as np

from numpy_grad import Tensor
from numpy_grad.nn import (
    AdamW,
    Embedding,
    LayerNorm,
    Linear,
    MultiHeadAttention,
    TransformerBlock,
    cross_entropy,
)


def test_embedding_module():
    np.random.seed(0)
    emb = Embedding(10, 4)
    out = emb(np.array([1, 3, 1]))
    assert out.shape == (3, 4)
    out.sum().backward()
    assert emb.weight.grad.shape == (10, 4)
    # index 1 was used twice
    assert np.allclose(emb.weight.grad[1], np.full(4, 2.0))


def test_layernorm_zero_mean_unit_var():
    np.random.seed(0)
    ln = LayerNorm(8)
    x = Tensor(np.random.randn(5, 8))
    y = ln(x)
    # ln scales by gamma=1, beta=0 → output should have ~0 mean, ~1 var per row
    assert np.allclose(y.data.mean(axis=-1), 0, atol=1e-6)
    assert np.allclose(y.data.var(axis=-1), 1, atol=1e-3)


def test_mha_shape():
    np.random.seed(0)
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    x = Tensor(np.random.randn(2, 5, 16))
    y = mha(x)
    assert y.shape == (2, 5, 16)
    y.sum().backward()
    # all 4 projection weight matrices got grad
    for layer in [mha.Wq, mha.Wk, mha.Wv, mha.Wo]:
        assert layer.W.grad is not None


def test_mha_causal_mask():
    """Attention output at position t must not depend on positions > t."""
    np.random.seed(0)
    mha = MultiHeadAttention(d_model=8, n_heads=2)
    x = Tensor(np.random.randn(1, 4, 8))
    y_full = mha(x).data.copy()

    # perturb position 3 only — y[:, 0..2] must NOT change (causal)
    x.data[0, 3] += 1.0
    y_perturbed = mha(x).data
    assert np.allclose(y_full[:, :3], y_perturbed[:, :3], atol=1e-8), \
        "causal mask violated — past positions changed when future was perturbed"


def test_swiglu():
    """SwiGLU should pass gradient through all 3 weight matrices."""
    from numpy_grad.nn import SwiGLU
    np.random.seed(0)
    swiglu = SwiGLU(d_model=8, d_ff=24)
    x = Tensor(np.random.randn(3, 8))
    y = swiglu(x)
    assert y.shape == (3, 8)
    y.sum().backward()
    for layer in [swiglu.w1, swiglu.gate, swiglu.w2]:
        assert layer.W.grad is not None and layer.W.grad.any()


def test_clip_grad_norm():
    from numpy_grad.nn import clip_grad_norm_
    p1 = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    p1.grad = np.array([3.0, 4.0])  # norm = 5
    p2 = Tensor(np.array([0.0]), requires_grad=True)
    p2.grad = np.array([12.0])      # adds 144 to sum_sq
    # total norm = sqrt(25 + 144) = 13
    n = clip_grad_norm_([p1, p2], max_norm=1.3)
    assert abs(n - 13.0) < 1e-6
    # after clip: scale 0.1, norms divide
    assert abs(np.sqrt((p1.grad ** 2).sum() + (p2.grad ** 2).sum()) - 1.3) < 1e-5


def test_transformer_block_trainable():
    """One block + tiny optimizer should reduce loss on a synthetic target."""
    np.random.seed(0)
    block = TransformerBlock(d_model=16, n_heads=4)
    head = Linear(16, 5)
    x = Tensor(np.random.randn(2, 4, 16))
    targets = np.array([[0, 1, 2, 3], [4, 0, 1, 2]])

    params = block.parameters() + head.parameters()
    opt = AdamW(params, lr=1e-2)

    losses = []
    for _ in range(50):
        opt.zero_grad()
        h = block(x)
        logits = head(h)
        loss = cross_entropy(logits, targets)
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert losses[-1] < losses[0] * 0.5, \
        f"loss did not drop ({losses[0]:.3f} -> {losses[-1]:.3f})"
