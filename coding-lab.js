"use strict";
const CODING_LAB = [
  {
    "id": "heads",
    "title": "Split tokens into attention heads",
    "level": "01 / TENSORS & SHAPES",
    "objective": "Transform [batch, tokens, width] into [batch, heads, tokens, head_width] without scrambling token order.",
    "scaffold": "import torch\n\ndef split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:\n    batch, tokens, width = x.shape\n    if heads <= 0 or width % heads:\n        raise ValueError(\"heads must divide width\")\n    separated = __SLOT_0__\n    return __SLOT_1__",
    "slots": [
      {"label": "Reshape into [B, T, H, D]", "placeholder": "x.reshape(...)", "answers": ["x.reshape(batch, tokens, heads, width // heads)", "x.view(batch, tokens, heads, width // heads)"], "hint": "Keep batch and token dimensions first, then split width into heads and per-head width."},
      {"label": "Move heads before tokens", "placeholder": "separated.transpose(...)", "answers": ["separated.transpose(1, 2)", "separated.permute(0, 2, 1, 3)"], "hint": "Swap axes 1 and 2. A transpose is often a non-contiguous view."}
    ],
    "expected": "For x shaped [2, 4, 12] and heads=3: output shape [2, 3, 4, 4]. Head slices preserve the original token's contiguous width groups.",
    "explanation": "Reshaping alone to [B,H,T,D] would group the underlying elements incorrectly. First split the final width axis, then permute axes. reshape also supports inputs whose current strides prevent view.",
    "tests": "x = torch.arange(96.0).reshape(2, 4, 12)\ny = split_heads(x, 3)\nassert y.shape == (2, 3, 4, 4)\nassert torch.equal(y[0, 1, 2], x[0, 2, 4:8])\nrestored = y.transpose(1, 2).reshape(2, 4, 12)\nassert torch.equal(restored, x)\ntry:\n    split_heads(x, 5)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError(\"Invalid head count must be rejected\")\nprint(\"PASS: head shape, token ordering, reconstruction, invalid heads\")",
    "lesson": "pytorch"
  },
  {
    "id": "softmax",
    "title": "Write numerically stable softmax",
    "level": "02 / NUMERICAL STABILITY",
    "objective": "Normalize the last dimension without overflow, preserving batch dimensions and gradient flow.",
    "scaffold": "import torch\n\ndef stable_probs(logits: torch.Tensor) -> torch.Tensor:\n    shifted = __SLOT_0__\n    weights = __SLOT_1__\n    return weights / __SLOT_2__",
    "slots": [
      {"label": "Subtract the row maximum", "placeholder": "logits - ...", "answers": ["logits - logits.amax(dim=-1, keepdim=True)", "logits - logits.max(dim=-1, keepdim=True).values"], "hint": "A per-row maximum with keepdim=True broadcasts over the original last axis."},
      {"label": "Exponentiate shifted logits", "placeholder": "torch....", "answers": ["torch.exp(shifted)", "shifted.exp()"], "hint": "Subtracting a constant leaves softmax unchanged but prevents exp(large positive number)."},
      {"label": "Sum weights along the last axis", "placeholder": "weights.sum(...)", "answers": ["weights.sum(dim=-1, keepdim=True)", "torch.sum(weights, dim=-1, keepdim=True)"], "hint": "Do not sum across the batch. Preserve the final axis for broadcasting."}
    ],
    "expected": "For [1000, 1001, 1002], probabilities are approximately [0.0900, 0.2447, 0.6652], all finite, with row sum 1.",
    "explanation": "Subtracting a shared row constant cancels in softmax's numerator and denominator. Reducing the last axis independently keeps separate batch items from influencing one another. All-masked rows need a separate policy; this exercise assumes finite, nonempty logits.",
    "tests": "logits = torch.tensor([[1000., 1001., 1002.], [3., 1., -2.]], requires_grad=True)\nprobs = stable_probs(logits)\nassert torch.isfinite(probs).all()\nassert torch.allclose(probs.sum(dim=-1), torch.ones(2))\nassert torch.allclose(probs, torch.softmax(logits, dim=-1), atol=1e-6)\nprobs[:, 0].sum().backward()\nassert logits.grad is not None and torch.isfinite(logits.grad).all()\nprint(\"PASS: large logits, normalization, reference parity, gradients\")",
    "lesson": "transformers"
  },
  {
    "id": "causal",
    "title": "Make future tokens invisible",
    "level": "03 / PYTORCH SDPA",
    "objective": "Create a boolean causal mask with the correct SDPA semantics and explicitly disable functional dropout.",
    "scaffold": "import torch\nfrom torch.nn import functional as F\n\ndef causal_attention(q: torch.Tensor, k: torch.Tensor,\n                     v: torch.Tensor) -> torch.Tensor:\n    tokens = q.size(-2)\n    if k.size(-2) != tokens or v.size(-2) != tokens:\n        raise ValueError(\"This exercise requires equal sequence lengths\")\n    allowed = __SLOT_0__\n    return F.scaled_dot_product_attention(\n        q, k, v, attn_mask=allowed, dropout_p=__SLOT_1__\n    )",
    "slots": [
      {"label": "Construct the allowed-position mask", "placeholder": "torch.ones(...).tril()", "answers": ["torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).tril()", "torch.ones((tokens, tokens), dtype=torch.bool, device=q.device).tril()"], "hint": "A lower triangular boolean matrix includes current and previous keys. In SDPA, True means allowed."},
      {"label": "Disable SDPA dropout for this inference exercise", "placeholder": "A numeric probability", "answers": ["0.0", "0"], "hint": "Functional SDPA applies the provided dropout probability regardless of an enclosing module's eval setting."}
    ],
    "expected": "Output has the same [B,H,T,D] shape as Q. Perturbing K and V at the final position must not change earlier outputs.",
    "explanation": "Mask scores before normalization. This exercise is square attention with no padding or KV cache; cached unequal-length attention needs explicit offset handling. We do not pass is_causal together with this explicit mask.",
    "tests": "torch.manual_seed(7)\nq, k, v = [torch.randn(2, 2, 4, 8) for _ in range(3)]\nout = causal_attention(q, k, v)\nassert out.shape == q.shape\nchanged_k, changed_v = k.clone(), v.clone()\nchanged_k[..., -1, :] += 100\nchanged_v[..., -1, :] -= 100\nchanged = causal_attention(q, changed_k, changed_v)\nassert torch.allclose(out[..., :-1, :], changed[..., :-1, :], atol=1e-5)\nmask = torch.ones(4, 4, dtype=torch.bool).tril()\nscores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)\nreference = scores.masked_fill(~mask, float('-inf')).softmax(-1) @ v\nassert torch.allclose(out, reference, atol=1e-5)\nprint(\"PASS: output shape, future-token isolation, explicit attention parity\")",
    "lesson": "pytorch"
  },
  {
    "id": "loss",
    "title": "Align next-token targets and mask padding",
    "level": "04 / CAUSAL LANGUAGE-MODEL LOSS",
    "objective": "Write a custom next-token loss that shifts once, ignores -100 labels, and rejects an all-ignored batch.",
    "scaffold": "import torch\nfrom torch.nn import functional as F\n\ndef shifted_lm_loss(logits: torch.Tensor,\n                    labels: torch.Tensor) -> torch.Tensor:\n    shifted_logits = __SLOT_0__\n    shifted_labels = __SLOT_1__\n    if not (shifted_labels != -100).any():\n        raise ValueError(\"At least one supervised target is required\")\n    return F.cross_entropy(\n        shifted_logits.reshape(-1, shifted_logits.size(-1)),\n        shifted_labels.reshape(-1), ignore_index=__SLOT_2__\n    )",
    "slots": [
      {"label": "Keep predictions before the final position", "placeholder": "logits[:, ...]", "answers": ["logits[:, :-1, :]", "logits[:, :-1, :].contiguous()"], "hint": "The prediction at position t supervises the token at position t+1."},
      {"label": "Keep targets after the first position", "placeholder": "labels[:, ...]", "answers": ["labels[:, 1:]", "labels[:, 1:].contiguous()"], "hint": "Pair logits at 0..T-2 with labels at 1..T-1."},
      {"label": "Specify the ignored label", "placeholder": "The mask label used above", "answers": ["-100"], "hint": "Cross entropy must use the same ignore index as your label mask."}
    ],
    "expected": "With zero logits over a 3-token vocabulary, loss is log(3) ≈ 1.0986 over valid targets. Masked targets and unused final-position logits have zero gradient.",
    "explanation": "This is a custom loss, so it performs its own shift. Many Hugging Face causal-LM classes already shift when labels are supplied—do not apply both. The loss is normalized by valid targets, not all padded positions.",
    "tests": "import math\nlogits = torch.zeros(1, 4, 3, requires_grad=True)\nlabels = torch.tensor([[0, 1, -100, 2]])\nloss = shifted_lm_loss(logits, labels)\nassert abs(loss.item() - math.log(3)) < 1e-6\nloss.backward()\nassert torch.equal(logits.grad[:, 1, :], torch.zeros(1, 3))\nassert torch.equal(logits.grad[:, -1, :], torch.zeros(1, 3))\nassert logits.grad[:, 0, :].abs().sum() > 0\ntry:\n    shifted_lm_loss(logits.detach(), torch.full_like(labels, -100))\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError(\"All-ignored labels must be rejected\")\nprint(\"PASS: shifted loss, ignored gradients, last-position gradient, empty supervision\")",
    "lesson": "sft"
  },
  {
    "id": "train",
    "title": "Wire a clean optimizer update",
    "level": "05 / AUTOGRAD & OPTIMIZATION",
    "objective": "Clear stale gradients, backpropagate, and update parameters exactly once per training step.",
    "scaffold": "import torch\nfrom torch import nn\n\ndef train_once(model: nn.Module, optimizer: torch.optim.Optimizer,\n               x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:\n    model.train()\n    __SLOT_0__\n    loss = (model(x) - target).square().mean()\n    __SLOT_1__\n    __SLOT_2__\n    return loss.detach()",
    "slots": [
      {"label": "Clear previous gradients", "placeholder": "optimizer....", "answers": ["optimizer.zero_grad(set_to_none=True)", "optimizer.zero_grad()"], "hint": "backward accumulates. This exercise is one update per batch, not intentional gradient accumulation."},
      {"label": "Compute gradients", "placeholder": "loss....", "answers": ["loss.backward()"], "hint": "Backward traverses the autograd graph, filling parameter gradients."},
      {"label": "Apply the optimizer update", "placeholder": "optimizer....", "answers": ["optimizer.step()"], "hint": "Computing gradients alone does not update weights."}
    ],
    "expected": "For y = w*x, initial w=0, x=1, target=2 and SGD lr=0.1: first w=0.4; after a second clean update w=0.72.",
    "explanation": "The first loss is (0−2)^2=4, giving gradient −4 and update +0.4. The next gradient is −3.2, not −7.2: stale gradients must be cleared. A detached returned loss avoids retaining a training graph in logging history.",
    "tests": "model = nn.Linear(1, 1, bias=False)\nwith torch.no_grad():\n    model.weight.zero_()\noptimizer = torch.optim.SGD(model.parameters(), lr=0.1)\nx, target = torch.tensor([[1.]]), torch.tensor([[2.]])\nfirst_loss = train_once(model, optimizer, x, target)\nassert abs(first_loss.item() - 4.0) < 1e-6\nassert abs(model.weight.item() - 0.4) < 1e-6\nassert not first_loss.requires_grad\ntrain_once(model, optimizer, x, target)\nassert abs(model.weight.item() - 0.72) < 1e-6\nprint(\"PASS: backward, parameter update, gradient clearing, detached logging loss\")",
    "lesson": "training"
  },
  {
    "id": "inference",
    "title": "Separate eval mode from autograd",
    "level": "06 / INFERENCE CORRECTNESS",
    "objective": "Disable training-only module behavior and gradient recording independently.",
    "scaffold": "import torch\nfrom torch import nn\n\ndef predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:\n    __SLOT_0__\n    with __SLOT_1__:\n        return torch.softmax(model(x), dim=-1)",
    "slots": [
      {"label": "Switch module behavior", "placeholder": "model....", "answers": ["model.eval()", "model.train(False)"], "hint": "Dropout changes behavior with the module's training flag, not just with gradient tracking."},
      {"label": "Enter a no-gradient context", "placeholder": "torch....()", "answers": ["torch.inference_mode()", "torch.no_grad()"], "hint": "eval alone does not disable autograd. This prediction is not being reused for training."}
    ],
    "expected": "Repeated predictions are identical with dropout disabled, probabilities sum to one, and returned tensors do not require gradients.",
    "explanation": "Module mode and autograd mode are separate controls. inference_mode removes extra bookkeeping and imposes additional restrictions; no_grad is also valid for this exercise. Restore train mode before training again. SDPA functional dropout is a separate explicit argument.",
    "tests": "torch.manual_seed(11)\nmodel = nn.Sequential(nn.Linear(4, 3), nn.Dropout(p=0.9))\nmodel.train()\nx = torch.randn(2, 4, requires_grad=True)\na, b = predict(model, x), predict(model, x)\nassert not model.training\nassert not a.requires_grad\nassert torch.equal(a, b)\nassert torch.allclose(a.sum(-1), torch.ones(2))\nassert x.grad is None\nprint(\"PASS: eval behavior, no autograd, deterministic dropout, normalized output\")",
    "lesson": "pytorch"
  }
];
