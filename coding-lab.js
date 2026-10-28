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
  },
  {
    "id": "numpy-storage",
    "title": "Share NumPy storage, then make an independent copy",
    "level": "07 / NUMPY & TENSOR OWNERSHIP",
    "objective": "Distinguish a CPU tensor that aliases a writable NumPy array from a cloned tensor with independent storage.",
    "scaffold": "import numpy as np\nimport torch\n\ndef bridge_array(array: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:\n    if not array.flags.writeable:\n        raise ValueError(\"Use a writable input array\")\n    shared = __SLOT_0__\n    independent = __SLOT_1__\n    return shared, independent",
    "slots": [
      {"label": "Create a CPU tensor sharing the NumPy array", "placeholder": "torch....(array)", "answers": ["torch.from_numpy(array)", "torch.as_tensor(array)"], "hint": "For this writable float32 CPU array, from_numpy shares memory rather than copying."},
      {"label": "Make independent tensor storage", "placeholder": "shared....()", "answers": ["shared.clone()"], "hint": "detach breaks autograd history but still shares storage. clone is the copy operation needed here."}
    ],
    "expected": "For [2, 4, 8], setting shared[1]=9 also changes array[1], but independent[1] remains 4. All tensors are on CPU with dtype float32.",
    "explanation": "Ownership and gradient tracking are different questions. from_numpy aliases compatible CPU memory; a dtype/device conversion can allocate a new tensor. This exercise assumes a supported writable NumPy dtype and layout; read-only input is explicitly rejected. Do not write through tensors backed by read-only arrays.",
    "tests": "array = np.array([2., 4., 8.], dtype=np.float32)\nshared, independent = bridge_array(array)\nassert shared.dtype == independent.dtype == torch.float32\nassert shared.device.type == 'cpu'\nassert shared.data_ptr() != independent.data_ptr()\nshared[1] = 9\nassert array[1] == 9 and independent[1].item() == 4\narray[2] = 12\nassert shared[2].item() == 12 and independent[2].item() == 8\nindependent[0] = -3\nassert array[0] == 2\nreadonly = np.array([1.], dtype=np.float32)\nreadonly.flags.writeable = False\ntry:\n    bridge_array(readonly)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Read-only input must be rejected')\nprint('PASS: shared storage, independent clone, dtype, read-only rejection')",
    "lesson": "pytorch"
  },
  {
    "id": "autograd-branch",
    "title": "Inspect gradients through a changing computation graph",
    "level": "08 / LEAF & NON-LEAF GRADIENTS",
    "objective": "Retain an intermediate gradient, choose a forward branch, and verify the chain rule without accidentally accumulating old leaf gradients.",
    "scaffold": "import torch\n\ndef branch_grad(x: torch.Tensor, squared: bool) -> tuple[torch.Tensor, torch.Tensor]:\n    if not x.is_leaf or not x.requires_grad:\n        raise ValueError(\"x must be a leaf requiring gradients\")\n    x.grad = None\n    h = 3 * x\n    __SLOT_0__\n    loss = __SLOT_1__\n    __SLOT_2__\n    return h, loss.detach()",
    "slots": [
      {"label": "Keep the non-leaf h gradient for inspection", "placeholder": "h....()", "answers": ["h.retain_grad()"], "hint": "Autograd computes intermediate derivatives, but non-leaf .grad is not normally populated unless retained."},
      {"label": "Use squared or linear loss according to the branch", "placeholder": "... if squared else ...", "answers": ["h.square().sum() if squared else h.sum()", "(h ** 2).sum() if squared else h.sum()"], "hint": "The dynamic forward graph includes only the selected branch. Sum produces a scalar loss."},
      {"label": "Compute gradients", "placeholder": "loss....()", "answers": ["loss.backward()"], "hint": "The chain rule carries dloss/dh through h=3*x to x. It does not update x."}
    ],
    "expected": "At x=2: h=6. Squared branch: loss=36, h.grad=12, x.grad=36. Linear branch: loss=6, h.grad=1, x.grad=3.",
    "explanation": "For h=3x and L=h^2, dL/dx=(2h)*3=18x. The linear branch gives 3. Each call rebuilds the graph and deliberately clears x.grad. No retain_graph=True is needed. A detached loss with no remaining differentiable path cannot backward; it does not produce a successful all-zero training step.",
    "tests": "x = torch.tensor([2.], requires_grad=True)\nh, loss = branch_grad(x, True)\nassert h.grad is not None and x.grad is not None\nassert h.item() == 6 and loss.item() == 36\nassert h.grad.item() == 12 and x.grad.item() == 36\nh2, loss2 = branch_grad(x, False)\nassert h2.grad is not None\nassert loss2.item() == 6 and h2.grad.item() == 1 and x.grad.item() == 3\nassert not loss2.requires_grad\nnegative = torch.tensor([-1.], requires_grad=True)\nbranch_grad(negative, True)\nassert negative.grad.item() == -18\ntry:\n    branch_grad(torch.tensor([2.]), True)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('A leaf requiring gradients is required')\nprint('PASS: branch-specific chain rule, retained non-leaf gradient, fresh gradients')",
    "lesson": "pytorch"
  },
  {
    "id": "custom-module",
    "title": "Register a custom layer's learnable and fixed state",
    "level": "09 / NN.MODULE & PARAMETERS",
    "objective": "Build a composable feature-wise scale layer with a learned parameter, a non-learned buffer, and a differentiable forward method.",
    "scaffold": "import torch\nfrom torch import nn\n\nclass FeatureScale(nn.Module):\n    def __init__(self, width: int) -> None:\n        super().__init__()\n        if width <= 0:\n            raise ValueError(\"width must be positive\")\n        self.scale = __SLOT_0__\n        __SLOT_1__\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        if x.shape[-1] != self.scale.numel():\n            raise ValueError(\"Input feature width does not match\")\n        return __SLOT_2__",
    "slots": [
      {"label": "Register a trainable feature scale", "placeholder": "nn.Parameter(...)", "answers": ["nn.Parameter(torch.ones(width))"], "hint": "Assigning an nn.Parameter to a Module attribute makes it discoverable by model.parameters()."},
      {"label": "Register a fixed offset buffer", "placeholder": "self.register_buffer(...)", "answers": ["self.register_buffer(\"offset\", torch.zeros(width))", "self.register_buffer('offset', torch.zeros(width))"], "hint": "A persistent buffer moves with the module and is in state_dict, but is not an optimizer parameter."},
      {"label": "Compute the feature-wise affine result", "placeholder": "x * ... + ...", "answers": ["x * self.scale + self.offset", "self.scale * x + self.offset"], "hint": "The [width] vectors broadcast over leading batch dimensions. Implement forward, not __call__."}
    ],
    "expected": "At initialization the layer preserves its input. After summing output [1,2,3], scale.grad=[1,2,3]; SGD at 0.1 changes scale to [0.9,0.8,0.7] while offset stays zero.",
    "explanation": "Module.__call__ dispatches forward and supports hooks; use model(x) rather than bypassing it. Parameters and buffers have different optimization roles but both participate in state serialization and module device/dtype moves. Sequential can compose this custom layer with other modules.",
    "tests": "layer = FeatureScale(3)\nx = torch.tensor([[1., 2., 3.]])\nassert torch.equal(layer(x), x)\nassert list(dict(layer.named_parameters())) == ['scale']\nassert list(dict(layer.named_buffers())) == ['offset']\nassert set(layer.state_dict()) == {'scale', 'offset'}\noptimizer = torch.optim.SGD(layer.parameters(), lr=0.1)\nlayer(x).sum().backward()\nassert torch.equal(layer.scale.grad, torch.tensor([1., 2., 3.]))\noptimizer.step()\nassert torch.allclose(layer.scale, torch.tensor([0.9, 0.8, 0.7]))\nassert torch.equal(layer.offset, torch.zeros(3))\nassert nn.Sequential(layer, nn.ReLU())(x).shape == x.shape\nlayer.to(dtype=torch.float64)\nassert layer.scale.dtype == layer.offset.dtype == torch.float64\ntry:\n    layer(torch.ones(2, 4))\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Mismatched width must fail')\nprint('PASS: registration, buffers, gradients, optimizer, composition, dtype move')",
    "lesson": "pytorch"
  },
  {
    "id": "device-batch",
    "title": "Move a batch without breaking feature and target dtypes",
    "level": "10 / DEVICES & DTYPES",
    "objective": "Move features and class indices onto an explicitly chosen device, preserving the different dtypes required by a classification loss.",
    "scaffold": "import torch\n\ndef prepare_batch(features: torch.Tensor, labels: torch.Tensor,\n                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:\n    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:\n        raise ValueError(\"Expected [batch, features] and [batch] labels\")\n    x = __SLOT_0__\n    y = __SLOT_1__\n    return x, y",
    "slots": [
      {"label": "Move features as float32", "placeholder": "features.to(...)", "answers": ["features.to(device=device, dtype=torch.float32)", "features.to(device, dtype=torch.float32)"], "hint": "Tensor.to returns a tensor; it does not necessarily mutate the original tensor or force a copy."},
      {"label": "Move integer class targets as long", "placeholder": "labels.to(...)", "answers": ["labels.to(device=device, dtype=torch.long)", "labels.to(device, dtype=torch.long)"], "hint": "Index targets for cross-entropy use long, not floating feature dtype. Inputs here already contain integer class IDs."}
    ],
    "expected": "CPU fixture: int16 features become float32; int16 class IDs become long. Shapes and values stay unchanged, and the original tensors keep their dtypes.",
    "explanation": "Move the model and all participating tensors to compatible devices. CUDA availability does not prove a speedup, and a CPU test does not validate CUDA kernels. .to may return the same tensor when no conversion is needed. Casting class IDs is not a substitute for validating their range or integral meaning.",
    "tests": "features = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int16)\nlabels = torch.tensor([0, 2], dtype=torch.int16)\nx, y = prepare_batch(features, labels, torch.device('cpu'))\nassert x.dtype == torch.float32 and y.dtype == torch.long\nassert x.device.type == y.device.type == 'cpu'\nassert torch.equal(x, features.float()) and torch.equal(y, labels.long())\nassert features.dtype == labels.dtype == torch.int16\nloss = torch.nn.functional.cross_entropy(x, y)\nassert torch.isfinite(loss)\ntry:\n    prepare_batch(features, labels[:1], torch.device('cpu'))\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Mismatched batch sizes must fail')\nprint('PASS: CPU transfer, feature/label dtypes, preserved values, shape rejection')",
    "lesson": "cuda"
  },
  {
    "id": "scheduler",
    "title": "Apply the learning rate to the intended optimizer update",
    "level": "11 / LEARNING-RATE SCHEDULING",
    "objective": "Record the rate actually used for each update and advance a StepLR only after optimizer.step in this per-update schedule.",
    "scaffold": "import torch\nfrom torch import nn\n\ndef scheduled_updates(steps: int) -> tuple[list[float], torch.Tensor]:\n    if steps < 0:\n        raise ValueError(\"steps must be nonnegative\")\n    model = nn.Linear(1, 1, bias=False)\n    with torch.no_grad():\n        model.weight.zero_()\n    optimizer = torch.optim.SGD(model.parameters(), lr=0.4)\n    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)\n    rates = []\n    for _ in range(steps):\n        rates.append(__SLOT_0__)\n        optimizer.zero_grad(set_to_none=True)\n        loss = (model(torch.ones(1, 1)) - 1).square().mean()\n        loss.backward()\n        __SLOT_1__\n        __SLOT_2__\n    return rates, model.weight.detach().clone()",
    "slots": [
      {"label": "Read the optimizer's current rate before the update", "placeholder": "optimizer.param_groups[0][...]", "answers": ["optimizer.param_groups[0][\"lr\"]", "optimizer.param_groups[0]['lr']"], "hint": "This records the rate actually used by the following optimizer.step."},
      {"label": "Update model weights first", "placeholder": "optimizer....()", "answers": ["optimizer.step()"], "hint": "Backward only computes gradients; optimizer.step applies the current rate."},
      {"label": "Advance this per-update schedule afterward", "placeholder": "scheduler....()", "answers": ["scheduler.step()"], "hint": "With step_size=2, the next pair of updates uses half the previous pair's rate."}
    ],
    "expected": "Four optimizer updates use [0.4,0.4,0.2,0.2], and the scalar weight ends near 0.9856. Six updates use [0.4,0.4,0.2,0.2,0.1,0.1].",
    "explanation": "The scheduling unit is a deliberate choice. Here one scheduler step follows each successful optimizer update; an epoch-based schedule instead advances at epoch boundaries. Metric-driven schedulers have another contract. Mixed-precision skipped updates require matching the schedule to actual updates, not blindly counting backward calls.",
    "tests": "rates, weight = scheduled_updates(4)\nassert rates == [0.4, 0.4, 0.2, 0.2]\nassert abs(weight.item() - 0.9856) < 1e-6\nsix, _ = scheduled_updates(6)\nassert six == [0.4, 0.4, 0.2, 0.2, 0.1, 0.1]\nempty, initial = scheduled_updates(0)\nassert empty == [] and initial.item() == 0\ntry:\n    scheduled_updates(-1)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Negative step count must fail')\nprint('PASS: rate sequence, update ordering, parameter value, zero/invalid steps')",
    "lesson": "training"
  },
  {
    "id": "accumulation",
    "title": "Accumulate unequal microbatches with the correct weighting",
    "level": "12 / BATCH-WEIGHTED GRADIENTS",
    "objective": "Match one full-batch mean-squared-error SGD update using unequal microbatches and a single optimizer step.",
    "scaffold": "import torch\nfrom torch import nn\n\ndef accumulated_step(model: nn.Module, optimizer: torch.optim.Optimizer,\n                     batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:\n    total_targets = sum(target.numel() for _, target in batches)\n    if total_targets == 0:\n        raise ValueError(\"At least one target element is required\")\n    model.train()\n    __SLOT_0__\n    logged_loss = 0.0\n    for x, target in batches:\n        prediction = model(x)\n        if prediction.shape != target.shape:\n            raise ValueError(\"Prediction and target shapes must match\")\n        squared_error = (prediction - target).square().sum()\n        loss = __SLOT_1__\n        __SLOT_2__\n        logged_loss += loss.detach().item()\n    optimizer.step()\n    return logged_loss",
    "slots": [
      {"label": "Clear gradients once before all microbatches", "placeholder": "optimizer....()", "answers": ["optimizer.zero_grad(set_to_none=True)", "optimizer.zero_grad()"], "hint": "Do not clear gradients inside this accumulation window."},
      {"label": "Normalize each summed error by the full target-element count", "placeholder": "squared_error / ...", "answers": ["squared_error / total_targets"], "hint": "Equal weighting of microbatch means is wrong when their target counts differ."},
      {"label": "Accumulate this microbatch's gradient", "placeholder": "loss....()", "answers": ["loss.backward()"], "hint": "Each microbatch builds a fresh graph. The single optimizer.step comes after the loop."}
    ],
    "expected": "For x=[1,2,3], target=2*x, initial w=0 and lr=0.1, splitting into sizes 1 and 2 must match the full-batch loss 56/3 and updated weight 28/15.",
    "explanation": "This exercise averages squared error across target elements, not microbatches. The analogous language-model denominator is the total supervised-token count. Exact full-batch equivalence here assumes the deterministic Linear toy, no batch-dependent layers, and no per-microbatch clipping or optimizer updates. Dropout and BatchNorm can change the equivalence assumptions.",
    "tests": "model = nn.Linear(1, 1, bias=False)\nreference = nn.Linear(1, 1, bias=False)\nwith torch.no_grad():\n    model.weight.zero_()\n    reference.weight.zero_()\nx = torch.tensor([[1.], [2.], [3.]])\ntarget = 2 * x\noptimizer = torch.optim.SGD(model.parameters(), lr=0.1)\nref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)\nref_optimizer.zero_grad(set_to_none=True)\nref_loss = (reference(x) - target).square().mean()\nref_loss.backward()\nref_optimizer.step()\nlogged = accumulated_step(model, optimizer, [(x[:1], target[:1]), (x[1:], target[1:])])\nassert abs(logged - 56 / 3) < 1e-5\nassert torch.allclose(model.weight, reference.weight, atol=1e-6)\nassert abs(model.weight.item() - 28 / 15) < 1e-6\nfor invalid in [[], [(x, target.reshape(-1))]]:\n    try:\n        accumulated_step(model, optimizer, invalid)\n    except ValueError:\n        pass\n    else:\n        raise AssertionError('Empty supervision and broadcast-shaped targets must fail')\nprint('PASS: unequal microbatches, full-batch parity, single update, invalid targets')",
    "lesson": "training"
  }
];
