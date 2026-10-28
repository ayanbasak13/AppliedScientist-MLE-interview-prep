"use strict";

const CHAPTER_DIAGRAMS = (() => {
  const takeaways = {
    transformers: [
      "Attention mixes tokens; the MLP transforms features. Residual paths preserve the incoming state.",
      "A tokenizer splits text into pieces and assigns IDs. IDs are addresses, not numerical meanings.",
      "Each token ID selects one learned row from an embedding table.",
      "Position information distinguishes identical tokens at different locations.",
      "Queries and keys score matches; values provide the content that attention mixes.",
      "Scale dot products, then exclude forbidden positions before softmax.",
      "Softmax creates one distribution over allowed keys for each query.",
      "Heads use different learned projections; concatenate their outputs and project back to model width.",
      "A residual adds a learned update to the original vector without changing its shape.",
      "Normalization acts on each token's features. LayerNorm centers; RMSNorm does not.",
      "The MLP applies shared nonlinear feature transformations separately at each position.",
      "Cross-attention uses decoder queries but encoder keys and values; sequence lengths may differ.",
      "Each block refines contextual token vectors while preserving the residual width.",
      "The vocabulary head converts each final hidden vector into one logit per possible token.",
      "Vocabulary softmax predicts next-token probabilities, not guaranteed factual correctness.",
      "Generation appends one chosen token, then predicts again using the updated prefix.",
      "Teacher forcing predicts shifted targets in parallel, with future positions still masked.",
      "Track named axes through every projection; equal numeric dimensions can represent different things.",
      "Longer context increases attention work and cache memory; position schemes do not remove those costs.",
      "Compute a tiny attention example before trusting a large tensor implementation.",
      "Defend architecture choices using a baseline, controlled changes, and measured outcomes."
    ],
    training: [
      "The training target is the next token, not a copy of the input token at the same position.",
      "Follow hidden states through the decoder, then compare predictions with shifted targets.",
      "The teaching update changes only the output head; full pretraining updates the backbone too.",
      "Subtract the largest logit before exponentiation to compute stable probabilities.",
      "Cross-entropy penalizes low probability on the correct token: loss = -log(p_correct).",
      "Perplexity is exp(mean token loss): lower is better only under comparable evaluation conditions.",
      "Gradients say which direction increases loss; gradient descent moves in the opposite direction.",
      "Zero gradients, run forward, compute valid-token loss, backpropagate, then update parameters.",
      "Compare held-out loss with consistent token weighting, splits, and preprocessing.",
      "Filtering, deduplication, and clean splits can matter more than another training run.",
      "Decoding changes the chosen output, not the underlying training objective.",
      "Diagnose data, numerics, optimization, and evaluation before adding compute.",
      "Use ablations and explicit uncertainty to distinguish a real improvement from noise."
    ],
    "gpt-training": [
      "Data, tokenizer, training, post-training, evaluation, and serving form one versioned lifecycle.",
      "Define success and freeze evaluation before tuning the model.",
      "Filter and deduplicate the corpus; preserve provenance and split boundaries.",
      "Tokenize and pack sequences while preserving the intended boundaries and loss masks.",
      "Choose decoder dimensions and initialization to fit the task and memory budget.",
      "Shift tokens once: each input position predicts the following token.",
      "Average over valid target tokens, not padded positions or unequal batch averages.",
      "AdamW uses gradients and optimizer state; weights alone do not describe the training update.",
      "Accumulate token-weighted contributions rather than giving unequal microbatches equal influence.",
      "Mixed precision, sharding, and recomputation address different memory and throughput bottlenecks.",
      "Balance parameters, training tokens, and compute rather than maximizing one independently.",
      "Track held-out quality and failure slices throughout the run.",
      "A restartable checkpoint includes optimizer, data position, RNG, and other relevant training state.",
      "Instruction tuning teaches desired responses using demonstrations and explicit target masking.",
      "Preference methods optimize different objectives; their data and reference requirements differ.",
      "Release only after quality, risk, latency, and cost gates pass.",
      "Draw the full training plan and defend its assumptions with measurable checks."
    ],
    numpy: [
      "Learn arrays and axes first, then numerical methods, then memory-aware optimization.",
      "An ndarray is a typed buffer plus shape and byte strides; a view can share that buffer.",
      "Reshape changes grouping; transpose changes axes; stacking adds an axis.",
      "Basic slices usually share storage; advanced indexing reads a copy. Pairing is not an outer rectangle.",
      "Broadcast from the right. Keeping a reduced axis makes its intended alignment explicit.",
      "Fit scaling and imputation on training data only, then reuse those statistics on held-out data.",
      "Max-shifting stabilizes softmax, but invalid inputs still need explicit handling.",
      "Use a local random generator; reproducibility also depends on versions, draw order, and parallelism.",
      "Use solve or least squares for the actual problem; an explicit inverse is usually unnecessary.",
      "Name batch, token, and feature axes before contracting them.",
      "Regression fits coefficients, clustering updates centers, and PCA learns directions from centered training data.",
      "Vectorized syntax can allocate enormous intermediates; count bytes and benchmark the whole operation.",
      "NumPy CPU arrays do not provide GPU execution or an autograd graph.",
      "Predict shapes, verify small values and edge cases, then discuss complexity and alternatives."
    ],
    pytorch: [
      "PyTorch tensors, Hugging Face model interfaces, and transformer architectures are different layers.",
      "Practice small working programs, not just remembered API names.",
      "Views share storage; clones do not. Shape alone does not describe layout.",
      "Autograd adds contributions from every active path; detaching cuts a gradient path.",
      "Register parameters, buffers, and child modules so optimizers and state dictionaries can find them.",
      "Keep data, parameters, labels, and intended dtypes on compatible devices.",
      "Weight unequal microbatches correctly and step the scheduler at its documented update boundary.",
      "eval() changes layer behavior; no_grad/inference_mode control gradient recording.",
      "Mask future keys before attention softmax and test that forbidden values cannot affect the output.",
      "Match shifted labels, ignored positions, and returned loss to the model API contract.",
      "Test shape, causality, gradients, devices, and ownership before optimizing.",
      "Explain an observed result and a controlled alternative rather than claiming undocumented behavior."
    ],
    sft: [
      "SFT learns response behavior from demonstrations; it is not a reliable store for changing facts.",
      "Template the conversation consistently and train only on the intended target tokens.",
      "LoRA trains low-rank updates; QLoRA also quantizes the frozen base model.",
      "Compare data quality, adaptation method, and compute under the same held-out evaluation.",
      "Measure grounding and task success separately from a preferred writing style."
    ],
    dpo: [
      "A preference pair compares chosen and rejected answers to the same prompt.",
      "DPO rewards a relative chosen-versus-rejected improvement against a reference policy.",
      "Noisy labels, length bias, and proxy shortcuts can reduce loss without improving usefulness.",
      "Prefer supported answers for the right reason, not just because they sound confident.",
      "Validate factuality independently of preference scores."
    ],
    ppo: [
      "RLHF connects demonstrations, preference-based reward learning, and policy optimization.",
      "PPO clips the policy-ratio incentive, not every possible policy change.",
      "The old rollout policy and the reference policy serve different purposes.",
      "Choose PPO only when its feedback and optimization flexibility justify extra complexity.",
      "Inspect reward, KL, task quality, and failure cases together."
    ],
    inference: [
      "Prefill processes the prompt; decode generates one new token at a time.",
      "KV memory grows with retained tokens, layers, KV heads, batch size, and precision.",
      "Choose an optimization for the measured bottleneck, not its popularity.",
      "FlashAttention reduces attention-memory traffic without replacing the dense attention objective.",
      "Benchmark representative traffic with synchronized timing and quality parity.",
      "A faster isolated kernel does not guarantee a faster end-to-end service."
    ],
    cuda: [
      "Threads form blocks; blocks form a grid. Execution and synchronization have scope.",
      "Registers, shared memory, and device memory have different capacity and access costs.",
      "GPU launches are asynchronous; host launch time is not completed device work.",
      "Streams can overlap independent work, but data dependencies still require coordination.",
      "Multi-GPU speed depends on communication, parallelism, and load balance.",
      "Profile the bottleneck and verify numerical parity before claiming a speedup."
    ],
    retrieval: [
      "Candidate generation must retrieve useful evidence before a reranker can improve its order.",
      "Use cheap broad retrieval first and expensive interactions on a smaller candidate set.",
      "Ranking metrics reward different relevance and ordering properties; state the judged universe.",
      "Indexing, query rewriting, and evidence selection solve different parts of retrieval.",
      "Freshness, permission changes, and label quality can invalidate a seemingly strong score.",
      "Use query slices and controlled ablations to explain the measured benefit."
    ],
    rag: [
      "RAG joins versioned ingestion, authorized retrieval, context assembly, generation, and evaluation.",
      "A citation must support its specific claim; citing a document is not enough.",
      "Measure retrieval, evidence use, answer quality, and operating cost separately.",
      "Document edits require version checks, approval, and recoverable writes.",
      "A useful evaluation fixture defines questions, authorized sources, expected evidence, and grading rules.",
      "Compute scorecard numerators and denominators rather than listing metric names.",
      "Choose the evaluated unit explicitly: query, document, answer, claim, or citation.",
      "Recall measures found evidence; MRR rewards the first relevant hit; nDCG accounts for graded ordering.",
      "Faithfulness, real-world correctness, and citation support can disagree.",
      "Exact match, token overlap, and semantic rubrics answer different grading questions.",
      "Abstention trades answer coverage against the risk among attempted answers.",
      "Oracle evidence and component ablations help locate the source of a failure.",
      "Calibrate model judges against independent human or deterministic anchors.",
      "Use uncertainty, slices, production outcomes, and explicit release gates.",
      "Freeze the comparison, inspect errors, and connect offline evidence to a guarded rollout.",
      "Explain which component changed and what observation would reverse the decision."
    ],
    evaluation: [
      "Start with a decision, target population, and unit of measurement.",
      "Precision conditions on predicted positives; recall conditions on actual positives.",
      "A model judge is a measurement instrument, not unquestionable ground truth.",
      "Percentage-point change and relative percentage change are different calculations.",
      "Separate evidence, uncertainty, and causal attribution."
    ],
    agents: [
      "Use a fixed workflow when possible; autonomy adds decisions and failure modes.",
      "Observe, propose, validate, execute, and verify inside a bounded loop.",
      "Typed tool arguments do not grant authorization or guarantee safe side effects.",
      "Durable task state needs explicit ownership, versioning, and recovery rules.",
      "Orchestration and tool protocols do not replace application policy.",
      "Defend autonomy with measured task outcomes and operational cost."
    ],
    "agent-eval": [
      "Evaluate actions and verified external state, not just the final message.",
      "Test failures across tools, policy, environment, state, and planning.",
      "Retrieved instructions are untrusted content, not authority to change policy.",
      "Retries must distinguish a failed operation from an unknown commit outcome.",
      "A benchmark needs resettable initial state, authorized actions, validators, and trace capture.",
      "Per-attempt success, pass@k, and repeated-run consistency are different quantities.",
      "Use state validators to establish actual completion.",
      "Name the denominator for every success, compliance, recovery, and cost metric.",
      "A successful outcome can still violate policy or exceed the budget.",
      "Success at least once does not mean success on every repeated attempt.",
      "Inject realistic faults and grade the recovery path.",
      "Combine deterministic validators, human judgment, and calibrated model judges.",
      "Measure the whole attempt, including retries, waiting, and human intervention.",
      "Track uncertainty and production regressions by task and environment slice.",
      "State both the task benefit and the residual failure risk."
    ],
    "system-design": [
      "Budget the interview across requirements, architecture, a deep dive, evaluation, and operations.",
      "Separate the ingestion/update path from the permission-aware query path.",
      "Defend architecture choices using authorized examples and measured trade-offs.",
      "Prepare working code and scientific reasoning alongside architecture diagrams.",
      "Rehearse a complete decision with failure handling and a release gate."
    ],
    "anthropic-fde": [
      "Reported interview loops vary; use the question-led mock rather than memorizing one supposed process.",
      "Show evidence of customer-facing engineering and honest ownership.",
      "Discover the workflow and success criteria before recommending a model.",
      "The host owns tool policy; the model proposes actions within that boundary.",
      "Keep durable state and evidence outside a growing chat transcript.",
      "Incremental coding should preserve tested behavior as requirements change.",
      "Practice caches, nested traces, and file identity with explicit edge cases.",
      "A batch-collection timer does not cap total synchronous request latency.",
      "Queueing, execution, and response work all contribute to end-to-end latency.",
      "Hosted APIs impose provider request/token limits, not your own GPU batch scheduler.",
      "Treat authorization, evaluation, rollout, and rollback as deployment requirements.",
      "Use specific evidence, explain uncertainty, and own mistakes under pressure.",
      "Practice timed rounds and follow-ups, then self-score the demonstrated reasoning."
    ],
    "openai-aie": [
      "Application/FDE reports and ML-systems guides describe different preparation needs.",
      "Match coding, evaluation, and infrastructure depth to the actual role.",
      "A correct attention implementation must prevent forbidden-token leakage.",
      "Start clustering and convolution with exact tiny fixtures and explicit edge policies.",
      "A document-review assistant needs evidence, permissions, typed proposals, and approval.",
      "Evaluation needs versioned fixtures, grading, analysis, and release decisions.",
      "Training memory includes more than weights: gradients, optimizer state, activations, and buffers.",
      "Restarting training requires compatible data, optimizer, RNG, and checkpoint state.",
      "Weight memory, KV cache, prefill, decode, and queueing are different serving costs.",
      "Use a bounded build-and-defend mock, not a rumored universal company loop."
    ],
    spark: [
      "Spark helps when distributed execution is justified by data and workload, not merely available.",
      "Lazy transformations build a plan; an action triggers stages and tasks.",
      "Explicit schemas, null semantics, and native expressions preserve predictable behavior.",
      "Joins can multiply rows; windows need deterministic ordering for deterministic results.",
      "Partitioning, skew, spill, caching, and output layout affect different execution costs.",
      "Preserve data versions, event-time contracts, and training/evaluation split hygiene.",
      "Validate schema and multiset/grain parity before comparing runtime."
    ]
  };
  const escape = value => String(value).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const tokens = (values, muted = []) => `<div class="cd-tokens">${values.map((value, i) => `<span class="cd-token cd-color-${i % 6}${muted.includes(i) ? " cd-muted" : ""}">${escape(value)}</span>`).join("")}</div>`;
  const grid = (label, rows) => `<figure class="cd-grid"><figcaption>${escape(label)}</figcaption><table aria-label="${escape(label)}"><tbody>${rows.map(row => `<tr>${row.map(value => `<td class="cd-color-${Math.abs(Math.round(Number(value) || 0)) % 6}">${escape(value)}</td>`).join("")}</tr>`).join("")}</tbody></table></figure>`;
  const card = (label, html) => `<div class="cd-card"><strong>${escape(label)}</strong>${html}</div>`;
  const flow = (...cards) => `<div class="cd-flow">${cards.join('<span class="cd-arrow" aria-hidden="true">&rarr;</span>')}</div>`;
  const text = value => `<p>${escape(value)}</p>`;
  const formula = value => `<div class="cd-formula">${escape(value)}</div>`;
  const bars = items => `<div class="cd-bars">${items.map(([label, percent]) => `<div><span>${escape(label)}</span><div class="cd-track"><i style="width:${percent}%"></i></div><b>${percent}%</b></div>`).join("")}</div>`;
  const compare = (left, right) => `<div class="cd-compare">${left}${right}</div>`;
  const figure = (title, html, caption) => `<figure class="concept-diagram" aria-label="${escape(title)}"><figcaption>${escape(title)}</figcaption>${html}<p class="cd-caption">${escape(caption)}</p></figure>`;
  const paired = () => compare(card("Paired coordinates", tokens(["(0,1)", "(3,4)"]) + grid("a[[0,3],[1,4]]", [[1, 19]])), card("All row/column combinations", grid("a[np.ix_([0,3],[1,4])]", [[1, 4], [16, 19]])));
  const shifted = () => flow(card("Input positions", tokens(["I", "like", "ML"])), card("Predict the next token", tokens(["like", "ML", "<end>"])));
  const evalCounts = () => flow(card("100 held-out tasks", bars([["Answered", 80], ["Abstained", 20]])), card("64 answers verified correct", formula("Coverage = 80/100 = 80%") + formula("Attempted accuracy = 64/80 = 80%") + formula("Overall success = 64/100 = 64%")));
  const weightState = () => `<div class="cd-memory">${[["Weights", "2P bytes"], ["Gradients", "2P"], ["Adam moments", "8P"], ["Master weights", "4P"]].map(([label, bytes]) => card(label, text(bytes))).join("")}</div>${formula("Toy persistent state = 16P bytes; add activations + temporary/communication buffers.")}`;
  const transformerSteps = [
    () => flow(card("Text pieces", tokens(["play", "ing"])), card("Integer addresses", tokens([17, 42]))),
    () => flow(card("Token ID", tokens([2])), grid("Embedding table: select row 2", [[0.1, 0.3], [0.2, 0.4], [0.8, 0.6]]), grid("Selected token vector", [[0.8, 0.6]])),
    () => compare(card("Same token, different positions", tokens(["cat @ 0", "cat @ 3"])), card("Position distinguishes them", text("Absolute embeddings add a position vector; RoPE instead rotates Q/K by position."))),
    () => flow(grid("Token states X", [[1, 0], [0, 1]]), card("Three learned projections", tokens(["W_Q", "W_K", "W_V"])), card("Different roles", text("Q: request; K: match; V: content"))),
    () => flow(grid("Scaled scores", [[1, 2, 3]]), grid("Causal exclusion for first query", [[1, "-inf", "-inf"]]), grid("Probabilities", [[1, 0, 0]])),
    () => flow(card("Key probabilities", bars([["key 0", 25], ["key 1", 75]])), grid("Value vectors", [[4, 0], [0, 8]]), grid("0.25 v0 + 0.75 v1", [[1, 6]])),
    () => flow(card("Head 1 output", tokens([1, 2])), card("Head 2 output", tokens([3, 4])), card("Concatenate, then project", tokens([1, 2, 3, 4]))),
    () => flow(grid("Incoming state x", [[1, 2]]), card("Add update F(x)", grid("Update", [[0.2, -0.5]])), grid("x + F(x)", [[1.2, 1.5]])),
    () => compare(card("LayerNorm: center and scale", formula("[1,3] -> [-1,1]")), card("RMSNorm: scale, do not center", formula("[1,3] / sqrt(5) -> [0.447,1.342]"))) + text("Toy normalization uses unit gain, zero bias, and omits epsilon."),
    () => flow(card("Per-token features", tokens(["d = 4"])), card("Expand + nonlinearity", tokens(["d_ff = 12"])), card("Project back", tokens(["d = 4"]))),
    () => flow(card("Decoder Q", tokens(["T_query = 2"])), card("Encoder K and V", tokens(["T_source = 3"])), grid("Allowed cross-attention score shape (2,3)", [["q0/k0", "q0/k1", "q0/k2"], ["q1/k0", "q1/k1", "q1/k2"]])),
    () => flow(card("Block 1", grid("State [T,d]", [[1, 0], [0, 1]])), card("Block 2 ... L", text("Attention + MLP updates")), card("Final contextual state", text("Same residual shape [T,d]; different features"))),
    () => flow(grid("Hidden vector [d=2]", [[1, 2]]), grid("Head weights [2,3]", [[1, 0, 1], [0, 1, 1]]), grid("Vocabulary logits [3]", [[1, 2, 3]])),
    () => flow(card("Vocabulary logits", tokens([1, 2, 3])), card("Softmax over vocabulary", bars([["token A", 9], ["token B", 24.5], ["token C", 66.5]]))),
    () => flow(card("Prefix", tokens(["I", "like"])), card("Choose next token", tokens(["ML"])), card("New prefix; repeat", tokens(["I", "like", "ML"]))),
    shifted
  ];
  const diagrams = {
    transformers: {
      0: () => figure("Two different mixing operations", compare(card("Across token positions", tokens(["I", "like", "ML"]) + text("Attention: which other tokens matter here?")), card("Within each position", grid("One token's feature vector", [[0.2, 0.7, -0.1]]) + text("MLP: transform the collected features."))), "Residual paths add updates; normalization controls feature scale."),
      17: () => figure("Track meanings, not coincidentally equal dimensions", flow(card("IDs", tokens(["B=2", "T=4"])), card("Hidden state", tokens(["B=2", "T=4", "d=8"])), card("Output logits", tokens(["B=2", "T=4", "V=6"]))), "Changing vocabulary size affects the output head, not the query/key token axes.")
    },
    training: {
      0: () => figure("The target is one position ahead", shifted(), "All valid training positions can be evaluated in parallel under a causal mask."),
      6: () => figure("A gradient points uphill; SGD steps downhill", flow(card("Current weight", formula("w = 0.50")), card("Measured gradient", formula("dL/dw = +0.20")), card("Learning rate 0.1", formula("w_new = 0.50 - 0.1 x 0.20 = 0.48"))), "This is a scalar SGD illustration, not an AdamW update."),
      8: () => figure("Avoid averaging unequal batches equally", compare(card("Batch A: 2 valid tokens", formula("Loss sum = 2")), card("Batch B: 8 valid tokens", formula("Loss sum = 24"))) + formula("Correct mean = (2+24)/(2+8) = 2.6; mean of batch means = (1+3)/2 = 2."), "Weight the loss by the unit you actually intend to average.")
    },
    "gpt-training": {
      2: () => figure("Data hygiene is a branching operation", flow(card("Raw documents", tokens(["A", "A copy", "B", "private"])), card("Filter + deduplicate", tokens(["A", "B"])), card("Versioned splits", tokens(["train", "validation", "test"]))), "Split by the appropriate source/entity/time boundary; do not let near-duplicates leak across splits."),
      5: () => figure("Shift labels, then mask invalid targets", shifted() + tokens(["valid", "valid", "padding: ignore"]), "Packing and boundary rules determine which next-token targets count."),
      9: () => figure("Training state is much larger than weights", weightState(), "Explicit toy dtype assumptions: BF16 weights/gradients, FP32 moments and master weights."),
      12: () => figure("An inference save is not a full restart checkpoint", compare(card("Inference", tokens(["weights", "config", "tokenizer"])), card("Restart training", tokens(["+ optimizer", "+ scheduler", "+ data cursor", "+ RNG / scaler"]))), "Restore compatible versions and verify a recovery run before depending on it.")
    },
    numpy: {
      1: () => figure("One buffer; shape and strides tell you where to look", grid("int64 array a, shape (2,3)", [[0, 1, 2], [3, 4, 5]]) + flow(card("Contiguous buffer", tokens([0, 1, 2, 3, 4, 5])), card("Byte offsets", tokens([0, 8, 16, 24, 32, 40]))) + formula("strides = (24,8); a[1,2] offset = 1x24 + 2x8 = 40 bytes"), "Each int64 takes 8 bytes. Shape is not a copy of the values."),
      2: () => figure("Reshape regroups; transpose swaps axes", flow(grid("a: shape (2,3)", [[0, 1, 2], [3, 4, 5]]), grid("a.reshape(3,2)", [[0, 1], [2, 3], [4, 5]]), grid("a.T: shape (3,2)", [[0, 3], [1, 4], [2, 5]])), "Both outputs have shape (3,2), but their values differ. Reshape may copy; transpose usually changes metadata only."),
      3: () => figure("Pairing and outer selection are different shapes", paired(), "The source is arange(20).reshape(4,5). Advanced indexing reads copies."),
      5: () => figure("Fit once on train, reuse on validation", flow(card("Train feature", tokens([1, 3, 5])), card("Fit statistics", formula("mean = 3; population std = sqrt(8/3)")), card("Held-out value 7", formula("z = (7-3)/sqrt(8/3) = 2.449"))), "Do not refit statistics using the held-out value."),
      6: () => figure("Stable softmax preserves relative differences", flow(card("Large logits", tokens([1000, 1001, 1002])), card("Subtract max", tokens([-2, -1, 0])), card("Normalize exponentials", bars([["p0", 9], ["p1", 24.5], ["p2", 66.5]]))), "Rounded probabilities are shown. Reject invalid inputs instead of treating max-shifting as complete validation."),
      7: () => figure("Local random state does not reset everyone else's generator", compare(card("Generator A: seed 23", tokens(["draw 1", "draw 2"])), card("Fresh generator B: seed 23", tokens(["same draw 1", "same draw 2"]))) + formula("Same tested algorithm/version + same seed + same call sequence -> reproducible local draws."), "Do not assume an identical stream across all versions or algorithms."),
      8: () => figure("Least squares fits the line without an explicit inverse", flow(grid("Design [1,x]", [[1, 0], [1, 1], [1, 2]]), card("lstsq(A, y)", grid("y", [[1], [3], [5]])), card("Intercept and slope", tokens([1, 2]))) + formula("prediction = 1 + 2x"), "Rank-deficient data may fit with multiple coefficient vectors; inspect rank and conditioning."),
      9: () => figure("Only the feature axis is summed", flow(card("Q[b,t,d]", tokens(["B=2", "T=2", "D=3"])), card("K[b,u,d]", tokens(["B=2", "U=4", "D=3"])), card("Scores[b,t,u]", tokens(["B=2", "T=2", "U=4"]))) + formula("einsum('btd,bud->btu', Q, K): keep b,t,u; sum d."), "Batch identity is preserved. Different batches do not attend to each other."),
      10: () => figure("One k-means update, with actual points", flow(card("Points", tokens([0, 2, 8, 10])), card("Initial centers 0 and 10", text("Assign {0,2} to 0; {8,10} to 10.")), card("New centers", tokens([1, 9]))), "Choose empty-cluster/tie policies and recompute labels after the final update."),
      11: () => figure("Broadcasting can create a giant temporary", flow(card("X[N,D]", tokens(["10000", "128"])), card("C[K,D]", tokens(["1000", "128"])), card("X[:,None,:] - C[None,:,:]", tokens(["10000", "1000", "128"]))) + formula("float64 delta alone: 10000 x 1000 x 128 x 8 = 10.24 GB (decimal)."), "The broadcast inputs need not be tiled, but the subtraction output is materialized. Chunk when necessary."),
      12: () => figure("Sharing storage is different from sharing an autograd graph", compare(card("NumPy CPU buffer", tokens(["array", "shared storage may be possible"])), card("PyTorch tensor", tokens(["device contract", "gradient graph contract"]))), "A NumPy conversion does not carry PyTorch's differentiable computation graph.")
    },
    pytorch: {
      0: () => figure("Three layers with different responsibilities", flow(card("Architecture", text("Transformer blocks and tensor operations")), card("Framework", text("PyTorch tensors, autograd, modules")), card("Model library", text("Transformers checkpoint/config/tokenizer APIs"))), "The library name does not specify the architecture or execution backend."),
      5: () => figure("A batch is a set of contracts", flow(card("Inputs", tokens(["float features", "[B,D]", "device A"])), card("Parameters", tokens(["compatible dtype", "[D,C]", "device A"])), card("Class labels", tokens(["integer IDs", "[B]", "device A"]))), "Specific losses differ: class-index cross-entropy uses integer labels, not arbitrary float targets."),
      7: () => figure("Two independent switches", compare(card("model.eval()", text("Changes dropout / batch-normalization behavior.")), card("no_grad() / inference_mode()", text("Controls autograd recording and related inference behavior."))), "eval() alone does not disable gradients; disabling gradients alone does not switch training-mode layers.")
    },
    sft: {
      1: () => figure("Which tokens receive loss?", flow(card("Conversation tokens", tokens(["system", "user", "assistant response"])), card("Target mask", tokens(["ignore", "ignore", "train here"], [0, 1]))), "Response-only SFT is one explicit policy. Templates and boundary tokens must match the intended model contract."),
      2: () => figure("LoRA: a small update beside frozen weights", flow(grid("Frozen W [2,2]", [[1, 0], [0, 1]]), card("B[2,1] x A[1,2]", grid("B", [[1], [2]]) + grid("A", [[0.1, 0.2]])), grid("W + BA; scale=1", [[1.1, 0.2], [0.2, 1.4]])), "This toy omits the usual alpha/r scaling by setting it to 1. QLoRA also quantizes the frozen base.")
    },
    dpo: {
      0: () => figure("One prompt, two candidate continuations", flow(card("Same question", text("Answer using the supplied evidence.")), compare(card("Chosen", tokens(["supported", "qualified"])), card("Rejected", tokens(["unsupported", "confident"])))), "The preference label is a training signal, not automatically a factuality guarantee."),
      1: () => figure("Compare improvement against a reference", compare(card("Reference probabilities", bars([["chosen", 60], ["rejected", 40]])), card("Policy probabilities", bars([["chosen", 80], ["rejected", 20]]))) + formula("margin = log(0.8/0.6) - log(0.2/0.4) = 0.981") + formula("beta = 0.1; loss = -log sigmoid(beta x margin) = 0.645"), "A two-continuation toy. Real DPO uses sequence log-probabilities, not a softmax over a list of candidate answers.")
    },
    ppo: {
      0: () => figure("Two feedback paths in classical RLHF", flow(card("Demonstrations", text("Supervised instruction policy")), card("Preference pairs", text("Fit a reward model")), card("Policy rollouts", text("Score reward; optimize under constraints"))), "The critic estimates expected returns; it is not the reference policy or reward model."),
      1: () => figure("Clipping changes the incentive", compare(card("Positive advantage A=1", formula("r=1.4; clip range [0.8,1.2]") + formula("min(1.4 x 1, 1.2 x 1) = 1.2")), card("Negative advantage A=-1", formula("same r=1.4") + formula("min(-1.4, -1.2) = -1.4"))), "PPO uses the minimum of the two advantage-weighted terms. Clipping is not a hard bound on the resulting policy."),
      2: () => figure("Do not confuse the two comparison policies", compare(card("Old rollout policy", text("Denominator for the PPO probability ratio.")), card("Reference policy", text("Anchor for a separate KL-based behavior constraint."))), "The two policies can initially coincide, but their optimization roles differ.")
    },
    inference: {
      0: () => figure("Prefill and decode do different work", flow(card("Prompt", tokens(["A", "B", "C"])), card("Prefill", text("Process the prompt; create cached K/V.")), card("Decode", tokens(["+ D", "+ E", "+ F"]))), "Decode appends tokens sequentially. Cached keys and values avoid recomputing the full prefix's K/V."),
      4: () => figure("Benchmark the entire request, not just one kernel", flow(card("Queue", text("admission + waiting")), card("Model", text("prefill + decode")), card("Rest of service", text("retrieval + serialization + network"))) + formula("A 2x speedup in 10% of runtime improves total time only from 1.00 to 0.95."), "Hold quality, traffic, tokens, precision, and hardware conditions comparable.")
    },
    cuda: {
      0: () => figure("Indexing a thread's work", flow(card("Grid", text("Many independently scheduled blocks")), card("Block", text("Threads share block-local memory/synchronization")), card("Global thread index", formula("i = blockIdx.x * blockDim.x + threadIdx.x"))), "Bounds-check i against the actual data length; a block barrier does not synchronize the whole grid."),
      2: () => figure("A host timer can stop before device work ends", flow(card("Host launch", tokens(["enqueue kernel"])), card("Device executes later", tokens(["actual GPU work"])), card("Completed timing", text("Use appropriate device events or explicit synchronization."))), "Measure the interval you intend; synchronizing everything can itself remove useful overlap.")
    },
    retrieval: {
      0: () => figure("A reranker cannot resurrect a missing document", flow(card("Judged relevant set", tokens(["A", "B"])), card("Candidates", tokens(["B", "X", "Y"])), card("Reranked candidates", tokens(["B", "Y", "X"]))) + formula("Evidence recall = 1/2 = 50%; A is still absent."), "Better ranking within the candidate set is not improved candidate recall."),
      2: () => figure("The first relevant hit changes reciprocal rank", compare(card("Ranking 1", tokens(["irrelevant", "relevant", "relevant"])) + formula("RR = 1/2"), card("Ranking 2", tokens(["relevant", "irrelevant", "relevant"])) + formula("RR = 1")), "For multiple queries, MRR averages reciprocal ranks under the stated missing-hit policy.")
    },
    rag: {
      1: () => figure("Attach evidence to a claim, not merely to an answer", compare(card("Claim", text("The policy permits 30 days.")), card("Source says 14 days", tokens(["citation exists", "claim unsupported"]))), "A real citation can still fail support. A faithful answer to stale evidence can still be wrong today."),
      7: () => figure("Recall and ordering have different denominators", flow(card("Relevant universe", tokens(["A", "B", "C", "D"])), card("Top-3 retrieved", tokens(["X", "B", "A"])), card("Recall@3", formula("2 relevant found / 4 relevant total = 0.5"))), "Precision@3 is 2/3. nDCG needs graded labels and an ideal ranking; ANN neighbor recall measures something else."),
      10: () => figure("Three percentages; three different questions", evalCounts(), "Selective risk among attempted answers is 16/80 = 20%. Abstention alone can make attempted accuracy look better."),
      11: () => figure("Use oracle evidence to locate the bottleneck", compare(card("Normal pipeline", tokens(["retriever", "generator", "grade"])), card("Controlled intervention", tokens(["gold evidence", "same generator", "same grade"]))), "If quality improves with gold evidence, investigate retrieval/context. If not, inspect generation, task ambiguity, or grading.")
    },
    evaluation: {
      1: () => figure("A confusion matrix makes denominators visible", grid("100 cases: actual rows / predicted columns", [["", "pred +", "pred -"], ["actual +", 30, 10], ["actual -", 5, 55]]) + formula("Precision = 30/(30+5) = 85.7%; recall = 30/(30+10) = 75%."), "Accuracy = (30+55)/100 = 85%. Class balance and error costs still matter."),
      3: () => figure("Percentage points are not relative percent", flow(card("Baseline", bars([["success", 60]])), card("New version", bars([["success", 66]]))) + formula("Absolute change: +6 percentage points. Relative change: (66-60)/60 = +10%."), "Neither number proves a causal effect without a sound comparison design and uncertainty.")
    },
    agents: {
      0: () => figure("Choose the smaller decision surface", compare(card("Known sequence", tokens(["retrieve", "validate", "answer"])), card("Adaptive task", tokens(["observe", "choose allowed action", "verify / stop"]))), "A fixed workflow can be easier to evaluate and operate. Add autonomy only when the task benefits."),
      2: () => figure("The model proposes; application policy decides", flow(card("Model output", text("Proposed tool + arguments")), card("External gate", tokens(["schema", "identity", "permission", "budget"])), card("Tool execution", text("Typed result or explicit failure"))), "A valid argument schema is not authorization. Tool results are data, not permission to change the policy.")
    },
    "agent-eval": {
      3: () => figure("Timeout does not tell you whether the write committed", flow(card("Write requested", tokens(["idempotency key K"])), card("Response lost", tokens(["outcome unknown"])), card("Reconcile first", text("Read durable operation status; retry safely using K."))), "Blind retries can duplicate external side effects even when the final answer looks correct."),
      8: () => figure("Completion is not the only axis", compare(card("Task result", tokens(["document edited: yes"])), card("Policy / resources", tokens(["unauthorized file touched", "budget exceeded"]))), "Grade verified final state, side effects, resource use, and trajectory compliance separately."),
      9: () => figure("At least once is not every time", grid("One task; four independent reset attempts", [["success", "failure", "failure", "success"]]) + formula("Observed per-attempt success = 2/4. At least one succeeds: yes. All four succeed: no."), "pass@k and all-k consistency answer different questions; aggregate with the stated sampling design.")
    },
    "system-design": {
      0: () => figure("A 45-minute rehearsal budget", `<div class="cd-memory">${[["Clarify", "5 min"], ["Architecture", "10"], ["Deep dive", "15"], ["Evaluation", "10"], ["Operations", "5"]].map(([label, value]) => card(label, text(value))).join("")}</div>`, "An original practice allocation, not an official interview format."),
      1: () => figure("Separate update and query paths", compare(card("Versioned update path", tokens(["parse", "chunk", "index", "invalidate"])), card("Authorized query path", tokens(["retrieve", "rerank", "cite", "abstain"]))), "Join the paths with document versions, permission checks, provenance, and freshness monitoring.")
    },
    "anthropic-fde": {
      2: () => figure("A model pitch is not customer discovery", flow(card("Current workflow", text("What fails? For whom? At what cost?")), card("Pilot comparison", text("Baseline, data restrictions, measured success")), card("Customer decision", text("Adopt, change scope, or do not migrate."))), "Use the question-led mock to role-play the skeptical CTO rather than memorize a sales script."),
      8: () => figure("A collection timer is only one part of latency", flow(card("Collection", text("10 ms")), card("GPU queue", text("0 ms in this idle fixture")), card("Execution", text("20 ms"))) + formula("Arrivals 0,4,9 ms -> dispatch 10 -> finish 30 -> latencies 30,26,21 ms."), "A busy GPU adds queueing. A collection timer is not an end-to-end latency cap.")
    },
    "openai-aie": {
      5: () => figure("An evaluation harness produces a decision artifact", flow(card("Versioned cases", tokens(["input", "expected state", "allowed evidence"])), card("Run + grade", tokens(["validators", "calibrated judges", "error slices"])), card("Release decision", tokens(["quality gate", "risk gate", "cost / latency"]))), "Use the evaluation mock to calculate metrics and defend the decision; a single aggregate score is insufficient."),
      6: () => figure("Inventory state before choosing distributed training", weightState(), "This toy accounting excludes activation peaks and temporary allocations; measure real peaks for the chosen implementation.")
    },
    spark: {
      1: () => figure("A shuffle creates a stage boundary", flow(card("Read partitions", tokens(["P0", "P1"])), card("Narrow filter", text("Each partition works locally.")), card("Group by key", text("Shuffle matching keys together.")), card("Aggregate + write", text("Tasks run over output partitions."))), "The action triggers execution. Partition order is not a global row-order guarantee."),
      3: () => figure("A join can multiply the number of rows", flow(grid("Left keys", [["A"], ["A"]]), grid("Right keys", [["A"], ["A"], ["A"]]), card("Inner join on A", formula("2 left rows x 3 right rows = 6 matches"))), "Validate key uniqueness and output grain before trying to optimize the join.")
    }
  };
  function validate() {
    for (const lesson of GUIDE.lessons) {
      if (!takeaways[lesson.id] || takeaways[lesson.id].length !== lesson.sections.length) throw new RangeError(`Visual-first takeaways must cover every section in ${lesson.id}.`);
      if (!diagrams[lesson.id] || Object.keys(diagrams[lesson.id]).some(index => !lesson.sections[index])) throw new RangeError(`Diagram placement is invalid for ${lesson.id}.`);
    }
  }
  function markup(lesson, section, index) {
    const summary = takeaways[lesson.id][index];
    const diagram = section.diagramStep !== undefined ? figure(`Transformer component ${section.diagramStep + 1}: ${section.title.replace(/^Step \d+: /, "")}`, transformerSteps[section.diagramStep](), "Independent tiny teaching fixture; component examples are not one continuous trained-model forward pass.") : diagrams[lesson.id][index]?.() || "";
    return `<section class="lesson-section" id="section-${index}"><h2>${section.title}</h2><p class="chapter-takeaway">${escape(summary)}</p>${diagram}<details class="chapter-details" id="details-${index}"><summary>Go deeper: explanation, math, and examples</summary><div class="chapter-detail-body">${section.body}</div></details></section>`;
  }
  validate();
  return { markup, takeaways, validate };
})();
