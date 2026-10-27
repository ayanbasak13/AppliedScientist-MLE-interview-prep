"use strict";

const VISUALS = (() => {
  const tokens = ["I", "like", "ML"];
  const X = [[1, 0], [0, 1], [1, 1]];
  const weights = {
    Q: [[1, 1], [0, 1]],
    K: [[1, 0], [1, 1]],
    V: [[1, 2], [0, 1]]
  };
  const format = value => value === -Infinity ? "-inf" : Number.isInteger(value) ? String(value) : value.toFixed(3);
  const sum = values => values.reduce((total, value) => total + value, 0);
  const transpose = matrix => matrix[0].map((_, column) => matrix.map(row => row[column]));
  function multiply(a, b) {
    if (!a.length || !b.length || a[0].length !== b.length || a.some(row => row.length !== b.length) || b.some(row => row.length !== b[0].length)) {
      throw new RangeError("Matrix multiplication requires rectangular [m,k] and [k,n] inputs.");
    }
    return a.map(row => b[0].map((_, column) => sum(row.map((value, k) => value * b[k][column]))));
  }
  function softmax(values) {
    const maximum = Math.max(...values);
    if (!Number.isFinite(maximum) || values.some(value => Number.isNaN(value) || value === Infinity)) {
      throw new RangeError("Softmax requires at least one finite logit and no NaN or positive infinity.");
    }
    const exponents = values.map(value => Math.exp(value - maximum));
    const denominator = sum(exponents);
    return exponents.map(value => value / denominator);
  }
  function attention(q, k, v, causal = false) {
    const raw = multiply(q, transpose(k));
    const scaled = raw.map((row, i) => row.map((value, j) => causal && j > i ? -Infinity : value / Math.sqrt(q[0].length)));
    const probabilities = scaled.map(softmax);
    return { raw, scaled, probabilities, output: multiply(probabilities, v) };
  }
  function onlineAttention(q, k, v, tileSize = 2) {
    if (!Number.isInteger(tileSize) || tileSize < 1 || k.length !== v.length || q.length !== k[0].length) {
      throw new RangeError("Online attention requires compatible Q/K/V and a positive integer tile size.");
    }
    const scores = k.map(row => sum(row.map((value, i) => value * q[i])) / Math.sqrt(q.length));
    let maximum = -Infinity;
    let normalizer = 0;
    let numerator = v[0].map(() => 0);
    const snapshots = [{ maximum, normalizer, numerator: [...numerator], output: null, processed: 0, rescale: 0 }];
    for (let start = 0; start < scores.length; start += tileSize) {
      const tileScores = scores.slice(start, start + tileSize);
      const nextMaximum = Math.max(maximum, ...tileScores);
      const rescale = Math.exp(maximum - nextMaximum);
      const tileWeights = tileScores.map(score => Math.exp(score - nextMaximum));
      normalizer = rescale * normalizer + sum(tileWeights);
      numerator = numerator.map((value, column) => rescale * value + sum(tileWeights.map((weight, i) => weight * v[start + i][column])));
      maximum = nextMaximum;
      snapshots.push({
        maximum, normalizer, numerator: [...numerator], output: numerator.map(value => value / normalizer),
        processed: Math.min(start + tileSize, scores.length), rescale
      });
    }
    return { scores, snapshots };
  }
  function pageAllocation(a, b, size = 4) {
    const aPages = [4, 1, 5].slice(0, Math.ceil(a / size));
    const bPages = [0, 3].slice(0, Math.ceil(b / size));
    return { aPages, bPages, used: aPages.length + bPages.length, free: 6 - aPages.length - bPages.length, tail: size * (aPages.length + bPages.length) - a - b };
  }

  function matrix(label, values, options = {}) {
    const { row = -1, column = -1, hit = false, heat = false, scores = false } = options;
    return `<figure class="viz-matrix"><figcaption>${label} <small>[${values.length}, ${values[0].length}]</small></figcaption><table aria-label="${label} matrix, ${values.length} rows and ${values[0].length} columns"><tbody>${values.map((items, r) => `<tr>${items.map((value, c) => {
      const selected = r === row || c === column;
      const cellHit = r === row && c === column;
      const color = heat && value !== -Infinity ? ` style="--heat:${Math.max(0, Math.min(1, value))}"` : "";
      const className = `${selected ? " selected" : ""}${hit && cellHit ? " cell-hit" : ""}${value === -Infinity ? " masked" : ""}${heat ? " heat" : ""}`;
      const content = scores ? `<button type="button" data-score-row="${r}" data-score-column="${c}" aria-label="Inspect query ${tokens[r]}, key ${tokens[c]}, raw score ${format(value)}">${format(value)}</button>` : format(value);
      return `<td class="${className.trim()}"${color}>${content}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table></figure>`;
  }
  const node = (label, detail = "", tone = "") => `<div class="viz-node ${tone}"><strong>${label}</strong>${detail ? `<small>${detail}</small>` : ""}</div>`;
  const arrow = '<span class="viz-arrow" aria-hidden="true">&rarr;</span>';
  const formula = text => `<div class="viz-formula">${text}</div>`;
  const note = text => `<p class="viz-note">${text}</p>`;
  const metric = (label, value, id = "") => `<div class="viz-metric"${id ? ` data-metric="${id}"` : ""}><small>${label}</small><strong>${value}</strong></div>`;
  const control = (id, label, input) => `<label class="viz-control" for="${id}"><span>${label}</span>${input}</label>`;
  const select = (id, name, choices, selected) => `<select id="${id}" data-control="${name}">${choices.map(([value, label]) => `<option value="${value}"${String(value) === String(selected) ? " selected" : ""}>${label}</option>`).join("")}</select>`;
  const range = (id, name, minimum, maximum, value) => `<input id="${id}" data-control="${name}" type="range" min="${minimum}" max="${maximum}" step="1" value="${value}">`;
  const stepButtons = count => `<div class="viz-step-controls"><button type="button" class="button secondary small-button" data-action="previous">Previous</button><span data-step-label></span><button type="button" class="button small-button" data-action="next">Next &rarr;</button><span class="viz-sr-only">${count} steps available</span></div>`;

  function graph(id, nodes, edges, active, label) {
    return `<div class="viz-graph-scroll" tabindex="0" role="group" aria-label="${label}. Scroll horizontally on small screens."><svg class="viz-graph" viewBox="0 0 760 255" role="img" aria-labelledby="${id}-graph-title"><title id="${id}-graph-title">${label}. Current node: ${active}.</title><defs><marker id="${id}-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#809778"/></marker></defs>${edges.map(edge => `<path d="${edge}" fill="none" stroke="#9eaf96" stroke-width="2" marker-end="url(#${id}-arrow)"/>`).join("")}${nodes.map(([key, x, y, title, detail]) => `<g class="viz-graph-node ${key === active ? "is-current" : ""}"><rect x="${x}" y="${y}" width="145" height="62" rx="8"/><text x="${x + 72}" y="${y + 26}" text-anchor="middle">${title}</text><text class="viz-svg-small" x="${x + 72}" y="${y + 46}" text-anchor="middle">${detail}</text></g>`).join("")}</svg></div>`;
  }

  const specs = {
    attention: {
      title: "Q, K, V: watch the matrix multiplication", subtitle: "Follow one head from input vectors to a weighted mixture of values. Every displayed result is computed, not a decorative image.",
      initial: () => ({ step: 0, query: 1, key: 0, projection: "Q", causal: true, component: 0 }),
      steps: 5,
      controls: id => `<div class="viz-controls">${control(`${id}-query`, "Query token / highlighted row", select(`${id}-query`, "query", tokens.map((token, i) => [i, token]), 1))}${control(`${id}-key`, "Key token / highlighted column", select(`${id}-key`, "key", tokens.map((token, i) => [i, token]), 0))}${control(`${id}-projection`, "Projection to inspect", select(`${id}-projection`, "projection", ["Q", "K", "V"].map(key => [key, key]), "Q"))}${control(`${id}-component`, "Output component", select(`${id}-component`, "component", [[0, "0"], [1, "1"]], 0))}<label class="viz-check"><input type="checkbox" data-control="causal" checked> Causal mask</label></div>${stepButtons(5)}`,
      paint: state => {
        const q = multiply(X, weights.Q);
        const k = multiply(X, weights.K);
        const v = multiply(X, weights.V);
        const result = attention(q, k, v, state.causal);
        const r = state.query;
        const c = state.key;
        const titles = ["1. Project the same input three ways", "2. Each query row meets each key row", "3. Scale, then hide future positions", "4. Softmax each row independently", "5. Mix value vectors with those weights"];
        let body = "";
        if (state.step === 0) {
          body = `<div class="viz-projections">${["Q", "K", "V"].map(key => node(key, { Q: "What am I looking for?", K: "What can I match?", V: "What content do I provide?" }[key], key === state.projection ? "accent" : "")).join("")}</div><div class="viz-equation-row">${matrix("X", X, { row: r })}<span>&times;</span>${matrix(`W${state.projection}`, weights[state.projection])}<span>=</span>${matrix(state.projection, multiply(X, weights[state.projection]), { row: r })}</div>${formula(`${state.projection} = X W${state.projection}: [T,d_model] &times; [d_model,d_head] = [T,d_head]`)}${note("These small, fixed weights are illustrative, not trained embeddings. Q, K, and V are different learned projections of the same hidden states in self-attention.")}`;
        } else if (state.step === 1) {
          body = `<div class="viz-equation-row">${matrix("Q", q, { row: r })}<span>&times;</span>${matrix("K transpose", transpose(k), { column: c })}<span>=</span>${matrix("QK transpose", result.raw, { row: r, column: c, hit: true, scores: true })}</div>${formula(`score[${r},${c}] = ${q[r].map((value, i) => `${value} &times; ${k[c][i]}`).join(" + ")} = <strong>${format(result.raw[r][c])}</strong>`)}${note(`The highlighted Q row belongs to "${tokens[r]}". The highlighted column of K transpose is the key vector for "${tokens[c]}". The shared dimension d_head=2 is summed out; [3,2] times [2,3] gives one score for every query/key pair: [3,3]. Click a score cell to inspect another pair.`)}`;
        } else if (state.step === 2) {
          body = `<div class="viz-equation-row">${matrix("Raw scores", result.raw, { row: r })}<span>&rarr;</span>${matrix("Scaled + mask", result.scaled, { row: r })}</div>${formula(`score / sqrt(2) + mask; selected pair = ${format(result.raw[r][c])} / 1.414 ${state.causal && c > r ? "+ (-inf) = -inf" : `= ${format(result.scaled[r][c])}`}`)}${note(state.causal ? "Rows are queries, columns are keys. Positions above the diagonal are future tokens: -inf logits become zero probability after softmax. The diagonal stays visible." : "Mask disabled for comparison: future keys can now affect earlier queries. This is not valid causal next-token attention.")}`;
        } else if (state.step === 3) {
          const rowMax = Math.max(...result.scaled[r]);
          const exps = result.scaled[r].map(value => Math.exp(value - rowMax));
          body = `<div class="viz-equation-row">${matrix("Scaled + mask", result.scaled, { row: r })}<span>&rarr;</span>${matrix("A = row softmax", result.probabilities, { row: r, heat: true })}</div>${formula(`Row "${tokens[r]}": exp(logits - max) = [${exps.map(format).join(", ")}]<br>divide each by ${format(sum(exps))}; row sum = <strong data-attention-sum>${sum(result.probabilities[r]).toFixed(6)}</strong>`)}${note("Darker cells carry more attention, and the numbers give the exact rounded weights. A masked key has weight 0. Normalize over keys within each query row, never over the entire matrix.")}`;
        } else {
          const component = state.component;
          body = `<div class="viz-equation-row">${matrix("A", result.probabilities, { row: r, heat: true })}<span>&times;</span>${matrix("V", v, { column: component })}<span>=</span>${matrix("O", result.output, { row: r, column: component, hit: true })}</div>${formula(`O[${r},${component}] = ${result.probabilities[r].map((weight, j) => `${format(weight)} &times; ${v[j][component]}`).join(" + ")} = <strong data-attention-output>${format(result.output[r][component])}</strong>`)}${note("For each output component, multiply each key's attention weight by that key's value component and sum. Attention returns a vector mixture, not a token ID. Logits and token sampling come later.")}`;
        }
        return `<div class="viz-step-name">${titles[state.step]}</div>${body}${note("Single-head toy: batch is omitted, T=3, d_model=d_head=2. Real models use larger projections, multiple heads, positional handling, residuals, and normalization.")}`;
      }
    },
    architecture: {
      title: "Architecture first: three transformer families", subtitle: "Begin with the whole model. Compare who can attend to whom, then inspect the components inside a common decoder block.",
      initial: () => ({ family: "decoder", stage: 0 }),
      controls: id => `<div class="viz-controls">${control(`${id}-family`, "Architecture family", select(`${id}-family`, "family", [["decoder", "Decoder-only (generative LLM)"], ["encoder", "Encoder-only (representations)"], ["seq2seq", "Encoder–decoder (source → target)"]], "decoder"))}${control(`${id}-stage`, "Inspect a stage", select(`${id}-stage`, "stage", [[0, "Tokens"], [1, "Embeddings"], [2, "Attention + residual"], [3, "MLP + residual"], [4, "Output / prediction"]], 0))}</div>`,
      paint: state => {
        if (state.family === "encoder") {
          const descriptions = [
            "The input is tokenized as one source sequence. Padding is not evidence: exclude padded key positions with the appropriate mask.",
            "Token embeddings and positional information initialize a vector per position. Task-specific special tokens or segment embeddings depend on the architecture.",
            "Bidirectional self-attention lets each input position use other non-padding input positions. All Q/K/V come from that encoder layer's input representations; there is no future-token restriction in this illustrative encoder.",
            "Attention and position-wise MLP sublayers each have residual/normalization structure. The MLP transforms features independently at each position. The original paper used post-norm; other encoders differ.",
            "The encoder produces contextual representations [B,T,d_model]. A task head may predict a class per token, pool into a sequence representation, or predict masked tokens. An embedding model needs an appropriate pooling/projection and training objective—not just an arbitrary vector taken from a generative model."
          ];
          return `<div class="viz-flow-row">${node("Source text", "Entire input", state.stage === 0 ? "accent" : "")}${arrow}${node("Token + position", "[B,T,d_model]", state.stage === 1 ? "accent" : "")}</div><div class="viz-decoder-block"><span class="eyebrow">ENCODER BLOCK &times; L</span><div class="viz-flow-row">${node("Self-attention", "Bidirectional; padding masked", state.stage === 2 ? "accent" : "")}${arrow}${node("Residual + norm", "Placement varies")}${arrow}${node("Position-wise MLP", "Residual + norm", state.stage === 3 ? "accent" : "")}</div></div><div class="viz-flow-row">${node("Contextual representations", "[B,T,d_model]")}${arrow}${node("Task-specific head", "Pooling / tagging / classification", state.stage === 4 ? "accent" : "")}</div><div class="viz-explanation" role="status">${descriptions[state.stage]}</div>${note("Bidirectional describes which positions may exchange information, not a recurrent forward/backward pass. Encoder architectures and downstream heads vary.")}`;
        }
        if (state.family === "seq2seq") {
          const descriptions = [
            "Source and target are separate sequences and can have different lengths S and T. During teacher-forced training, target inputs are shifted right so each position predicts the next target token.",
            "The encoder embeds source tokens and the decoder embeds target tokens, each with positional information. Do not concatenate the two diagrams into a single causal-only stack.",
            "The encoder uses bidirectional self-attention. The decoder first uses causal self-attention, then cross-attention: Q comes from decoder states; K/V come from encoder outputs. Cross-attention scores are [B,H,T,S], not necessarily square.",
            "Both stacks include position-wise MLPs plus residual/normalization around sublayers. The diagram groups these operations and is not a claim about exact norm placement. Encoder outputs are reused as cross-attention memory.",
            "The decoder output head predicts a target-vocabulary distribution. Training evaluates target positions in parallel with a causal mask; generation feeds back one sampled target token at a time until a stop condition."
          ];
          return `<div class="viz-flow-row">${node("Source tokens", "Length S", state.stage === 0 ? "accent" : "")}${arrow}${node("Source embeddings", "[B,S,d_model]", state.stage === 1 ? "accent" : "")}${arrow}${node("Encoder blocks", "Bidirectional attention + MLP", state.stage === 2 ? "accent" : "")}</div><div class="viz-cross-memory"><strong>Encoder output memory [B,S,d_model]</strong><span>↓ supplies K and V to decoder cross-attention</span></div><div class="viz-flow-row">${node("Shifted target tokens", "Length T", state.stage === 0 ? "accent" : "")}${arrow}${node("Target embeddings + position", "[B,T,d_model]", state.stage === 1 ? "accent" : "")}</div><div class="viz-decoder-block"><span class="eyebrow">DECODER BLOCK &times; L / RESIDUALS + NORMALIZATION AROUND SUBLAYERS</span><div class="viz-flow-row">${node("Causal self-attention", "Q/K/V from decoder", state.stage === 2 ? "accent" : "")}${arrow}${node("Cross-attention", "Q: decoder / K,V: encoder", state.stage === 2 ? "accent" : "")}${arrow}${node("Position-wise MLP", "Feature mixing", state.stage === 3 ? "accent" : "")}</div></div><div class="viz-flow-row">${node("Target vocabulary head", "[B,T,V_target]", state.stage === 4 ? "accent" : "")}${arrow}${node("Next target token", "Predict then feed back", state.stage === 4 ? "accent" : "")}</div><div class="viz-explanation" role="status">${descriptions[state.stage]}</div>${note("This is the high-level family used by the original Transformer. Source attention may read the entire authorized source; the causal restriction applies to target self-attention. Norm placement and position encoding are separate architecture choices.")}`;
        }
        const descriptions = [
          "Tokenization maps text to IDs. The toy sentence has three tokens for illustration; real tokenizer boundaries can differ.",
          "Embedding lookup produces [B,T,d_model] hidden states. Position must also be represented. Absolute embeddings can be added here; RoPE instead rotates Q/K inside attention.",
          "In this illustrative pre-norm block, normalize, compute causal multi-head attention, project the concatenated heads, then add the original residual stream. RoPE, when used, acts on Q/K, not V.",
          "Normalize and apply a position-wise MLP (often gated), then add its input residual. The MLP mixes feature dimensions separately at each token; attention mixes information across tokens.",
          "Repeat the block L times, apply a final normalization and vocabulary projection, and softmax over vocabulary logits. At inference, the final prompt position predicts the first new token."
        ];
        return `<div class="viz-flow-row">${node("Text", '"I like ML"', state.stage === 0 ? "accent" : "")}${arrow}${node("Token IDs", "[1, T]", state.stage === 0 ? "accent" : "")}${arrow}${node("Embedding lookup", "[1, T, d_model]", state.stage === 1 ? "accent" : "")}</div><div class="viz-decoder-block"><span class="eyebrow">ONE PRE-NORM DECODER BLOCK &times; L</span><div class="viz-residual"><span>Residual x bypasses the sublayer</span><div class="viz-flow-row">${node("Norm", "Per token")}${arrow}${node("Multi-head attention", "Causal Q/K/V + position", state.stage === 2 ? "accent" : "")}${arrow}${node("Add x", "Residual sum", state.stage === 2 ? "accent" : "")}</div></div><div class="viz-residual"><span>Updated hidden state bypasses the MLP</span><div class="viz-flow-row">${node("Norm", "Per token")}${arrow}${node("MLP", "Feature mixing", state.stage === 3 ? "accent" : "")}${arrow}${node("Add residual", "[1, T, d_model]", state.stage === 3 ? "accent" : "")}</div></div></div><div class="viz-flow-row">${node("Final norm")}${arrow}${node("Vocabulary projection", "[1, T, Vocab]")}${arrow}${node("Softmax / sample", "Next-token distribution", state.stage === 4 ? "accent" : "")}</div><div class="viz-explanation" role="status">${descriptions[state.stage]}</div>${note("This is one common architecture, not a claim that every transformer has identical norm placement, MLP type, position encoding, or output-weight tying.")}`;
      }
    },
    heads: {
      title: "Split heads. Attend. Concatenate.", subtitle: "Follow the feature dimension without confusing a reshape with a change in token order.",
      initial: () => ({ width: 8, heads: 2 }),
      controls: id => `<div class="viz-controls">${control(`${id}-width`, "Model width", select(`${id}-width`, "width", [[8, "8"], [16, "16"], [24, "24"]], 8))}${control(`${id}-heads`, "Query heads", select(`${id}-heads`, "heads", [[1, "1"], [2, "2"], [4, "4"], [8, "8"]], 2))}</div>`,
      paint: state => {
        const d = state.width / state.heads;
        return `<div class="viz-flow-row">${node("Project Q / K / V", `[B=1,T=3,d=${state.width}]`)}${arrow}${node("Reshape", `[1,3,${state.heads},${d}]`)}${arrow}${node("Transpose", `[1,${state.heads},3,${d}]`)}</div><div class="viz-heads">${Array.from({ length: state.heads }, (_, i) => `<div class="viz-head"><strong>Head ${i}</strong><small>Q/K/V: [3, ${d}]</small><div class="viz-feature-row">${Array.from({ length: d }, () => '<i></i>').join("")}</div><small>Own attention weights [3,3]</small></div>`).join("")}</div><div class="viz-flow-row">${node("Head outputs", `${state.heads} &times; [3,${d}]`)}${arrow}${node("Concat + W output", `[1,3,${state.width}]`)}</div>${formula(`d_head = d_model / H = ${state.width} / ${state.heads} = <strong data-head-width>${d}</strong>`)}${note("The feature bars illustrate the split only, not the numerical head outputs. Real heads have different learned projections. This diagram uses ordinary multi-head attention; GQA/MQA can have fewer K/V heads than query heads.")}`;
      }
    },
    kv: {
      title: "Prefill once. Reuse keys and values.", subtitle: "Move from a parallel prompt prefill to sequential decode. Caching avoids re-projecting old K/V; attention still reads relevant cached state.",
      initial: () => ({ added: 0 }),
      controls: id => control(`${id}-added`, "Decode tokens already fed back and processed (0-4)", range(`${id}-added`, "added", 0, 4, 0)),
      paint: state => {
        const sequence = ["The", "model", "predicts", "one", "token", "at", "a-time"];
        const length = 3 + state.added;
        return `<div class="viz-token-strip">${sequence.map((token, i) => `<div class="viz-token ${i < length ? "present" : "pending"} ${i === length - 1 ? "current" : ""}"><strong>${token}</strong><small>${i < 3 ? "prompt" : i < length ? "processed decode" : "not processed"}</small></div>`).join("")}</div><div class="viz-flow-row">${node(state.added ? "One current query" : "Prompt queries", state.added ? "Q: [1,d_head]" : "Q: [3,d_head]", "accent")}${arrow}${node("Attend to visible K/V", `Cache: ${length} rows each`)}${arrow}${node("Predict next token", "Logits, then sampling")}</div><div class="viz-cache-rows"><div><strong>K cache</strong>${sequence.slice(0, length).map((token, i) => `<span class="${i === length - 1 ? "current" : ""}">K(${token})</span>`).join("")}</div><div><strong>V cache</strong>${sequence.slice(0, length).map((token, i) => `<span class="${i === length - 1 ? "current" : ""}">V(${token})</span>`).join("")}</div></div><div class="viz-metrics">${metric("Processed sequence length", length, "cache-length")}${metric("Stored rows per layer / KV head", `2 &times; ${length}`)}</div><div class="viz-explanation" role="status">${state.added === 0 ? "Prefill processes all three prompt positions (with a causal mask). Its last-position logits predict the first new token. That newly sampled token is not in the cache until it is fed back and processed." : `After ${state.added} sampled token${state.added === 1 ? " has" : "s have"} been fed back, their K/V rows are appended. The current query reads ${length} visible keys; this step's logits predict the following token.`}</div>${formula("KV bytes = 2 &times; batch &times; sequence length &times; layers &times; KV heads &times; head width &times; bytes/value")}${note("This diagram omits batching, padding, sliding windows, cache eviction, and tensor layouts. The cache grows linearly with processed sequence length; a full prefill score matrix grows quadratically.")}`;
      }
    },
    paged: {
      title: "PagedAttention: logical order, scattered storage", subtitle: "A request's token order is logical. Its KV blocks can live in noncontiguous physical memory pages.",
      initial: () => ({ a: 7, b: 5, address: 0, release: false }),
      controls: id => `<div class="viz-controls">${control(`${id}-a`, "Request A cached tokens (1-12)", range(`${id}-a`, "a", 1, 12, 7))}${control(`${id}-b`, "Request B cached tokens (1-8)", range(`${id}-b`, "b", 1, 8, 5))}${control(`${id}-address`, "Inspect A token offset", range(`${id}-address`, "address", 0, 6, 0))}<label class="viz-check"><input type="checkbox" data-control="release"> Release request B</label></div>`,
      paint: state => {
        const b = state.release ? 0 : state.b;
        const pages = pageAllocation(state.a, b);
        const logical = Math.floor(state.address / 4);
        const physical = pages.aPages[logical];
        const offset = state.address % 4;
        const owners = new Map([...pages.aPages.map((page, i) => [page, { name: "A", logical: i, count: state.a }]), ...pages.bPages.map((page, i) => [page, { name: "B", logical: i, count: b }])]);
        return `<div class="viz-page-tables">${["A", "B"].map(name => {
          const mapping = name === "A" ? pages.aPages : pages.bPages;
          return `<div><h4>Request ${name} page table</h4><div class="viz-page-map">${mapping.length ? mapping.map((page, i) => `<div class="${name === "A" && i === logical ? "current" : ""}">Logical ${i} &rarr; physical ${page}</div>`).join("") : "<p>Released: no allocated pages</p>"}</div></div>`;
        }).join("")}</div><div class="viz-physical-pages">${Array.from({ length: 6 }, (_, page) => {
          const owner = owners.get(page);
          return `<div class="viz-page ${owner ? `owner-${owner.name.toLowerCase()}` : "free"}"><strong>Physical ${page} ${owner ? `/ ${owner.name}` : "/ FREE"}</strong><div>${Array.from({ length: 4 }, (_, slot) => {
            const token = owner ? owner.logical * 4 + slot : -1;
            const occupied = owner && token < owner.count;
            return `<span class="${occupied ? "occupied" : "unused"} ${owner && owner.name === "A" && page === physical && slot === offset ? "address-hit" : ""}">${occupied ? `${owner.name}${token}` : owner ? "tail" : "free"}</span>`;
          }).join("")}</div></div>`;
        }).join("")}</div>${formula(`A token ${state.address}: logical page floor(${state.address}/4) = ${logical}; physical page = <strong data-physical-address>${physical}</strong>; offset = ${offset}`)}<div class="viz-metrics">${metric("Payload tokens", state.a + b)}${metric("Allocated pages", pages.used)}${metric("Free pages", pages.free, "free-pages")}${metric("Unused allocated tail slots", pages.tail, "tail-slots")}</div>${note("Each slot denotes storage for a token's K/V state, not its attention score. Page size is 4 toy slots. A free page can be reused; unused tail slots remain allocated until that request's page is freed. Paging reduces allocation waste and the need for contiguous growth; it does not change attention into an approximation or remove KV reads.")}${note("Prefix sharing and copy-on-write are possible extensions but are not simulated here. Memory savings versus a contiguous allocator depend on its reservation policy; no universal speedup or byte savings are claimed.")}`;
      }
    },
    flash: {
      title: "FlashAttention: stream tiles, keep a running softmax", subtitle: "Follow actual online-softmax arithmetic for one query and four keys. No full score/probability matrix needs to be stored in device HBM.",
      initial: () => ({ step: 0 }),
      steps: 3,
      controls: () => stepButtons(3),
      paint: state => {
        const q = [1, 1];
        const k = [[1, 0], [1, 1], [0, 1], [2, 1]];
        const v = [[1, 0], [0, 2], [1, 1], [2, 0]];
        const online = onlineAttention(q, k, v, 2);
        const snapshot = online.snapshots[state.step];
        const dense = attention([q], k, v).output[0];
        const start = Math.max(0, snapshot.processed - 2);
        const currentK = state.step ? k.slice(start, snapshot.processed) : [];
        const currentV = state.step ? v.slice(start, snapshot.processed) : [];
        return `<div class="viz-memory-compare"><div><h4>Naive materialization</h4><div class="viz-flow-row">${node("HBM: Q, K, V")}${arrow}${node("Write / read S and P", "Full [T,T] intermediates", "warm")}${arrow}${node("Output O")}</div></div><div><h4>Flash-style computation</h4><div class="viz-flow-row">${node("HBM: Q, K, V")}${arrow}${node("On-chip tile", "Q + current K/V block", "accent")}${arrow}${node("Running m, l, u", "Write final O to HBM")}</div></div></div><div class="viz-tile-strip">${k.map((_, i) => `<div class="${i < snapshot.processed ? "processed" : "pending"} ${state.step && i >= start && i < snapshot.processed ? "current" : ""}"><strong>Key ${i}</strong><small>score ${format(online.scores[i])}</small><span>${i < snapshot.processed ? "processed" : "not processed"}</span></div>`).join("")}</div>${state.step ? `<div class="viz-equation-row">${matrix("Q", [q])}${matrix("Current K tile", currentK)}${matrix("Current V tile", currentV)}</div>` : note("Before the first tile: m=-inf, l=0, u=[0,0]. No output is defined yet because no key has been processed.")}<div class="viz-metrics">${metric("Running maximum m", format(snapshot.maximum), "flash-max")}${metric("Running exp sum l", format(snapshot.normalizer), "flash-normalizer")}${metric("Old-state rescale", format(snapshot.rescale))}${metric("Weighted numerator u", `[${snapshot.numerator.map(format).join(", ")}]`)}</div>${formula("m' = max(m, max(tile scores))<br>l' = exp(m - m') l + sum exp(tile scores - m')<br>u' = exp(m - m') u + sum exp(tile scores - m') V<br>O = u' / l'")}<div class="viz-output-pair">${metric(state.step === 2 ? "Final tiled output" : "Current prefix output", snapshot.output ? `[${snapshot.output.map(format).join(", ")}]` : "Not defined", "flash-output")}${metric("Dense reference: all four keys", `[${dense.map(format).join(", ")}]`)}</div><div class="viz-explanation" role="status">${state.step === 2 ? "Both tiles are included: tiled and dense outputs agree up to floating-point rounding. Rescaling the old numerator and denominator when the maximum changes is essential." : state.step === 1 ? "Only the first two keys are included. This is a prefix result, so it need not equal the four-key reference yet. Next, the larger score in tile 2 changes the running maximum." : "Step forward to load a two-key tile into the illustrative on-chip working set."}</div>${note("This isolates one query row, two-component values, and a two-key tile. Real kernels tile many queries and use hardware-specific scheduling. FlashAttention reduces IO/materialization; dense attention arithmetic is still quadratic in sequence length. FlashAttention-2 improves parallelism/work partitioning. PagedAttention, by contrast, manages KV storage. No GPU execution or measured speedup is implied.")}`;
      }
    },
    gpu: {
      title: "The GPU memory and timing picture", subtitle: "See why a smaller intermediate can matter as much as fewer operations—and why dispatch time is not completion time.",
      initial: () => ({ bottleneck: "memory" }),
      controls: id => control(`${id}-bottleneck`, "Inspect a bottleneck", select(`${id}-bottleneck`, "bottleneck", [["memory", "Memory traffic"], ["compute", "Matrix compute"], ["launch", "Host dispatch / synchronization"]], "memory")),
      paint: state => {
        const messages = {
          memory: "If reading weights, KV state, or intermediates is the bottleneck, reduce traffic and improve reuse. Flash-style tiling keeps local work on-chip. Paged KV storage improves allocation, not unlimited memory bandwidth.",
          compute: "If supported matrix multiplications dominate, efficient shapes, precision modes, Tensor Core kernels, and sufficient work can help. Lower precision still requires numerical and task-quality validation.",
          launch: "Many tiny operations or accidental synchronization can leave device gaps. Profile dispatch, copies, dependencies, and kernels. A CPU timer around asynchronous launches may measure enqueue time rather than GPU completion."
        };
        return `<div class="viz-flow-row">${node("CPU / host", "Launches work; host RAM", state.bottleneck === "launch" ? "accent" : "")}${arrow}${node("Device HBM", "Weights, KV, activations", state.bottleneck === "memory" ? "accent" : "")}${arrow}${node("On-chip memory", "Shared SRAM + registers", state.bottleneck === "memory" ? "accent" : "")}${arrow}${node("Compute units", "CUDA / Tensor Core work", state.bottleneck === "compute" ? "accent" : "")}</div><div class="viz-timeline"><div><strong>CPU</strong><span>enqueue A</span><span>enqueue B</span><i>host can continue</i></div><div><strong>GPU stream</strong><span class="gpu-work">kernel A</span><span class="gpu-work">kernel B</span><b>completion event</b></div></div><div class="viz-explanation" role="status">${messages[state.bottleneck]}</div>${note("Timeline blocks are qualitative, not a measured trace. Use device events or appropriate synchronization for device timing; include queueing, transfers, retrieval, and networking for an end-to-end product metric. Separate streams need explicit dependencies where applicable.")}`;
      }
    },
    agent: {
      title: "An agent is a bounded decision loop", subtitle: "Walk through observations and actions. Tool execution, permissions, and human approval stay outside the model's authority.",
      initial: () => ({ scenario: "success", step: 0 }),
      controls: id => `<div class="viz-controls">${control(`${id}-scenario`, "Simulate a case", select(`${id}-scenario`, "scenario", [["success", "Evidence found"], ["timeout", "Read tool times out"], ["injection", "Document requests unauthorized publication"], ["budget", "Step budget exhausted"]], "success"))}</div>${stepButtons(8)}`,
      paint: (state, id) => {
        const traces = {
          success: [
            ["task", "Task", "Find policy evidence and draft a supported update. Publication is not authorized."],
            ["choose", "Choose an action", "The model proposes a read-only search tool with typed arguments."],
            ["gate", "Validate outside the model", "The executor checks schema, source permissions, remaining budget, and tool scope."],
            ["tool", "Execute allowed tool", "A simulated read returns a source ID, version, and a relevant policy paragraph."],
            ["observe", "Update structured state", "Record the source version and observation. The model may decide whether more evidence is needed."],
            ["choose", "Choose again from evidence", "The model proposes a patch rather than inventing a missing fact."],
            ["review", "Validate the draft", "Check claim support, source version, diff scope, and the expected proposed-document state."],
            ["finish", "Draft ready, not published", "The verified simulated result is a draft. A separate authorized reviewer must approve publication."]
          ],
          timeout: [
            ["task", "Task", "Fetch evidence within a two-attempt read budget."],
            ["choose", "Propose read", "Choose a permitted read tool."],
            ["gate", "Check budget and permissions", "Attempt 1 is allowed."],
            ["tool", "Timeout", "No result was observed. For this read-only operation a bounded retry may be appropriate."],
            ["observe", "Record typed failure", "Do not replace an error with fabricated evidence. Back off and check remaining attempts."],
            ["choose", "Retry once", "The simulated retry succeeds and returns versioned evidence."],
            ["review", "Validate evidence and draft", "The draft must still pass support checks."],
            ["finish", "Return verified draft", "Report the recovery and result. A timed-out write would instead require an outcome/idempotency check before retry."]
          ],
          injection: [
            ["task", "Task", "Read evidence and propose a draft; no publication permission."],
            ["choose", "Choose search", "A normal read is proposed."],
            ["gate", "Allow read only", "The read contract and scope are valid."],
            ["tool", "Untrusted tool result", "A retrieved document includes text asking the system to publish and send private data."],
            ["observe", "Data, not authority", "Treat the embedded request as untrusted content, not a new permission."],
            ["gate", "Block prohibited action", "Even if a model proposes publication, the executor denies an unauthorized tool action."],
            ["review", "Assess remaining evidence", "Use legitimate evidence if sufficient; otherwise ask or abstain. Log only appropriate, redacted failure information."],
            ["finish", "No unauthorized side effect", "The simulated safety outcome is externally enforced denial—not trust in a promise from the model."]
          ],
          budget: [
            ["task", "Task", "Find evidence with a bounded step/time/cost budget."],
            ["choose", "Propose another search", "The next proposed action must pass a budget check."],
            ["gate", "Budget exhausted", "The executor detects no remaining tool allowance and does not execute the call."],
            ["finish", "Explicit incomplete result", "Stop, preserve useful authorized state, and explain what remains unresolved. Do not claim completion."]
          ]
        };
        const trace = traces[state.scenario];
        const current = trace[Math.min(state.step, trace.length - 1)];
        const nodes = [
          ["task", 5, 15, "Task + state", "Goal / constraints"], ["choose", 200, 15, "Choose action", "Model proposes"],
          ["gate", 400, 15, "Execution gate", "Auth / schema / budget"], ["tool", 600, 15, "Tool execution", "Typed result / error"],
          ["observe", 600, 165, "Observation", "Update state"], ["review", 200, 165, "Review / verify", "Support / approval"],
          ["finish", 5, 165, "Finish or stop", "Verified / incomplete"]
        ];
        const edges = ["M150,46 H198", "M345,46 H398", "M545,46 H598", "M672,77 V163", "M600,195 H575 V112 H272 V79", "M272,79 V163", "M200,196 H152", "M472,79 V245 H77 V229"];
        return `${graph(id, nodes, edges, current[0], "Agent loop with a return from observation to action selection and an external stopping gate")}<div class="viz-agent-status"><span>SIMULATED STEP ${state.step + 1} / ${trace.length}</span><h4>${current[1]}</h4><p role="status">${current[2]}</p></div><ol class="viz-trace">${trace.slice(0, state.step + 1).map((item, i) => `<li class="${i === state.step ? "current" : ""}"><strong>${item[1]}</strong><span>${i === state.step ? item[2] : "Completed simulated transition"}</span></li>`).join("")}</ol>${note("The back edge means a tool result can trigger another decision. A fixed workflow predetermines transitions; an agent chooses within allowed transitions. This is a deterministic educational simulation: no model, tool, network action, write, or hidden reasoning trace is executed.")}`;
      }
    },
    rag: {
      title: "RAG: see where each failure is measured", subtitle: "Follow a synthetic question through evidence and generation. Retrieval quality and answer quality are separate checks.",
      initial: () => ({ scenario: "supported", stage: 0 }),
      controls: id => `<div class="viz-controls">${control(`${id}-scenario`, "Evidence condition", select(`${id}-scenario`, "scenario", [["supported", "Authorized and current"], ["missing", "Required evidence missing"], ["stale", "Evidence is stale"], ["denied", "User lacks source access"]], "supported"))}${control(`${id}-stage`, "Inspect checkpoint", select(`${id}-stage`, "stage", [[0, "Ingest / version"], [1, "Authorize / retrieve"], [2, "Rerank / context"], [3, "Generate / cite"], [4, "Evaluate / monitor"]], 0))}</div>`,
      paint: state => {
        const conditions = {
          supported: ["v3: effective date Feb 1", "Authorized v3 chunk retrieved", "Required date is in context", "Answer with supported date + source ID", "Check retrieval, claim support, correctness, latency, cost"],
          missing: ["Required record absent", "No relevant authorized chunk", "Context cannot support the date", "Abstain or clarify; do not invent", "Evidence recall fails; correct abstention can still succeed"],
          stale: ["Only v2 indexed; v3 exists", "A relevant but stale chunk retrieved", "Context faithfully contains the old date", "A faithful answer can be wrong today", "Check source age, index lag, and temporal correctness separately"],
          denied: ["Source carries an access policy", "External ACL gate excludes the chunk", "No restricted text enters context", "Explain lack of accessible evidence", "Audit authorization across retrieval, caches, traces, and judges"]
        };
        const labels = ["Ingest + version", "ACL + retrieve", "Rerank + context", "Generate + cite", "Evaluate + monitor"];
        const descriptions = [
          "Parse and chunk sources, preserve IDs/spans/version/access metadata, embed, and index. Propagate updates and deletions.",
          "Enforce permissions before exposing content. Then measure relevant-evidence recall and candidate coverage on labeled cases.",
          "Rerank candidates, deduplicate, and fit a context budget. Measure ordering quality and coverage of required facts in the final context.",
          "Generate from evidence, cite supporting source spans, and follow an explicit abstention policy. A citation's presence alone is not support.",
          "Evaluate retrieval recall, context recall, claim support, citation precision/completeness, correctness, abstention and coverage. Monitor freshness, online utility, cost, and latency."
        ];
        return `<div class="viz-pipeline">${labels.map((label, i) => `<div class="${i === state.stage ? "current" : ""}"><span>0${i + 1}</span><strong>${label}</strong><small>${conditions[state.scenario][i]}</small></div>`).join("")}</div><div class="viz-explanation" role="status">${descriptions[state.stage]}</div><div class="viz-flow-row">${node("Retrieval scorecard", "Evidence recall / ranking")}${arrow}${node("Generation scorecard", "Support / correctness")}${arrow}${node("Product scorecard", "Coverage / utility / SLOs")}</div>${note("Synthetic example only. Oracle-evidence testing gives the generator the required authorized evidence: success there points toward an upstream retrieval bottleneck. A missing source in the corpus is a different failure from a retriever missing an existing source.")}`;
      }
    }
  };

  const flows = {
    transformers: ["From hidden states to probabilities", [
      ["Token IDs", "Tokenizer output indexes learned embeddings; token boundaries are model-specific."],
      ["Hidden states", "Embeddings and position handling represent tokens before each decoder layer."],
      ["Q/K/V attention", "Queries compare with keys, then softmax weights mix values."],
      ["Residual + MLP", "Residuals preserve a path for information and gradients; the MLP mixes features."],
      ["Vocabulary logits", "A projection maps hidden features into vocabulary-sized logits."],
      ["Next token", "Decode from a probability distribution; fluency is not evidence of truth."]
    ]],
    training: ["One supervised training update", [
      ["Versioned data", "Deduplicate and split without leaking entities, documents, or future information."],
      ["Tokenize + labels", "Align each prediction with its next-token target; preserve supervision masks."],
      ["Forward + loss", "Compute valid-token cross entropy, not an average over padding."],
      ["Backward", "Autograd accumulates gradients; watch finite values and intended trainable parameters."],
      ["Optimizer step", "Clear gradients deliberately, apply clipping/scaling where needed, then update."],
      ["Held-out checks", "Loss, factuality, task metrics, slices, and product outcomes answer different questions."]
    ]],
    pytorch: ["The tensor contract", [
      ["[B,T,d]", "Start from a batch of hidden states with explicit dtype, device, and stride."],
      ["Project Q/K/V", "Linear projections produce the head representations; parameters must be registered."],
      ["[B,H,T,d_head]", "Split width, then transpose token/head axes. Reshape alone can scramble layout semantics."],
      ["SDPA + mask", "For functional SDPA, boolean True means allowed; eval dropout must be explicitly zero."],
      ["Concat + output", "Transpose back, merge heads, and project to model width."],
      ["Test parity", "Check causality, padding, gradients, reference parity, and save/reload behavior."]
    ]],
    sft: ["Teach desired behavior", [
      ["Demonstrations", "Use realistic instructions, evidence, and correct targets, including unanswerable cases."],
      ["Template + masks", "Match serving chat format; supervise only the intended target tokens."],
      ["Base + adapters", "LoRA freezes a base weight and trains low-rank A/B updates; full SFT trains more state."],
      ["Cross entropy", "Optimize token likelihood on demonstrations. This is not preference comparison."],
      ["Ablations", "Hold retrieval and prompts fixed to isolate model adaptation."],
      ["Release gate", "Test task quality, general ability, abstention, latency, memory, and rollback."]
    ]],
    dpo: ["A pairwise preference update", [
      ["Same prompt", "Chosen and rejected responses must answer the same prompt and context."],
      ["Chosen / rejected", "Use a rubric that does not reward style over grounded correctness."],
      ["Policy + reference", "Compute each response's policy-versus-reference log-probability improvement."],
      ["Relative margin", "Subtract the rejected improvement from the chosen improvement."],
      ["-log sigmoid", "Apply beta to the margin; standard DPO needs no separately trained reward model."],
      ["Independent eval", "Preference wins do not guarantee truth. Validate support and correctness independently."]
    ]],
    ppo: ["RLHF rollout, then update", [
      ["SFT starting point", "Start from a behaviorally useful policy and a separate fixed reference."],
      ["On-policy rollout", "Collect responses from the old rollout policy; store action log probabilities."],
      ["Reward + value", "Score outcomes and estimate advantages relative to expected returns."],
      ["Clipped ratio", "Compare current to old rollout probabilities; clipping is not a hard KL guarantee."],
      ["Reference KL", "Control drift from a fixed reference separately from the rollout-policy ratio."],
      ["Task eval", "Check reward hacking, held-out human quality, cost, and operational stability."]
    ]],
    inference: ["Three different optimizations", [
      ["Prefill", "Process prompt positions and populate K/V caches; measure time to first token."],
      ["KV reuse", "Decode avoids re-projecting old K/V but still reads visible cache state."],
      ["Paged storage", "Map logical KV blocks to noncontiguous physical pages to manage allocation."],
      ["Flash kernels", "Tile exact attention to reduce intermediate HBM IO; this is not KV paging."],
      ["Batch + precision", "Continuous batching, quantization, and engine kernels have workload-dependent effects."],
      ["Quality + SLOs", "Compare quality and end-to-end percentiles at representative load, not only kernel speed."]
    ]],
    cuda: ["From host code to completed device work", [
      ["Host dispatch", "The CPU enqueues CUDA work, often asynchronously."],
      ["Device memory", "Weights, activations, KV, optimizer state, and workspaces compete for HBM."],
      ["Tiled kernels", "Threads cooperate using supported on-chip memory and synchronization."],
      ["Compute units", "Supported matrix operations can use Tensor Cores; not every operator benefits."],
      ["Completion event", "Use events or appropriate synchronization to time actual finished GPU work."],
      ["System profile", "Separate compute, memory traffic, transfer, dispatch, and queueing bottlenecks."]
    ]],
    retrieval: ["A relevance funnel", [
      ["Versioned corpus", "Preserve text, metadata, source freshness, and access scope."],
      ["Lexical + dense", "Generate complementary candidates. ANN neighbor recall differs from relevance recall."],
      ["Candidate recall", "A reranker cannot recover an absent relevant document."],
      ["Rerank", "Use richer interactions on fewer candidates under a latency budget."],
      ["Evidence selection", "Deduplicate and select useful, sufficiently complete context."],
      ["Evaluate", "Measure recall, NDCG/MRR as appropriate, query slices, uncertainty, and downstream utility."]
    ]],
    rag: ["Evidence to an evaluated answer", [
      ["Ingest", "Parse, chunk, version, and index sources with access and provenance metadata."],
      ["Retrieve", "Enforce source authorization and measure evidence recall."],
      ["Context", "Rerank, budget, and preserve source-span mappings."],
      ["Answer / abstain", "Ground claims in evidence; missing evidence is not permission to guess."],
      ["Claim checks", "Citations must support claims; faithful-to-stale evidence can still be wrong today."],
      ["Online monitoring", "Track coverage, usefulness, freshness, latency, cost, and incidents together."]
    ]],
    evaluation: ["From a hypothesis to defensible evidence", [
      ["Define decision", "Choose primary outcome, guardrails, unit, population, and minimum useful effect."],
      ["Clean holdout", "Keep tuning out of the final test and control entity/time leakage."],
      ["Offline metrics", "Measure task-appropriate quality with explicit denominators and slices."],
      ["Calibrate judges", "Use human anchors and check order, verbosity, and evidence biases."],
      ["Randomized rollout", "Check assignment, exposure, dependence, sample ratios, and guardrails."],
      ["Effect + uncertainty", "Report absolute and relative effects with uncertainty and limitations."]
    ]],
    agents: ["Autonomy stays inside a contract", [
      ["Goal + budget", "State the user's authorized objective, constraints, and resource limits."],
      ["Choose tool", "The model proposes an action; proposal is not permission."],
      ["External gate", "Validate arguments, authorization, side effects, and remaining budget."],
      ["Execute + observe", "Use typed results/errors. Untrusted tool content is data, not authority."],
      ["Choose again", "Update structured state and choose the next allowed action from observations."],
      ["Verify / stop", "Check actual outcomes, request approval where required, or return incomplete status."]
    ]],
    "agent-eval": ["Evaluate actions, not just fluent conclusions", [
      ["Task fixture", "Specify initial state, authorized tools, expected final state, and validators."],
      ["Fresh environment", "Reset controlled state for every attempt so runs do not contaminate one another."],
      ["Run + trace", "Capture allowed actions, arguments, observations, resource use, and typed errors."],
      ["Inject failures", "Test timeout, stale evidence, injection, duplicate retry, and concurrent edit cases."],
      ["Verify external state", "A final 'done' is not proof; check real state and unauthorized side effects."],
      ["Repeated reliability", "Report per-attempt success, repeated-task consistency, recovery, cost, and latency."]
    ]],
    "system-design": ["A complete interview design", [
      ["Clarify", "Define users, scale, freshness, correctness, permissions, cost, and SLOs."],
      ["Baseline", "Draw a simple system with explicit data and service contracts."],
      ["Deep dive", "Explain one difficult model, data, retrieval, or serving decision."],
      ["Evaluate", "Choose component, end-to-end, online, and failure-suite evidence."],
      ["Operate", "Plan rollout, monitoring, data/version updates, incident handling, and rollback."],
      ["Defend trade-offs", "State assumptions and alternatives; say which observation would change the design."]
    ]],
    spark: ["A Spark execution plan", [
      ["Read + schema", "Define grain, types, nulls, freshness, and source contracts."],
      ["Lazy transforms", "select/filter/join build a logical plan, not Python row-by-row work."],
      ["Action + optimizer", "An action triggers optimized physical execution; inspect explain and the UI."],
      ["Stages / shuffle", "Wide dependencies redistribute data. Check skew, spill, and join cardinality."],
      ["Tasks / partitions", "Executors process partitions; partition count is not an output row-order guarantee."],
      ["Write + validate", "Verify schema, multiset/grain parity, freshness, full job duration, and cost."]
    ]]
  };
  const lessonWidgets = {
    transformers: ["architecture", "attention", "heads"], pytorch: ["heads", "attention"],
    inference: ["kv", "paged", "flash"], cuda: ["gpu"], rag: ["rag"],
    agents: ["agent"], "agent-eval": ["agent", "rag"], "system-design": ["rag", "agent"]
  };
  const atlasTypes = ["architecture", "attention", "heads", "kv", "paged", "flash", "gpu", "rag", "agent"];

  function widgetMarkup(type) {
    const spec = specs[type];
    return `<section id="visual-${type}" class="visual-widget" data-visual="${type}" aria-labelledby="visual-${type}-title"><header><span class="eyebrow">VISUAL EXPLAINER / INTERACTIVE</span><h3 id="visual-${type}-title">${spec.title}</h3><p>${spec.subtitle}</p></header><div class="viz-inputs">${spec.controls(`viz-${type}`)}</div><div class="viz-canvas" data-canvas></div></section>`;
  }
  function flowMarkup(lessonId) {
    const [title, stages] = flows[lessonId];
    return `<section class="viz-lifecycle" data-flow="${lessonId}" aria-label="${title}"><h3>${title}</h3><p>Select a checkpoint to inspect its role and what can go wrong.</p><div class="viz-flow-steps">${stages.map(([label], index) => `<button type="button" data-flow-step="${index}" aria-pressed="${index === 0}"><span>0${index + 1}</span>${label}</button>`).join("")}</div><div class="viz-explanation" data-flow-description role="status">${stages[0][1]}</div></section>`;
  }
  function lessonMarkup(lessonId) {
    const widgets = (lessonWidgets[lessonId] || []).map(widgetMarkup).join("");
    const flow = flowMarkup(lessonId);
    return `<section id="visual-explainer" class="lesson-visuals" aria-label="Visual concept walkthroughs"><div class="section-heading"><div><span class="eyebrow">SEE IT BEFORE YOU DERIVE IT</span><h2>A visual mental model.</h2></div><a href="#visuals">Open visual atlas &rarr;</a></div>${lessonId === "transformers" ? widgets + flow : flow + widgets}</section>`;
  }
  function atlasMarkup() {
    return `<div class="page-heading"><div><span class="eyebrow">SHAPES. MEMORY. DECISIONS.</span><h1>Make the invisible visible.</h1><p class="intro">Nine interactive explainers. Follow the numbers, inspect the memory, and step through the decisions—not just the equations.</p></div></div><div class="viz-atlas-jumps" aria-label="Jump to a visual">${atlasTypes.map(type => `<button type="button" class="filter-chip" data-viz-scroll="${type}">${{ attention: "Q/K/V math", architecture: "Architecture families", heads: "Multi-head", kv: "KV cache", paged: "PagedAttention", flash: "FlashAttention", gpu: "CUDA", rag: "RAG + eval", agent: "Agent loop" }[type]}</button>`).join("")}</div><p class="viz-note">All graphics are local educational simulations. Numerical examples use tiny synthetic inputs. They do not execute Python, run a model or GPU, perform tool calls, or change your saved study progress.</p>${atlasTypes.map(widgetMarkup).join("")}`;
  }
  function updateWidget(root, state) {
    const type = root.dataset.visual;
    const spec = specs[type];
    if (type === "paged") {
      state.address = Math.min(state.address, state.a - 1);
      const address = root.querySelector('[data-control="address"]');
      address.max = state.a - 1;
      address.value = state.address;
      root.querySelector('[data-control="b"]').disabled = state.release;
    }
    let count = spec.steps;
    if (type === "agent") count = state.scenario === "budget" ? 4 : 8;
    if (count) {
      state.step = Math.min(state.step, count - 1);
      root.querySelector('[data-action="previous"]').disabled = state.step === 0;
      root.querySelector('[data-action="next"]').disabled = state.step === count - 1;
      root.querySelector("[data-step-label]").textContent = `${state.step + 1} / ${count}`;
    }
    root.querySelector("[data-canvas]").innerHTML = spec.paint(state, `viz-${type}`);
  }
  function mount(container) {
    container.querySelectorAll("[data-flow]").forEach(root => {
      root.addEventListener("click", event => {
        const button = event.target.closest("[data-flow-step]");
        if (!button) return;
        root.querySelectorAll("[data-flow-step]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
        root.querySelector("[data-flow-description]").textContent = flows[root.dataset.flow][1][Number(button.dataset.flowStep)][1];
      });
    });
    container.querySelectorAll("[data-visual]").forEach(root => {
      const spec = specs[root.dataset.visual];
      const state = spec.initial();
      updateWidget(root, state);
      root.addEventListener("input", event => {
        const controlName = event.target.dataset.control;
        if (!controlName) return;
        const value = event.target.type === "checkbox" ? event.target.checked : event.target.value;
        state[controlName] = typeof state[controlName] === "number" ? Number(value) : value;
        if (controlName === "scenario") state.step = 0;
        updateWidget(root, state);
      });
      root.addEventListener("click", event => {
        const button = event.target.closest("button");
        if (!button) return;
        if (button.dataset.action) {
          state.step += button.dataset.action === "next" ? 1 : -1;
          updateWidget(root, state);
        } else if (button.dataset.scoreRow !== undefined) {
          const restoreFocus = document.activeElement === button;
          state.query = Number(button.dataset.scoreRow);
          state.key = Number(button.dataset.scoreColumn);
          root.querySelector('[data-control="query"]').value = state.query;
          root.querySelector('[data-control="key"]').value = state.key;
          updateWidget(root, state);
          if (restoreFocus) root.querySelector(`[data-score-row="${state.query}"][data-score-column="${state.key}"]`).focus({ preventScroll: true });
        }
      });
    });
    container.querySelectorAll("[data-viz-scroll]").forEach(button => button.addEventListener("click", () => {
      const destination = container.querySelector(`#visual-${button.dataset.vizScroll}`);
      destination.scrollIntoView();
      destination.setAttribute("tabindex", "-1");
      destination.focus({ preventScroll: true });
    }));
  }
  return { lessonMarkup, atlasMarkup, mount, math: { multiply, transpose, softmax, attention, onlineAttention, pageAllocation } };
})();
