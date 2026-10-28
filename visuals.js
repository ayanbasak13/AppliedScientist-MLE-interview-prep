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

  function architectureDiagram(state) {
    const marker = `architecture-${state.family}-arrow`;
    const box = (x, y, label, detail, parts) => `<g class="viz-architecture-part${parts.includes(state.stage) ? " is-current" : ""}" role="button" tabindex="0" data-architecture-node="${x}-${y}" data-architecture-part="${parts[0]}" aria-pressed="${parts.includes(state.stage)}" aria-label="${label}: explain ${TRANSFORMER_STEPS[parts[0]].short}"><rect x="${x}" y="${y}" width="220" height="60" rx="8"/><text x="${x + 110}" y="${y + 25}" text-anchor="middle">${label}</text><text x="${x + 110}" y="${y + 46}" text-anchor="middle" class="viz-svg-small">${detail}</text></g>`;
    const path = (d, residual = false) => `<path d="${d}" fill="none" stroke="${residual ? "#b18449" : "#7d9571"}" stroke-width="2" marker-end="url(#${marker})"/>`;
    const chain = (x, positions) => positions.slice(1).map((y, i) => path(`M${x},${positions[i]} V${y + 60}`)).join("");
    const stack = (x, y, width, height, label) => `<g role="button" tabindex="0" data-architecture-node="stack-${x}" data-architecture-part="11" aria-label="Explain repeated ${label} blocks" aria-pressed="${state.stage === 11}" class="viz-architecture-stack${state.stage === 11 ? " is-current" : ""}"><rect x="${x}" y="${y}" width="${width}" height="${height}" rx="14"/><text x="${x + 12}" y="${y + 20}">${label} block &times; L</text></g>`;
    let drawing;
    if (state.family === "seq2seq") {
      drawing = `${stack(35, 405, 295, 400, "Encoder")}${stack(430, 225, 300, 580, "Decoder")}
        ${chain(175, [1030, 940, 850, 720, 630, 540, 450])}${chain(575, [1030, 940, 850, 720, 630, 540, 450, 360, 270, 170, 80])}
        ${path("M175,450 V375 H400 V570 H465")}<text x="220" y="365" class="viz-architecture-label">Encoder memory supplies K,V</text><text x="590" y="617" class="viz-architecture-label">Q</text>
        ${path("M175,825 H320 V660 H285", true)}${path("M175,615 H310 V480 H285", true)}
        ${path("M575,825 H720 V660 H685", true)}${path("M575,615 H710 V480 H685", true)}${path("M575,435 H720 V300 H685", true)}
        ${box(65, 1030, "Source tokens", "Input IDs [B,S]", [0])}${box(65, 940, "Token embeddings", "[B,S,d_model]", [1])}${box(65, 850, "Add position encoding", "Embedding + position", [2])}
        ${box(65, 720, "Multi-head self-attention", "Bidirectional source attention", [3, 5, 6])}${box(65, 630, "Add &amp; Norm", "Residual, then LayerNorm", [7, 8])}${box(65, 540, "Feed-forward network", "Position-wise nonlinear MLP", [9])}${box(65, 450, "Add &amp; Norm", "Encoder output / memory", [7, 8])}
        ${box(465, 1030, "Shifted target tokens", "Previous target tokens [B,T]", [0, 14, 15])}${box(465, 940, "Target embeddings", "[B,T,d_model]", [1])}${box(465, 850, "Add position encoding", "Embedding + position", [2])}
        ${box(465, 720, "Causal multi-head self-attn", "Q/K/V from target states", [3, 4, 5, 6])}${box(465, 630, "Add &amp; Norm", "Residual, then LayerNorm", [7, 8])}${box(465, 540, "Cross-attention", "Q: decoder / K,V: encoder", [10])}${box(465, 450, "Add &amp; Norm", "Residual, then LayerNorm", [7, 8])}${box(465, 360, "Feed-forward network", "Position-wise nonlinear MLP", [9])}${box(465, 270, "Add &amp; Norm", "Target hidden states", [7, 8])}${box(465, 170, "Vocabulary projection", "Linear head: [B,T,V_vocab]", [12])}${box(465, 80, "Vocabulary softmax", "Distribution over next tokens", [13])}
        <text x="575" y="38" text-anchor="middle" class="viz-architecture-label">TARGET OUTPUT PROBABILITIES</text><text x="175" y="1130" text-anchor="middle" class="viz-architecture-label">ENCODER / SOURCE</text><text x="575" y="1130" text-anchor="middle" class="viz-architecture-label">DECODER / TARGET</text>`;
    } else if (state.family === "decoder") {
      drawing = `${stack(215, 335, 385, 505, "Pre-norm decoder")}${chain(380, [1030, 940, 850, 770, 690, 610, 530, 450, 370, 280, 190, 100])}
        ${path("M380,835 H585 V640 H490", true)}${path("M380,595 H560 V400 H490", true)}
        ${box(270, 1030, "Prompt / previous tokens", "One causal sequence [B,T]", [0, 14, 15])}${box(270, 940, "Token embeddings", "[B,T,d_model]", [1])}${box(270, 850, "Position handling", "RoPE rotates Q/K inside attention", [2])}
        ${box(270, 770, "Normalization", "Before attention (pre-norm)", [8])}${box(270, 690, "Causal multi-head self-attn", "Q/K/V + position + mask", [3, 4, 5, 6])}${box(270, 610, "Residual addition", "x + attention update", [7])}${box(270, 530, "Normalization", "Before the MLP", [8])}${box(270, 450, "Feed-forward network", "Feature mixing / often gated", [9])}${box(270, 370, "Residual addition", "h + MLP update", [7])}
        ${box(270, 280, "Final normalization", "After the complete stack", [8])}${box(270, 190, "Vocabulary projection", "[B,T,d_model] to [B,T,V_vocab]", [12])}${box(270, 100, "Vocabulary softmax", "Next-token distribution", [13])}<text x="380" y="48" text-anchor="middle" class="viz-architecture-label">SELECT TOKEN, APPEND, REPEAT</text>`;
    } else {
      drawing = `${stack(215, 415, 385, 425, "Post-norm encoder")}${chain(380, [1030, 940, 850, 750, 660, 570, 480, 360])}
        ${path("M380,825 H585 V690 H490", true)}${path("M380,645 H560 V510 H490", true)}
        ${box(270, 1030, "Input tokens", "Non-padding source positions", [0])}${box(270, 940, "Token embeddings", "[B,T,d_model]", [1])}${box(270, 850, "Position information", "Illustrative additive positions", [2])}${box(270, 750, "Multi-head self-attention", "Bidirectional; mask padding", [3, 5, 6])}${box(270, 660, "Add &amp; Norm", "Residual, then LayerNorm", [7, 8])}${box(270, 570, "Feed-forward network", "Per-token nonlinear processing", [9])}${box(270, 480, "Add &amp; Norm", "Contextual hidden states", [7, 8])}${box(270, 360, "Task-specific head", "Pooling / tagging / classification", [12])}<text x="380" y="290" text-anchor="middle" class="viz-architecture-label">OUTPUT DEPENDS ON THE TASK</text>`;
    }
    return `<div class="viz-graph-scroll viz-architecture-scroll" tabindex="0" role="group" aria-label="Transformer architecture. Data flows upward. Select a component to explain it; scroll horizontally on small screens."><svg class="viz-architecture-graph" viewBox="0 0 760 1160" role="group" aria-labelledby="architecture-diagram-title"><title id="architecture-diagram-title">${state.family === "seq2seq" ? "Original post-norm encoder–decoder" : state.family === "decoder" ? "Illustrative pre-norm decoder-only" : "Illustrative post-norm encoder-only"} transformer. Active topic: ${TRANSFORMER_STEPS[state.stage].short}.</title><defs><marker id="${marker}" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto-start-reverse"><path d="M0,0 L6,3 L0,6" fill="#809778"/></marker></defs>${drawing}</svg></div>`;
  }
  function architectureExample(stage) {
    if (stage === 0) return `<div class="viz-flow-row">${node("I like ML", "Illustrative token boundaries")}${arrow}${node("[7, 12, 4]", "IDs, not semantic magnitudes")}</div>`;
    if (stage === 1) return `<div class="viz-equation-row">${matrix("Embedding rows E[7], E[12], E[4]", X)}</div>`;
    if (stage === 2) return `<div class="viz-equation-row">${matrix("Token embedding", [[1, 0]])}<span>+</span>${matrix("Position 1", [[Math.sin(1), Math.cos(1)]])}<span>=</span>${matrix("Position-aware input", [[1 + Math.sin(1), Math.cos(1)]])}</div>${note("Additive sinusoidal toy, not RoPE. The full original encoding uses multiple frequencies.")}`;
    if (stage === 3) return `<div class="viz-equation-row">${matrix("Q = XW_Q", multiply(X, weights.Q))}${matrix("K = XW_K", multiply(X, weights.K))}${matrix("V = XW_V", multiply(X, weights.V))}</div>`;
    if (stage === 4) return `<div class="viz-equation-row">${matrix("Already-scaled scores", [[2, 1, 0]])}<span>+</span>${matrix("Causal mask for this row", [[0, 0, -Infinity]])}<span>=</span>${matrix("Masked logits", [[2, 1, -Infinity]])}</div>`;
    if (stage === 5) {
      const probabilities = [softmax([2, 1, -Infinity])], values = [[1, 0], [0, 2], [9, 9]];
      return `<div class="viz-equation-row">${matrix("Attention weights", probabilities)}<span>&times;</span>${matrix("Values", values)}<span>=</span>${matrix("Mixed output", multiply(probabilities, values))}</div>`;
    }
    if (stage === 6) return `<div class="viz-flow-row">${node("Head 1", "[T,4], own attention")}${node("Head 2", "[T,4], own attention")}${arrow}${node("Concatenate", "[T,8]")}${arrow}${node("Output projection W_O", "[T,8]")}</div>`;
    if (stage === 7) return `<div class="viz-equation-row">${matrix("Residual x", [[1, 2]])}<span>+</span>${matrix("Sublayer update", [[0.1, -0.2]])}<span>=</span>${matrix("Updated state", [[1.1, 1.8]])}</div>`;
    if (stage === 8) return `<div class="viz-equation-row">${matrix("Features at one token", [[1, 3]])}<span>&rarr;</span>${matrix("LayerNorm, gamma=1 / beta=0", [[-1 / Math.sqrt(1 + 1e-5), 1 / Math.sqrt(1 + 1e-5)]])}</div>${formula("Mean = 2; population variance = 1; epsilon = 0.00001. Normalization is across features, not tokens.")}`;
    if (stage === 9) {
      const expanded = multiply([[-1, 2]], [[1, 0, 1, 0], [0, 1, 0, -1]]);
      const activated = expanded.map(row => row.map(value => Math.max(0, value)));
      return `<div class="viz-equation-row">${matrix("Expanded xW_1", expanded)}<span>&rarr;</span>${matrix("ReLU", activated)}<span>&rarr;</span>${matrix("Reprojected via W_2", multiply(activated, [[1, 0], [0, 1], [1, 1], [0, -1]]))}</div>`;
    }
    if (stage === 10) return `<div class="viz-flow-row">${node("Decoder Q", "3 target positions")}${node("Encoder K,V", "5 source positions")}${arrow}${node("Scores", "[B,H,3,5], rectangular")}</div>`;
    if (stage === 11) return `<div class="viz-flow-row">${node("Block 1", "Parameters theta_1")}${arrow}${node("Block 2", "Parameters theta_2")}${arrow}${node("Block L", "Parameters theta_L")}</div>`;
    if (stage === 12) return `<div class="viz-equation-row">${matrix("Hidden state", [[1, 0]])}<span>&times;</span>${matrix("Toy W_vocab", [[2, 1, 0], [0, 0, 0]])}<span>=</span>${matrix("Logits: cat / dog / EOS", [[2, 1, 0]])}</div>`;
    if (stage === 13) return `<div class="viz-equation-row">${matrix("Vocabulary logits", [[2, 1, 0]])}<span>&rarr;</span>${matrix("Vocabulary probabilities", [softmax([2, 1, 0])])}</div>${note("Columns are cat, dog, EOS—not context positions. Probabilities are computed from the displayed logits.")}`;
    if (stage === 14) return `<div class="viz-flow-row">${node("Prompt / prefix")}${arrow}${node("All model layers")}${arrow}${node("Next-token choice")}${arrow}${node("Append + repeat")}</div>`;
    return `<div class="viz-flow-row">${node("Training inputs", "BOS | I | like | ML")}${arrow}${node("Shifted labels", "I | like | ML | EOS")}${arrow}${node("Loss + optimizer", "Valid targets only")}</div>`;
  }
  function architectureWalkthrough(state) {
    const step = TRANSFORMER_STEPS[state.stage];
    const missing = (state.stage === 10 && state.family !== "seq2seq") || ([12, 13, 14, 15].includes(state.stage) && state.family === "encoder");
    const scope = missing ? note("This operation is not part of the selected encoder/decoder family as illustrated. Its explanation is a comparison; switch to encoder–decoder to locate cross-attention, or a generative decoder to locate next-token prediction.") : "";
    return `<p class="viz-note">Read the diagram bottom to top. Green marks the selected operation; gold paths are residual bypasses. Click a block, use the topic selector, or use Previous/Next. The original model is post-norm; the decoder-only comparison is pre-norm. All 16 explanations also appear as ordinary chapter sections below. Worked examples are separate tiny fixtures, not a continuous numerical forward pass through a trained model.</p><div class="viz-architecture-layout">${architectureDiagram(state)}<div class="viz-architecture-detail"><h4 data-architecture-heading>Step ${state.stage + 1}: ${step.title}</h4><p class="viz-sr-only" role="status">Selected explanation: ${step.short}</p>${scope}${step.body}${architectureExample(state.stage)}</div></div>`;
  }

  function rate(numerator, denominator) {
    if (!Number.isInteger(numerator) || !Number.isInteger(denominator) || numerator < 0 || denominator < 0 || numerator > denominator) {
      throw new RangeError("A rate requires integer counts with 0 <= numerator <= denominator.");
    }
    return denominator === 0 ? null : numerator / denominator;
  }
  function ragDecisionMetrics({ total, answerable, answeredAnswerable, answeredUnanswerable, correctAnswered }) {
    const unanswerable = total - answerable;
    rate(answerable, total);
    rate(answeredAnswerable, answerable);
    rate(answeredUnanswerable, unanswerable);
    rate(correctAnswered, answeredAnswerable);
    const answered = answeredAnswerable + answeredUnanswerable;
    const warranted = unanswerable - answeredUnanswerable;
    const unnecessary = answerable - answeredAnswerable;
    return {
      total, answerable, unanswerable, answeredAnswerable, answeredUnanswerable, correctAnswered, answered, warranted, unnecessary,
      coverage: rate(answered, total), selectiveAccuracy: rate(correctAnswered, answered),
      decisionAccuracy: rate(correctAnswered + warranted, total),
      warrantedRate: rate(warranted, unanswerable), unnecessaryRate: rate(unnecessary, answerable)
    };
  }
  function repeatedRunMetrics(rows) {
    if (!rows.length || !rows[0].length || rows.some(row => row.length !== rows[0].length || row.some(value => value !== 0 && value !== 1))) {
      throw new RangeError("Repeated-run outcomes require a nonempty rectangular matrix of binary results.");
    }
    const tasks = rows.length, repeats = rows[0].length, attempts = tasks * repeats;
    const successes = sum(rows.map(sum));
    const any = rows.filter(row => row.some(Boolean)).length;
    const all = rows.filter(row => row.every(Boolean)).length;
    return { tasks, repeats, attempts, successes, any, all, perAttempt: rate(successes, attempts), anyRate: rate(any, tasks), allRate: rate(all, tasks) };
  }
  const rateMetric = (label, numerator, denominator, id) => {
    const value = rate(numerator, denominator);
    return metric(`${label} (${numerator} / ${denominator})`, value === null ? "N/A: zero denominator" : `${(value * 100).toFixed(1)}%`, id);
  };

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
      title: "Transformer architecture: explain every block", subtitle: "Start with the original encoder–decoder diagram, then compare encoder-only and modern pre-norm decoder-only models. Select any component for its purpose, math, worked example, and common mistakes.",
      initial: () => ({ family: "seq2seq", stage: 0 }),
      steps: TRANSFORMER_STEPS.length,
      controls: id => `<div class="viz-controls">${control(`${id}-family`, "Architecture family", select(`${id}-family`, "family", [["seq2seq", "Original encoder–decoder (post-norm)"], ["decoder", "Decoder-only example (pre-norm)"], ["encoder", "Encoder-only example (post-norm)"]], "seq2seq"))}${control(`${id}-stage`, "Explain an individual step", select(`${id}-stage`, "stage", TRANSFORMER_STEPS.map((step, index) => [index, `${index + 1}. ${step.short}`]), 0))}</div>${stepButtons(TRANSFORMER_STEPS.length)}`,
      paint: architectureWalkthrough
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
    "rag-scorecard": {
      title: "RAG evaluation: accuracy can rise while usefulness falls",
      subtitle: "Change the answer/abstention policy on the same 20 synthetic requests. Inspect every numerator and denominator, including the zero-answer case.",
      initial: () => ({ scenario: "balanced" }),
      controls: id => control(`${id}-scenario`, "Decision policy", select(`${id}-scenario`, "scenario", [["balanced", "Balanced: answer 10 / 20"], ["cautious", "Cautious: answer 4 / 20"], ["guess", "Guess on all 20"], ["abstain", "Abstain on all 20"]], "balanced")),
      paint: state => {
        const policies = {
          balanced: [9, 1, 8], cautious: [4, 0, 4], guess: [12, 8, 10], abstain: [0, 0, 0]
        };
        const [answeredAnswerable, answeredUnanswerable, correctAnswered] = policies[state.scenario];
        const m = ragDecisionMetrics({ total: 20, answerable: 12, answeredAnswerable, answeredUnanswerable, correctAnswered });
        return `<div class="viz-comparison-scroll"><table class="concept-table"><caption>Fixed gold answerability; only the policy changes</caption><thead><tr><th>Gold case type</th><th>Answered</th><th>Abstained</th><th>Total</th></tr></thead><tbody><tr><td>Answerable</td><td>${m.answeredAnswerable} (${m.correctAnswered} correct)</td><td>${m.unnecessary}</td><td>12</td></tr><tr><td>Unanswerable</td><td>${m.answeredUnanswerable} (incorrect assertions)</td><td>${m.warranted}</td><td>8</td></tr></tbody></table></div><div class="viz-metrics">${rateMetric("Coverage", m.answered, m.total, "rag-coverage")}${rateMetric("Accuracy when answering", m.correctAnswered, m.answered, "rag-selective")}${rateMetric("Warranted abstention", m.warranted, m.unanswerable, "rag-warranted")}${rateMetric("Unnecessary abstention", m.unnecessary, m.answerable, "rag-unnecessary")}${rateMetric("Toy decision accuracy", m.correctAnswered + m.warranted, m.total, "rag-decision")}</div><div class="viz-explanation" role="status">${state.scenario === "cautious" ? "Answered-query accuracy rises to 100%, but coverage falls to 20% and equal-weight decision accuracy falls to 60%. Do not report only the improved conditional metric." : state.scenario === "abstain" ? "No requests are answered, so answered-query accuracy is not applicable—not 100%. All answerable cases are missed, even though abstention is correct on the eight unanswerable cases." : state.scenario === "guess" ? "Coverage is 100%, but the system asserts answers on all eight unanswerable cases. More answers do not automatically mean more value." : "Eight correct answers plus seven warranted abstentions yield 15/20 correct decisions under the stated equal-weight toy policy."}</div>${note("Fixtures assume intentional abstentions and no operational errors. Toy decision accuracy awards one point for a correct answer or warranted abstention; real harm/benefit weights must be defined separately. These are decision metrics, not citation or faithfulness scores. No model or evaluation service is called.")}`;
      }
    },
    "agent-scorecard": {
      title: "Agent evaluation: best-of-three hides inconsistent runs",
      subtitle: "Inspect five synthetic tasks, each attempted three times from reset state. A Pass means independently verified goal, policy, and budget compliance—not merely a confident final answer.",
      initial: () => ({ scenario: "mixed" }),
      controls: id => control(`${id}-scenario`, "Observed repeated-run fixture", select(`${id}-scenario`, "scenario", [["mixed", "Mixed: 8 / 15 strict successes"], ["stable", "Stable on three tasks, fails two"], ["brittle", "Every task passes only once"], ["none", "No successful attempts"]], "mixed")),
      paint: state => {
        const fixtures = {
          mixed: [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 0]],
          stable: [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
          brittle: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
          none: Array.from({ length: 5 }, () => [0, 0, 0])
        };
        const rows = fixtures[state.scenario], m = repeatedRunMetrics(rows);
        return `<div class="viz-comparison-scroll"><table class="concept-table"><caption>Fixed illustrative outcomes, not real agent executions</caption><thead><tr><th>Task</th><th>Attempt 1</th><th>Attempt 2</th><th>Attempt 3</th></tr></thead><tbody>${rows.map((row, i) => `<tr><th>Task ${i + 1}</th>${row.map(value => `<td class="${value ? "viz-run-pass" : "viz-run-fail"}">${value ? "Pass" : "Fail"}</td>`).join("")}</tr>`).join("")}</tbody></table></div><div class="viz-metrics">${rateMetric("Attempt success", m.successes, m.attempts, "agent-attempt")}${rateMetric("Observed any-of-3", m.any, m.tasks, "agent-any")}${rateMetric("Observed all-3", m.all, m.tasks, "agent-all")}${metric("Total spend / strict success", m.successes ? `$18 / ${m.successes} = $${(18 / m.successes).toFixed(2)}` : "N/A: zero successes", "agent-cost")}</div><div class="viz-explanation" role="status">${state.scenario === "brittle" ? "Best-of-three looks perfect: every task succeeds once. Yet only one third of attempts pass and no task passes all three. Retries also spend time/money and require a trustworthy selector." : state.scenario === "none" ? "All attempts remain in the denominator. With no verified successes, cost per success is undefined; do not report $0 or drop the failures." : state.scenario === "stable" ? "Three tasks always pass and two always fail. Per-attempt, any-of-three, and all-three rates are all 60%, revealing a different failure pattern from random inconsistency." : "Eight of fifteen attempts pass. Four tasks pass at least once, but only one passes all three. These are three different evaluation questions."}</div>${note("Total spend is an illustrative fixed $18 across all 15 attempts, including failures. Observed any/all rates are computed directly from task rows; they are not IID probability estimates or the finite-sample pass@k estimator. Tasks can differ and runs can correlate. No real tools are invoked.")}`;
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
    "gpt-training": ["From a corpus to an evaluated GPT-style assistant", [
      ["Data + tokenizer", "Authorize and version the corpus, protect evaluation splits, tokenize, and define packing/boundary policies. Check provenance, contamination, and valid-target counts."],
      ["Initialize decoder", "Choose or load the decoder configuration/checkpoint. Test finite forward/loss/gradients and causal behavior on a tiny batch before scaling."],
      ["Pretrain loop", "Repeat forward logits, shifted/masked token loss, backward, and an optimizer update. Normalize accumulated/distributed losses over the intended valid tokens."],
      ["Validate + checkpoint", "Measure held-out loss and independent task slices. Save weights, optimizer/scheduler/RNG state, data position, versions, and counters; test resumption."],
      ["Post-train", "SFT learns demonstrations. A justified preference branch may use a reward model plus PPO, or direct DPO; these are not universally required consecutive stages."],
      ["Release + monitor", "Gate quality, behavior, safety, serving cost/latency, and export parity. Roll out with rollback. Production conversations do not automatically update model weights."]
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
    transformers: ["architecture"], pytorch: ["heads", "attention"],
    inference: ["kv", "paged", "flash"], cuda: ["gpu"], rag: ["rag", "rag-scorecard"],
    agents: ["agent"], "agent-eval": ["agent-scorecard", "agent", "rag"], "system-design": ["rag", "agent"]
  };
  const atlasTypes = ["architecture", "attention", "heads", "kv", "paged", "flash", "gpu", "rag", "rag-scorecard", "agent", "agent-scorecard"];

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
  function sectionMarkup(lessonId, section) {
    if (lessonId !== "transformers") return "";
    if (section.diagramStep === 3) return widgetMarkup("attention");
    if (section.diagramStep === 6) return widgetMarkup("heads");
    return "";
  }
  function atlasMarkup() {
    return `<div class="page-heading"><div><span class="eyebrow">SHAPES. MEMORY. DECISIONS.</span><h1>Make the invisible visible.</h1><p class="intro">${atlasTypes.length} interactive explainers. Follow the numbers, inspect the memory, and step through the decisions—not just the equations.</p></div></div><div class="viz-atlas-jumps" aria-label="Jump to a visual">${atlasTypes.map(type => `<button type="button" class="filter-chip" data-viz-scroll="${type}">${{ attention: "Q/K/V math", architecture: "Architecture families", heads: "Multi-head", kv: "KV cache", paged: "PagedAttention", flash: "FlashAttention", gpu: "CUDA", rag: "RAG + eval", "rag-scorecard": "RAG scorecard", agent: "Agent loop", "agent-scorecard": "Agent reliability" }[type]}</button>`).join("")}</div><p class="viz-note">All graphics are local educational simulations. Numerical examples use tiny synthetic inputs. They do not execute Python, run a model or GPU, perform tool calls, or change your saved study progress.</p>${atlasTypes.map(widgetMarkup).join("")}`;
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
      const stepKey = type === "architecture" ? "stage" : "step";
      state[stepKey] = Math.max(0, Math.min(state[stepKey], count - 1));
      root.querySelector('[data-action="previous"]').disabled = state[stepKey] === 0;
      root.querySelector('[data-action="next"]').disabled = state[stepKey] === count - 1;
      root.querySelector("[data-step-label]").textContent = `${state[stepKey] + 1} / ${count}`;
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
      function selectArchitecturePart(part) {
        const restoreFocus = document.activeElement === part;
        const key = part.dataset.architectureNode;
        state.stage = Number(part.dataset.architecturePart);
        root.querySelector('[data-control="stage"]').value = state.stage;
        updateWidget(root, state);
        if (restoreFocus) root.querySelector(`[data-architecture-node="${key}"]`).focus();
      }
      root.addEventListener("keydown", event => {
        const part = event.target.closest("[data-architecture-part]");
        if (part && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          selectArchitecturePart(part);
        }
      });
      root.addEventListener("input", event => {
        const controlName = event.target.dataset.control;
        if (!controlName) return;
        const value = event.target.type === "checkbox" ? event.target.checked : event.target.value;
        state[controlName] = typeof state[controlName] === "number" ? Number(value) : value;
        if (controlName === "scenario") state.step = 0;
        updateWidget(root, state);
      });
      root.addEventListener("click", event => {
        const part = event.target.closest("[data-architecture-part]");
        if (part) {
          selectArchitecturePart(part);
          return;
        }
        const button = event.target.closest("button");
        if (!button) return;
        if (button.dataset.action) {
          const stepKey = root.dataset.visual === "architecture" ? "stage" : "step";
          state[stepKey] += button.dataset.action === "next" ? 1 : -1;
          if (stepKey === "stage") root.querySelector('[data-control="stage"]').value = state.stage;
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
  return { lessonMarkup, sectionMarkup, atlasMarkup, mount, math: { multiply, transpose, softmax, attention, onlineAttention, pageAllocation, rate, ragDecisionMetrics, repeatedRunMetrics } };
})();
