# Applied Scientist / MLE Interview Prep

An interactive, browser-based learning guide covering transformers, NumPy, PyTorch, SFT/DPO/PPO, FlashAttention, CUDA, retrieval, end-to-end RAG and evaluation, agents and evaluation, system design, Anthropic FDE and OpenAI AI/ML-engineering preparation, and a Spark add-on.

## Use locally

Run `python -m http.server 8765 --bind 127.0.0.1` in this folder, then open `http://127.0.0.1:8765`. No web dependencies or build step are needed.

## Learn and practice

- 19 chapters, 57 concept questions, and a 38-question final (31/38 passes).
- A dedicated NumPy chapter covering arrays/dtypes, indexing/ownership, broadcasting/axes, statistics and leakage-free preprocessing, local RNGs, stable numerics, least squares/SVD, einsum, regression/clustering/PCA, ufuncs, and memory-aware performance. Topic coverage draws from the full [NumPy preparation repository](https://github.com/rohanmistry231/Numpy-Interview-Preparation); teaching and examples are original. Three array/axis visual explorers and eight NumPy exercises have a separate NumPy-only notebook and direct links such as #numpy-coding/numpy-indexing.
- End-to-end GPT-style training: data/tokenization, pretraining loss and optimizer updates, mixed precision/distribution, checkpoint recovery, SFT/preference branches, evaluation, and release; not an undisclosed proprietary recipe.
- Interactive attention, KV-cache, NDCG, and DPO calculators.
- Beginner-friendly chapter 2: eight numerical decoder-state snapshots, a head-only SGD update, gradients, cross-entropy/logarithms, and an interactive perplexity/geometric-mean lab.
- Visual atlas: architecture, Q/K/V multiplication, masks/softmax, multi-head shapes, KV cache, PagedAttention, FlashAttention online softmax, CUDA, RAG, and agent loops.
- Detailed RAG/agent evaluation with worked metric denominators, ranking and citation grading, risk/coverage, repeated-run reliability, fault injection, calibrated judges, uncertainty, cost/latency, monitoring, and interactive scorecards.
- Clickable lifecycle diagrams in every chapter, with worked toy numbers and explicit assumptions.
- A clickable architecture diagram with 16 detailed transformer component explanations, including positional information, residuals, normalization, feed-forward networks, cross-attention, and the distinction between attention and vocabulary softmax.
- Twelve original PyTorch and four PySpark guided coding exercises with hints and saved answers. The expanded PyTorch topic coverage is informed by the [Devinterview checklist](https://github.com/Devinterview-io/pytorch-interview-questions); its answers are not reproduced. The NumPy bridge exercise requires NumPy.
- Interactive tensor-storage/stride, autograd, and module-registration examples, plus a synchronous GPU-batching timeline. The Visual Atlas has 20 explainers.
- Visual-first chapters: 187 concise section takeaways and 71 topic-specific worked diagrams, with colored arrays/tokens, worked numerical diagrams, and optional full explanations. Use Show all explanations or Visual-first view within a chapter. The NumPy array explorer connects 1-D/2-D/3-D and transposed layouts to coordinates, shape, strides, and an actual byte offset. Diagrams use synthetic teaching fixtures, not a live model or Python runtime.
- Anthropic FDE preparation incorporates topics from [Rung](https://rungcode.io/loops/anthropic) and [Aced](https://www.aced.io/guides/anthropic-forward-deployed-engineer-interview): discovery, MCP, long-context reliability, coding, batching, enterprise rollout, and values.
- OpenAI AI/ML-engineering practice is informed by the [Interview Coder guide](https://www.interviewcoder.co/blog/openai-ml-engineer-interview): role-specific coding, application design, evals, distributed state, recovery, and inference. Third-party interview claims are not official hiring rubrics; all mock exercises are original.
- Question-first company mocks: 24 prompts across six round categories per company, with 48 staged follow-ups, guided six-round practice loops, pause/continue timers, saved answer/code drafts, hidden reference approaches, and question-specific self-scoring rubrics. Open #mock/anthropic or #mock/openai, also linked prominently from both company chapters and Interview Studio. Publicly reported paraphrases, guide samples, and original mock tasks are labeled separately; OpenAI FDE/application reports are not represented as a confirmed ML-engineer loop. Source evidence and its limits are shown on every question. Time budgets/order are practice design, not reported company instructions. Code is not executed and scores are self-assessments.
- Three downloadable notebooks with executable tests and reference solutions.
- Search, notes, progress tracking, score history, and JSON backup export.

Browser coding checks compare supported expressions; they do not execute Python. Run notebook assertions in your own compatible environment. NumPy exercises need only NumPy. PyTorch exercises use CPU. PySpark exercises require compatible PySpark and Java and use a local[2] session.

## Deploy with Netlify Drop

Open https://app.netlify.com/drop, sign in, and drop this folder containing index.html directly at its root. No build command or server backend is required. Do not upload an enclosing development repository. After deployment, verify lessons, coding tracks, notebook links, and the final assessment at the public URL.

## Privacy

This community edition contains generalized synthetic practice scenarios, not a personal profile. No accounts, telemetry, runtime code execution service, or database are used. Each visitor's notes, mock drafts/self-scores, and progress stay in their own browser's localStorage for this origin; they are not synchronized across devices or deployments. Do not enter secrets. External reading links open only when clicked.

## Validation scope

The browser UI and reference-expression matching are tested separately from Python execution. Notebook code and reference solutions are syntax-checked; runtime assertion execution requires the relevant installed frameworks and is not implied by browser checks. All eight NumPy exercises and 17 supported solution combinations passed actual assertions with NumPy 2.5.4 / Python 3.12 in an isolated validation environment. The generated NumPy notebook's solution/test pairs also passed execution. Unchanged PyTorch and Spark runtime assertions were not executed as part of this update.
