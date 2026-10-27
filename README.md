# Applied Scientist / MLE Interview Prep

An interactive, browser-based learning guide covering transformers, PyTorch, SFT/DPO/PPO, FlashAttention, CUDA, retrieval, end-to-end RAG and evaluation, agents and evaluation, system design, and a Spark add-on.

## Use locally

Run `python -m http.server 8765 --bind 127.0.0.1` in this folder, then open `http://127.0.0.1:8765`. No web dependencies or build step are needed.

## Learn and practice

- 16 chapters, 48 concept questions, and a 32-question final (26/32 passes).
- End-to-end GPT-style training: data/tokenization, pretraining loss and optimizer updates, mixed precision/distribution, checkpoint recovery, SFT/preference branches, evaluation, and release; not an undisclosed proprietary recipe.
- Interactive attention, KV-cache, NDCG, and DPO calculators.
- Visual atlas: architecture, Q/K/V multiplication, masks/softmax, multi-head shapes, KV cache, PagedAttention, FlashAttention online softmax, CUDA, RAG, and agent loops.
- Clickable lifecycle diagrams in every chapter, with worked toy numbers and explicit assumptions.
- A clickable architecture diagram with 16 detailed transformer component explanations, including positional information, residuals, normalization, feed-forward networks, cross-attention, and the distinction between attention and vocabulary softmax.
- Six PyTorch and four PySpark guided coding exercises with hints and saved answers.
- Two downloadable notebooks with executable tests and reference solutions.
- Search, notes, progress tracking, score history, and JSON backup export.

Browser coding checks compare supported expressions; they do not execute Python. Run notebook assertions in your own compatible environment. PyTorch exercises use CPU. PySpark exercises require compatible PySpark and Java and use a local[2] session.

## Deploy with Netlify Drop

Open https://app.netlify.com/drop, sign in, and drop this folder containing index.html directly at its root. No build command or server backend is required. Do not upload an enclosing development repository. After deployment, verify lessons, coding tracks, notebook links, and the final assessment at the public URL.

## Privacy

This community edition contains generalized synthetic practice scenarios, not a personal profile. No accounts, telemetry, runtime code execution service, or database are used. Each visitor's notes and progress stay in their own browser's localStorage for this origin; they are not synchronized across devices or deployments. Do not enter secrets. External reading links open only when clicked.

## Validation scope

The browser UI and reference-expression matching are tested separately from Python execution. Notebook code and reference solutions are syntax-checked; runtime assertion execution requires the relevant installed frameworks and is not implied by browser checks.
