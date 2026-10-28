"use strict";

const INTERVIEW_MOCKS = (() => {
  const sources = {
    rungAnthropic: { name: "Rung: Anthropic FDE", url: "https://rungcode.io/loops/anthropic", evidence: "August 2026; one detailed candidate account. Limited corroboration." },
    aced: { name: "Aced: Anthropic FDE", url: "https://www.aced.io/guides/anthropic-forward-deployed-engineer-interview", evidence: "Public guide labels its coding/project examples as candidate-reported; primary accounts are not linked." },
    rungOpenAI: { name: "Rung: OpenAI FDE", url: "https://rungcode.io/loops/openai", evidence: "August 2026; candidate-report/guide compilation. Application/FDE role, not a confirmed ML-engineer loop." },
    coder: { name: "Interview Coder: OpenAI ML engineering", url: "https://www.interviewcoder.co/blog/openai-ml-engineer-interview", evidence: "Third-party preparation guide. Its sample tasks and fixed-loop claims are not independently verified." }
  };
  const companies = {
    anthropic: {
      name: "Anthropic FDE", lesson: "anthropic-fde",
      description: "Customer discovery, staged coding, MCP deployment, systems, project ownership, and judgment.",
      plan: ["a-discovery", "a-lru", "a-mcp", "a-batching", "a-project", "a-values"],
      questions: [
        {
          id: "a-discovery", round: "Customer conversation", title: "Win a skeptical CTO through discovery", minutes: 20,
          kind: "Reported question", source: "rungAnthropic",
          prompt: "A CTO says their current ChatGPT-based workflow already works and sees no reason to adopt Claude. Lead the conversation without assuming a migration is justified.",
          brief: "You are the FDE; a partner plays the CTO. Ask questions before pitching. If alone, state each question and the decision its answer would change.",
          followups: [["The CTO will not share raw data and demands proof in two weeks. What do you propose?", "Use authorized/synthetic or redacted fixtures, a baseline comparison, measurable gates, and explicit limits on what the pilot establishes."], ["Your pilot shows no meaningful advantage. What do you tell the sponsor?", "Recommend against unjustified migration; explain uncertainty, total cost, operational impact, and what new evidence could change the decision."]],
          outline: ["Discover the workflow, failure cost, current metrics, data restrictions, latency, and adoption constraints.", "Design a small blinded comparison against the existing workflow, including abstention, human review, and operating cost.", "Agree on a decision and rollout/rollback gate rather than a blanket model-superiority claim."],
          checks: ["Asks discovery questions before selling", "Compares workflows rather than unsupported brand claims", "Defines a baseline and measurable pilot", "Handles data restrictions and inconclusive evidence", "Communicates a credible next decision"],
          pitfalls: ["Claiming one model is universally better", "Confusing a persuasive demo with measured customer value"]
        },
        {
          id: "a-motivation", round: "Customer conversation", title: "Why this role, not just this company?", minutes: 10,
          kind: "Original mock", source: "aced",
          prompt: "Explain why customer-facing deployment work fits you. Describe a case where an LLM was not the right solution and what evidence shaped that judgment.",
          brief: "Two-minute opening, then challenge your own reasoning. This is original practice based on guide topics, not a reported exact question.",
          followups: [["What would make you decline a customer's proposed AI project?", "State a concrete constraint, viable alternative, and respectful explanation."], ["What assumption about Anthropic would you want to test before joining?", "Offer an informed uncertainty, not flattery or unsupported inside knowledge."]],
          outline: ["Connect actual responsibilities to specific authorized experience.", "Explain a measurable customer outcome and an alternative you considered.", "Name a limitation and a question you would ask the team."],
          checks: ["Understands FDE responsibilities", "Uses specific truthful examples", "Justifies when not to use an LLM", "Names a genuine uncertainty", "Answers concisely without invented impact"],
          pitfalls: ["Substituting enthusiasm for role fit", "Inventing projects or private company facts"]
        },
        {
          id: "a-lru", round: "Practical coding", title: "Incremental LRU cache", minutes: 40,
          kind: "Reported question", source: "aced",
          prompt: "Implement get/put for a capacity-bounded least-recently-used cache. Explain the invariants and test eviction. Then handle the extensions one at a time.",
          brief: "Public report: an LRU cache with added stages. The exact stages below are original mock extensions. Use a language of your choice; the browser saves code but does not execute it.",
          followups: [["Add expiration using an injected clock. What happens when an expired key is read or capacity is checked?", "Specify TTL semantics and lazy cleanup; separate recency order from time order and test boundary equality."], ["Now support multiple concurrent callers. What operations must be atomic?", "Protect compound state updates and define consistency; distinguish a process lock from distributed coordination."]],
          outline: ["Use a key lookup plus an ordered structure; hits/updates move entries to most-recent position.", "Evict the least-recent live entry under a documented capacity/TTL policy.", "Test overwrite, hit ordering, zero/invalid capacity, expiration boundaries, and concurrent access."],
          checks: ["Defines capacity, misses, and expiration contracts", "Maintains map/order invariants correctly", "Tests eviction and staged boundary cases", "Explains time/space and synchronization", "Adds requirements without discarding working behavior"],
          pitfalls: ["Using wall-clock sleeps in deterministic tests", "Claiming O(1) while scanning every entry"]
        },
        {
          id: "a-traces", round: "Practical coding", title: "Turn sampled stacks into execution events", minutes: 35,
          kind: "Reported question", source: "aced",
          prompt: "Given time-ordered sampled call stacks, emit properly nested enter/exit events describing changes between samples.",
          brief: "Original fixture: t=0 [main,parse]; t=5 [main,save]; t=9 []. Expected transitions: enter main/parse at 0, exit parse then enter save at 5, exit save/main at 9. These are sample-boundary approximations, not true call durations.",
          followups: [["A stack is [f,f,g]. How do you handle recursion?", "Compare stack positions and common prefixes, not a set of function names."], ["The final sample is nonempty. How do you close the trace?", "Require an explicit final boundary or document unclosed frames; do not fabricate an observed end time."]],
          outline: ["Validate ordering and define the sampling/trace contract.", "Find each adjacent stack's longest common prefix; exit the old suffix in reverse, then enter the new suffix forward.", "Test recursion, repeated identical stacks, empty input, and the final-boundary policy."],
          checks: ["States what sampling can and cannot reveal", "Emits correct nested transitions", "Tests recursion and empty/unchanged stacks", "Analyzes work relative to total stack depth", "Explains boundary ambiguity clearly"],
          pitfalls: ["Treating samples as exact continuous execution", "Using sets and losing recursion/order"]
        },
        {
          id: "a-dedup", round: "Practical coding", title: "Find duplicate files safely", minutes: 30,
          kind: "Reported question", source: "aced",
          prompt: "Read a collection of files and identify byte-identical duplicate groups. Return a proposed plan; do not delete files.",
          brief: "Original fixture: alpha=[1,2], beta=[1,2], gamma=[2,1], empty1=[], empty2=[]. Groups should be {alpha,beta} and {empty1,empty2}. Assume a supplied read-only file interface.",
          followups: [["There are millions of large files. How do you avoid loading everything?", "Group by size, stream hashes for candidates, and verify bytes before a destructive action."], ["A file changes between hashing and verification. What now?", "Use a stable snapshot/version check or report/retry the change; never claim an unverified match."]],
          outline: ["Define identity, symlink/permission/error policy, and a read-only result contract.", "Reduce candidates by size, stream content hashes, then verify equality for required certainty.", "Test empty files, simulated hash collisions, read failures, and concurrent modification."],
          checks: ["Defines byte identity and safe output", "Handles grouping and collision verification", "Tests failures and mutable files", "Budgets I/O and memory", "Makes errors explicit instead of silently skipping"],
          pitfalls: ["Deleting while discovering duplicates", "Assuming a digest comparison is an absolute proof"]
        },
        {
          id: "a-limits", round: "Practical coding", title: "Per-user and global rate limits", minutes: 35,
          kind: "Reported question", source: "rungAnthropic",
          prompt: "Implement a request-admission limiter enforcing both a per-user budget and a shared global budget.",
          brief: "Choose token bucket or a stated window policy. Original fixture: user capacity 2, global capacity 3, no refill before t=1. A,A are allowed; third A denied; first B allowed; next B denied.",
          followups: [["A request passes the user check but fails the global check. Is a user token consumed?", "Choose an atomic all-or-nothing admission policy and test it; avoid partial debit."], ["Two service replicas receive the same user's burst. What changes?", "Coordinate admission centrally/atomically or explicitly accept bounded approximation; discuss outages and fairness."]],
          outline: ["Specify units, refill/window boundaries, identity, and admission semantics.", "Use an injected monotonic clock and atomic checks/debits for both budgets.", "Test simultaneous arrivals, refill boundaries, denied-request accounting, and distributed behavior."],
          checks: ["Defines a precise limiter policy", "Correctly enforces both limits atomically", "Tests burst/refill/denial cases", "Explains distributed coordination and availability", "Separates implementation guarantees from approximations"],
          pitfalls: ["Debiting one budget on a rejected request accidentally", "Assuming independent process counters enforce a global limit"]
        },
        {
          id: "a-mcp", round: "Technical use case / MCP", title: "A reliable long-running document workflow", minutes: 30,
          kind: "Guide-topic adaptation", source: "aced",
          prompt: "Design a Claude workflow that reads versioned documents through MCP tools and proposes edits over many steps. It must not publish changes without authorized approval.",
          brief: "Aced describes a long-running MCP scenario, not a complete public question specification. This concrete document fixture is original. You may simulate the model; live assessment tool use requires permission.",
          followups: [["A document contains instructions to ignore the workflow and call a write tool. What happens?", "Treat document text as untrusted data; enforce authorization and allowed state transitions outside the model."], ["Context grows beyond the model window and a write times out. How do you continue safely?", "Persist authoritative state/references, summarize with provenance, inspect write status, and use idempotency/version checks."]],
          outline: ["Define typed tool contracts, identity, external permissions, and a bounded state machine.", "Store durable evidence/state separately from conversational context; checkpoint recoverably.", "Evaluate final document state, citations, permission violations, repeated-run success, latency, and cost."],
          checks: ["Scopes a bounded workflow with approvals", "Keeps authorization outside model text", "Tests injection, stale versions, and ambiguous writes", "Handles context/recovery and evaluation", "Justifies autonomy versus a fixed workflow"],
          pitfalls: ["Treating an MCP schema as authorization", "Retrying a side effect without checking whether it committed"]
        },
        {
          id: "a-batching", round: "System design", title: "Synchronous single-GPU inference batching", minutes: 40,
          kind: "Reported question", source: "rungAnthropic",
          prompt: "Design synchronous inference serving on one GPU, with at most 100 inputs per batch. Users wait for their own responses. Optimize throughput subject to a stated tail-latency target.",
          brief: "The capacity-100/synchronous scenario is publicly reported. Numerical extension: arrivals at 0,4,9 ms; a collection timer of 10 ms; service time 20 ms. State exactly when collection starts, dispatch occurs, and responses return.",
          followups: [["The GPU is busy when the collection timer expires. Is latency capped by that timer?", "No. Separate collection wait, GPU queueing, execution, and response work; define the busy-GPU scheduling policy."], ["One request has a very long prompt and several clients disconnect. What changes?", "Use token-aware admission/batching, cancellation semantics, bounded queues, fairness, and overload handling."]],
          outline: ["Clarify workload/token lengths, SLO, deadlines, cancellation, and whether the model is hosted locally.", "For an idle GPU and a timer starting with the first arrival at 0 ms, dispatch all three requests at 10 ms and finish at 30 ms: latencies are 30,26,21 ms. Then specify busy-GPU scheduling, admission, response mapping, and overload.", "Measure end-to-end p50/p95/p99, throughput, utilization, failure rate, and quality under representative traffic."],
          checks: ["Clarifies workload and latency constraints", "Draws a correct queue/dispatch/response lifecycle", "Works a numerical timeline and load test", "Handles capacity, fairness, cancellation, overload", "Separates collection timer from latency guarantee"],
          pitfalls: ["Claiming a timer guarantees total latency", "Assuming batch size alone predicts token/GPU cost"]
        },
        {
          id: "a-queue", round: "System design", title: "Priority jobs, retries, and dead letters", minutes: 35,
          kind: "Reported question", source: "rungAnthropic",
          prompt: "Design a distributed job queue with priorities, retries, and dead-letter handling for customer workflows.",
          brief: "State the delivery guarantee rather than promising exactly-once side effects. Include a worker crash after an external action succeeds.",
          followups: [["A worker completes an action but crashes before acknowledgement. What happens?", "A replay is possible; use idempotency keys, durable action records, and reconciliation."], ["High-priority traffic never stops. How do lower-priority jobs progress?", "Choose and justify aging, reserved capacity, or weighted fair scheduling; measure starvation."]],
          outline: ["Define job identity/state, enqueue durability, leases, acknowledgements, and retryable versus permanent errors.", "Use bounded backoff/jitter, idempotent handlers, attempt budgets, and controlled dead-letter replay.", "Monitor queue age by priority, lease expiry, duplicate effects, retry rate, and recovery outcomes."],
          checks: ["Specifies delivery and priority semantics", "Handles crash/replay without false exactly-once claims", "Tests leases, poison jobs, and replay", "Controls starvation and retry storms", "Explains operational recovery"],
          pitfalls: ["Equating message deduplication with exactly-once external writes", "Replaying dead-letter jobs without fixing the cause"]
        },
        {
          id: "a-project", round: "Project deep dive", title: "Defend a significant deployment", minutes: 25,
          kind: "Reported question", source: "aced",
          prompt: "Choose a significant project. Explain the important technical decisions, why ML or an LLM fit, how adoption developed, and what it took to become production-ready.",
          brief: "Use only information you are authorized to share. Prepare a five-minute architecture and let the follow-ups drive the rest.",
          followups: [["Which measured result can you attribute to your change rather than another team's work?", "Separate ownership, baseline, experimental evidence, uncertainty, and confounders."], ["Scale increases tenfold and adoption stalls. What breaks first?", "Reason from measured bottlenecks and user workflow; propose an experiment rather than a generic scale slogan."]],
          outline: ["Explain the user problem, baseline, architecture, and your own contribution.", "Defend one decision with an alternative, evidence, and a failure you handled.", "Describe rollout/adoption, a limitation, and what you would change."],
          checks: ["Frames customer and baseline", "Explains technically correct decisions and ownership", "Supports results without invented metrics", "Handles adoption/failure/scale trade-offs", "Responds directly to unscripted follow-ups"],
          pitfalls: ["Narrating only the happy path", "Confusing team impact with personal ownership"]
        },
        {
          id: "a-values", round: "Values / judgment", title: "An unsafe launch under executive pressure", minutes: 20,
          kind: "Original mock", source: "rungAnthropic",
          prompt: "A sponsor wants to launch tomorrow. Evaluation found a rare but consequential unauthorized-action failure. Lead a decision conversation with the sponsor and your team.",
          brief: "An original scenario based on reported velocity/reliability/safety topics. There is no prescribed company slogan or guaranteed correct hiring answer.",
          followups: [["The sponsor says the expected revenue outweighs the risk. What do you do?", "Clarify authority, affected parties, severity/uncertainty, mitigations, and escalation; propose a bounded alternative."], ["You discover your earlier design contributed to the failure. How does your response change?", "Own the mistake, preserve evidence, fix the failure, and communicate without deflecting blame."]],
          outline: ["Establish facts, affected users, uncertainty, and decision authority.", "Compare delaying, removing the risky action, human approval, or a restricted pilot with explicit gates.", "Document/escalate the decision and monitor/rollback; explain how you update your own view."],
          checks: ["Identifies concrete harm and uncertainty", "Reasons beyond a memorized slogan", "Requests evidence and validates mitigation", "Handles pressure and ownership constructively", "States an actionable decision and escalation path"],
          pitfalls: ["Treating a business sponsor as authorization to ignore policy", "Claiming zero risk or hiding a known failure"]
        },
        {
          id: "a-feedback", round: "Values / judgment", title: "Feedback that changed your technical judgment", minutes: 15,
          kind: "Original mock", source: "aced",
          prompt: "Describe difficult feedback or a technical disagreement that changed your mind. Explain what you felt, what evidence mattered, and what you did differently.",
          brief: "Original practice based on Aced's feedback/self-awareness topics. Use an authentic example rather than a fabricated interview story.",
          followups: [["What part of the criticism was fair, and what part was not?", "Separate evidence and personal reaction; show a nuanced response rather than total surrender or defensiveness."], ["How would a teammate describe your behavior afterward?", "Offer observable changes and a way to verify improvement."]],
          outline: ["Give concise context and your initial belief.", "Explain the challenge, emotional response, evidence, and changed action.", "Name a lasting behavior change and an unresolved limitation."],
          checks: ["Uses a specific truthful event", "Explains the technical disagreement", "Identifies evidence that changed the decision", "Shows constructive emotional/ownership awareness", "Reflects on observable improvement"],
          pitfalls: ["A disguised boast with no real mistake", "Blaming the other person instead of examining your judgment"]
        }
      ]
    },
    openai: {
      name: "OpenAI AI / ML engineering", lesson: "openai-aie",
      description: "Application/FDE-style reported questions, plus separately labeled ML-coding and infrastructure guide practice.",
      plan: ["o-fit", "o-attention", "o-rag", "o-eval", "o-project", "o-judgment"],
      questions: [
        {
          id: "o-fit", round: "Role / production experience", title: "Why deployment engineering?", minutes: 10,
          kind: "Reported question", source: "rungOpenAI",
          prompt: "Why do you want a forward-deployed/application-engineering role rather than simply a job at OpenAI? Describe the messy parts of an AI system you operated in production.",
          brief: "Reported for OpenAI FDE. Use it as adjacent application-engineering practice, not evidence that every OpenAI ML-engineer role has this screen.",
          followups: [["What did you own beyond calling the model API?", "Explain data contracts, integration, evaluation, monitoring, incidents, and adoption that you actually owned."], ["Which role expectation would you clarify with the recruiter?", "Distinguish customer deployment from research/training infrastructure and confirm the assessment scope."]],
          outline: ["Connect the role's work to an honest example.", "Describe an operational problem, your decision, evidence, and customer result.", "State which technical areas you want to deepen and which role assumptions need confirmation."],
          checks: ["Distinguishes role fit from company enthusiasm", "Explains real operational ownership", "Provides evidence without invented scale", "Understands application versus ML-systems boundaries", "Gives a concise credible motivation"],
          pitfalls: ["Using ChatGPT as the only evidence of deployment skill", "Claiming the FDE report defines all OpenAI interviews"]
        },
        {
          id: "o-attention", round: "ML coding / NumPy", title: "Stable masked attention from scratch", minutes: 40,
          kind: "Guide sample, not verified report", source: "coder",
          prompt: "Implement scaled dot-product attention in NumPy for rectangular query/key lengths. Apply a mask before stable softmax, return the value mixture, and write tests.",
          brief: "Original fixture: q=[1,0]; keys [[1,0],[0,1]]; values [[2,0],[0,4]]; only key 0 allowed. Output must be [2,0]. Choose an explicit all-masked-row policy. Browser code is not executed.",
          followups: [["Why does multiplying scores by a zero/one mask fail?", "A forbidden zero score still contributes exp(0); use a correct exclusion operation before normalization."], ["Extend to [B,H,T,D] queries and [B,H,U,D] keys. What grows quadratically?", "Preserve batch/head axes, sum the feature axis, and account for [B,H,T,U] score storage."]],
          outline: ["Validate shapes, finite values, mask convention, and all-masked behavior.", "Compute QK transpose / sqrt(D), exclude forbidden logits, normalize stably, and multiply by V.", "Test nonsquare lengths, huge logits, row sums, zero forbidden mass, and perturbing a masked value."],
          checks: ["States shapes and mask contract", "Computes correct stable attention", "Tests causality/masking and failure cases", "Analyzes score memory and heads", "Explains the derivation while coding"],
          pitfalls: ["Multiplying logits by a mask", "A test that only checks shape or a square coincidence"]
        },
        {
          id: "o-kmeans", round: "ML coding / NumPy", title: "K-means with explicit edge policies", minutes: 35,
          kind: "Guide sample, not verified report", source: "coder",
          prompt: "Implement k-means using NumPy distances, assignments, and center updates. Define convergence, initialization, ties, and empty-cluster behavior.",
          brief: "Original fixture: points [0,2,8,10], initial centers [0,10]. After one update centers are [1,9]. Return final labels consistent with the final centers.",
          followups: [["A center receives no points. What happens?", "Choose and test a documented policy, such as retaining or reinitializing the center; do not silently produce NaNs."], ["The [N,K,D] distance temporary is too large. What changes?", "Chunk computation or use a numerically checked matrix identity; describe precision and memory trade-offs."]],
          outline: ["Validate k, data, initialization, iteration cap, and convergence policy.", "Compute squared distances, assign nearest centers, and update under the empty-cluster policy.", "Recompute final labels and test duplicates, ties, empty clusters, and deterministic seeds."],
          checks: ["Defines algorithm and edge contracts", "Updates assignments/centers consistently", "Tests empty clusters and convergence", "Budgets distance memory and numerical error", "Explains initialization/local-optimum limitations"],
          pitfalls: ["Returning stale labels", "Calling one run's solution a guaranteed global optimum"]
        },
        {
          id: "o-conv", round: "ML coding / NumPy", title: "A convolution reference with shape tests", minutes: 35,
          kind: "Guide sample, not verified report", source: "coder",
          prompt: "Write a NumPy 2-D convolution-style reference, then add stride and padding. State whether the operation is convolution or cross-correlation.",
          brief: "Original fixture: image [[1,2,3],[4,5,6],[7,8,9]], all-ones 2x2 filter, stride 1, no padding. Output is [[12,16],[24,28]]. Use a nonsymmetric filter to test flipping.",
          followups: [["How do dilation and multiple channels change indexing and output shape?", "Derive effective kernel size and sum over input channels with an explicit axis convention."], ["Can a sliding-window view make this memory-free?", "A view may avoid one copy, but later products/reshapes can allocate; analyze the full working set."]],
          outline: ["Define axis layout, kernel orientation, output formula, and valid parameter ranges.", "Start with an auditable loop reference and a known-value fixture.", "Add channel/stride/padding/dilation tests before comparing optimized implementations."],
          checks: ["Defines convolution/cross-correlation and layout", "Computes correct windows and output shape", "Tests nonsymmetric kernels and boundaries", "Explains operation count and temporary memory", "Optimizes only after validating a reference"],
          pitfalls: ["Testing flipping with only a symmetric kernel", "Confusing a window view with a free downstream computation"]
        },
        {
          id: "o-rag", round: "Application / system design", title: "Defend every stage of a RAG service", minutes: 40,
          kind: "Reported question", source: "rungOpenAI",
          prompt: "Design a production RAG system and defend embedding selection, chunking, retrieval, reranking, and context assembly rather than naming a vector database.",
          brief: "Original workload: 100,000 versioned internal documents, per-user permissions, daily updates, and answers with citations. Clarify traffic and SLOs before sizing.",
          followups: [["Retrieval recall improves but answer correctness drops. Diagnose it.", "Check ranking/context truncation, distractors, source freshness, generation behavior, and judge validity with component/oracle ablations."], ["A document's permissions change after indexing. How do you prevent disclosure?", "Enforce current authorization at retrieval/serving boundaries and invalidate stale caches; do not rely solely on embedding-time tags."]],
          outline: ["Specify source/version/permission and query contracts, ingestion, retrieval, and serving.", "Compare chunk/embedding/hybrid/reranking choices on fixed judged fixtures and failure slices.", "Evaluate evidence recall, answer correctness/grounding, abstention, latency/cost, and guarded rollout."],
          checks: ["Clarifies requirements and permissions", "Connects retrieval/context/generation contracts", "Defines component and end-to-end evaluations", "Handles freshness, cache, failures, and cost", "Defends alternatives with evidence"],
          pitfalls: ["Calling retrieval recall the answer-quality metric", "Treating a citation as proof the claim is supported"]
        },
        {
          id: "o-latency", round: "Application / system design", title: "Find latency across the whole LLM stack", minutes: 30,
          kind: "Reported question", source: "rungOpenAI",
          prompt: "An LLM application's p95 latency doubled. Walk the request path and isolate the cause before proposing caching or a faster model.",
          brief: "Original observation: provider time is unchanged, retrieval p95 rose from 80 to 700 ms, and retry count increased. These observations are clues, not a proof of one root cause.",
          followups: [["A cache improves median latency but p99 gets worse. Why might that happen?", "Inspect misses, stampedes, eviction, stale/permission-safe keys, and queueing; compare hit/miss slices."], ["Clients retry aggressively on rate limits. What do you change?", "Coordinate request/token budgets, honor retry guidance, use bounded jittered backoff and admission control, and avoid duplicated side effects."]],
          outline: ["Add correlated spans for queueing, retrieval, reranking, provider prefill/decode, and response work.", "Compare representative distributions and traffic/token/error slices before changing the system.", "Test the targeted mitigation under load with quality, tail-latency, availability, and cost gates."],
          checks: ["Frames latency and workload changes", "Uses end-to-end traces instead of guessing", "Validates hypotheses with sliced metrics", "Handles retry/cache/queue interactions", "Proposes a measurable reversible mitigation"],
          pitfalls: ["Adding a cache before finding the bottleneck", "Reporting average latency for a tail-latency regression"]
        },
        {
          id: "o-training", round: "Application / system design", title: "Training-state memory and checkpoint recovery", minutes: 40,
          kind: "Guide-topic adaptation", source: "coder",
          prompt: "Design a training run that exceeds one GPU's memory. Inventory persistent and transient state, choose a parallelization strategy, and explain recovery after a worker failure.",
          brief: "ML-systems practice, not an asserted AI-application interview requirement. Original arithmetic: for P parameters, use 2P weight bytes + 2P gradient bytes + 8P Adam moments + 4P master-weight bytes before activations/buffers, under this explicit toy dtype setup.",
          followups: [["Why does sharding weights alone not guarantee the run fits?", "Account for gradients, optimizer/master state, activations, transient gathers, and communication."], ["A node fails after a checkpoint. What must be restored beyond weights?", "Optimizer/scheduler, data position, RNG and precision/scaler state as applicable, plus compatible topology/version metadata."]],
          outline: ["State workload, dtypes, devices, units, and a complete memory budget.", "Choose measured remedies such as accumulation, recomputation, optimizer/parameter sharding, or model parallelism.", "Define atomic/versioned checkpoints, restart tests, lost-work budget, and throughput/quality parity."],
          checks: ["States memory assumptions explicitly", "Accounts for all relevant training state", "Tests correctness and restart behavior", "Justifies parallelism/communication trade-offs", "Separates estimates from measured peaks"],
          pitfalls: ["Treating weight bytes as total training memory", "Calling a weight-only save a reproducible training checkpoint"]
        },
        {
          id: "o-eval", round: "Evaluation / technical deep dive", title: "How do you know the AI system works?", minutes: 35,
          kind: "Reported question", source: "rungOpenAI",
          prompt: "You shipped an AI assistant. Explain how you establish that it actually solves the intended task and remains reliable after changes.",
          brief: "Original fixture: 100 held-out tasks; 80 attempted; 64 verified correct. Compute coverage, attempted-answer accuracy, and overall success before choosing a release decision. Respect the different denominators.",
          followups: [["A model judge says the new version is better, but users disagree. What do you investigate?", "Calibrate against blinded human/outcome labels, inspect slices and judge bias/leakage, and separate offline proxies from user outcomes."], ["It is an agent with external side effects. Is final-answer grading enough?", "Grade verified final state, authorization/side effects, trajectory failures, repeated-run reliability, and cost/latency."]],
          outline: ["Define the task and failure cost, then version a representative held-out evaluation with explicit labels.", "Coverage=80%; attempted accuracy=64/80=80%; overall verified success=64/100=64%; add slice/uncertainty and error analysis.", "Compare paired baselines, run ablations/adversarial tests, set release gates, and monitor online outcomes."],
          checks: ["Defines the real task and failure taxonomy", "Uses correct metrics and denominators", "Validates labels/judges and uncertainty", "Connects offline, agent, and online reliability", "States evidence-based release/rollback gates"],
          pitfalls: ["Reporting 80% without naming the denominator", "Treating judge preference as verified task completion"]
        },
        {
          id: "o-method", round: "Evaluation / technical deep dive", title: "Prompting, RAG, or fine-tuning?", minutes: 25,
          kind: "Reported question", source: "rungOpenAI",
          prompt: "A customer wants higher-quality answers over frequently changing private documents. When would you change the prompt, add RAG, fine-tune, or avoid an LLM?",
          brief: "Require a baseline and a failure diagnosis. The alternatives can be combined; they are not mutually exclusive marketing categories.",
          followups: [["The answers are grounded but the output format fails 8% of the time. What next?", "Test schema-constrained output and validation/repair before assuming factual retrieval or fine-tuning is the cause."], ["The customer says a system prompt is enough authorization. What guardrails do you require?", "Enforce permissions, tool approval, data isolation, output validation, and side-effect controls outside the model."]],
          outline: ["Separate missing/current knowledge, instruction/behavior failures, task fit, and policy requirements.", "Compare the simplest plausible fixes on the same held-out task set, with cost/latency and maintenance.", "Choose a justified combination and keep production guardrails independent of the model's willingness to comply."],
          checks: ["Diagnoses failure before choosing a method", "Explains each method's appropriate role", "Designs comparable held-out experiments", "Handles freshness, privacy, and guardrails", "Defends a simple reversible decision"],
          pitfalls: ["Using fine-tuning as a database for daily-changing facts", "Treating prompting as an access-control boundary"]
        },
        {
          id: "o-build", round: "Project / build-and-defend", title: "Build a bounded assistant and evaluation harness", minutes: 60,
          kind: "Reported task pattern; original specification", source: "rungOpenAI",
          prompt: "Build a small document-answering service plus a repeatable evaluation command. Present working tests and defend the design.",
          brief: "Public reports describe API-based RAG/agent/eval projects, not this exact task. Original 60-minute mock: three synthetic documents with IDs/versions/permissions, six fixed questions, a model adapter that may be stubbed, citations or abstention, and an evaluation JSON report. No account, paid API, or secret is required.",
          followups: [["Inject a provider timeout and a malformed model response. Demonstrate behavior.", "Expose explicit errors/abstention, bounded retries, schema validation, and tests rather than success-shaped defaults."], ["A reviewer asks why this needs an agent. Defend or simplify it.", "Compare a fixed retrieval workflow; introduce autonomy only when task requirements justify its extra failure surface."]],
          outline: ["Allocate 10 minutes to interfaces/fixtures, 30 to a minimal working path, 15 to tests/evaluation, and 5 to review.", "Separate retrieval/permissions/model adapter/policy; make the evaluation runnable without private data.", "Deliver test results and a report naming correctness, grounding, abstention/coverage, failures, and limitations."],
          checks: ["Scopes an achievable artifact and data contract", "Builds a working bounded path", "Includes reproducible tests and evaluation output", "Handles timeout/validation/permissions", "Defends choices and limitations honestly"],
          pitfalls: ["A polished demo without a runnable evaluation", "Treating the mock duration as a confirmed company take-home rule"]
        },
        {
          id: "o-project", round: "Project / build-and-defend", title: "A complex system under rapid follow-ups", minutes: 30,
          kind: "Reported question", source: "rungOpenAI",
          prompt: "Present a complex system you built and defend the architecture, alternatives, operational failures, and scaling limits under rapid follow-ups.",
          brief: "Five-minute opening only. Use actual authorized experience and explicitly distinguish implemented scale from a proposed tenfold-scale extension.",
          followups: [["Which part would you remove if the budget were cut in half?", "Reason from user value, bottlenecks, quality gates, and operating cost rather than defending every component."], ["What measurement most strongly challenged your original design?", "Explain contradictory evidence and the concrete change it caused."]],
          outline: ["Draw the data/control path and name your actual ownership.", "Defend one important trade-off using a baseline and observed evidence.", "Discuss an incident, recovery, scaling estimate, and a decision you would revisit."],
          checks: ["Frames requirements and ownership", "Explains architecture at useful depth", "Uses measurements and counter-evidence", "Handles failure, scale, and cost trade-offs", "Answers the follow-up instead of repeating a pitch"],
          pitfalls: ["Inventing high-scale production experience", "Describing tools without explaining decisions"]
        },
        {
          id: "o-judgment", round: "Collaboration / judgment", title: "Ship pressure versus contradictory evaluation", minutes: 20,
          kind: "Original mock", source: "rungOpenAI",
          prompt: "A partner wants to launch an AI workflow after a convincing demo. Your held-out evaluation shows failures on an important customer slice. Resolve the disagreement and propose a next step.",
          brief: "Original application-engineering judgment scenario, not a claimed actual company question. Do not invent OpenAI's internal decision policy.",
          followups: [["A teammate disputes your labels and says your evaluation is biased. What do you do?", "Audit labels, sample selection, blind review, and slice definitions; update your view if the criticism is supported."], ["The customer cannot wait for the full fix. Is there a bounded alternative?", "Discuss removing risky capabilities, human review, limited scope, explicit user expectations, monitoring, and rollback."]],
          outline: ["Establish the disputed evidence and impact without personalizing the disagreement.", "Propose a fast reproducible test and a risk-aware decision owner/process.", "Communicate limitations clearly and define a guarded alternative or delay."],
          checks: ["Understands the affected customer task", "Reasons consistently about uncertainty", "Investigates contrary evidence fairly", "Offers a bounded operational decision", "Communicates constructively and owns mistakes"],
          pitfalls: ["Hiding bad slices in an aggregate metric", "Treating disagreement as a persuasion contest"]
        }
      ]
    }
  };
  const escape = value => String(value).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const allQuestions = Object.values(companies).flatMap(company => company.questions);
  const validDrafts = drafts => drafts !== null && typeof drafts === "object" && !Array.isArray(drafts) &&
    Object.entries(drafts).every(([id, draft]) => allQuestions.some(q => q.id === id) && draft !== null && typeof draft === "object" &&
      typeof draft.answer === "string" && Number.isInteger(draft.phase) && draft.phase >= 0 && draft.phase <= 3 &&
      Number.isInteger(draft.elapsed) && draft.elapsed >= 0 && draft.elapsed <= 86400 &&
      Array.isArray(draft.scores) && draft.scores.length === 5 && draft.scores.every(n => n === null || Number.isInteger(n) && n >= 0 && n <= 2));
  function validRoute(company, question, mode) {
    return Object.hasOwn(companies, company) && (!question || companies[company].questions.some(q => q.id === question)) &&
      (!mode || ["practice", "loop"].includes(mode)) && (mode !== "loop" || !question || companies[company].plan.includes(question));
  }
  function mount(container, companyId, requested, requestedMode, drafts, save) {
    const company = companies[companyId];
    let questionId = requested || company.questions[0].id;
    let mode = requestedMode || "practice";
    let started = null;
    let interval = null;
    let disposed = false;
    const question = () => company.questions.find(q => q.id === questionId);
    const draft = () => drafts[questionId] || (drafts[questionId] = { answer: "", phase: 0, elapsed: 0, scores: [null, null, null, null, null] });
    const elapsed = () => Math.min(86400, draft().elapsed + (started === null ? 0 : Math.max(0, Math.floor((Date.now() - started) / 1000))));
    const clock = seconds => `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`;
    function persist() {
      const saved = save();
      const status = container.querySelector("#mock-save-status");
      if (status) status.textContent = saved ? "Draft and self-scores saved on this browser" : "Not saved: export study data before leaving.";
    }
    function pause() {
      if (started !== null) {
        draft().elapsed = elapsed();
        started = null;
        clearInterval(interval);
        interval = null;
        persist();
      }
    }
    function updateClock() {
      if (disposed) return;
      const remaining = Math.max(0, question().minutes * 60 - elapsed());
      container.querySelector("#mock-clock").textContent = clock(remaining);
      container.querySelector("#mock-timer").textContent = started === null ? "Start / continue timer" : "Pause timer";
      container.querySelector("#mock-timer").disabled = remaining === 0;
      container.querySelector("#mock-time-status").textContent = remaining === 0 ? "Time is up. You can still finish, reveal the rubric, and review." : started === null ? "Paused. Leaving this page pauses the clock; it does not erase your draft." : "Running. Rehearse aloud or write below.";
      if (remaining === 0) pause();
    }
    function paint() {
      const q = question(), d = draft(), source = sources[q.source];
      const sequence = mode === "loop" ? company.plan : company.questions.map(item => item.id);
      const index = sequence.indexOf(q.id);
      const total = d.scores.reduce((a, n) => a + (n || 0), 0);
      const scored = d.scores.every(n => n !== null);
      const completed = company.plan.filter(id => drafts[id]?.scores.every(n => n !== null)).length;
      history.replaceState(null, "", `#mock/${companyId}/${q.id}/${mode}`);
      container.innerHTML = `<div class="page-heading"><div><span class="eyebrow">QUESTION-FIRST / PRACTICE, NOT A GUARANTEED HIRING LOOP</span><h1>${company.name}: mock interview</h1><p class="intro">${company.description}</p></div></div>
        <div class="filter-bar">${Object.entries(companies).map(([id, c]) => `<a href="#mock/${id}" class="filter-chip ${id === companyId ? "active" : ""}"${id === companyId ? ' aria-current="page"' : ""}>${c.name}</a>`).join("")}</div>
        <div class="coding-notice"><strong>${company.questions.length} questions. Six sections. Practice one round or the full mock.</strong><p>Answer aloud or draft below. Reveal interviewer follow-ups before the rubric, then self-score. Round order and time budgets are our practice design.</p><details><summary>Sources, role differences, and limits</summary><p>Reported questions are paraphrases from public third-party accounts, not company-confirmed prompts. Guide samples and original extensions are labeled separately. OpenAI FDE/application reports are not automatically ML-engineer reports. Sources can disagree: confirm your role, rounds, and tool rules with the recruiter. No restricted assignments or source answers are reproduced. The guided loop selects six representative prompts. Answers stay on this browser; no model grades or runs your code.</p></details><button class="button secondary small-button" type="button" data-mock-action="loop">Run guided six-round mock</button> <a class="button text small-button" href="#lesson/${company.lesson}">Optional background reading &rarr;</a></div>
        <div class="mock-selectors"><div class="field"><label for="mock-round">INTERVIEW SECTION</label><select id="mock-round">${[...new Set(company.questions.map(item => item.round))].map(round => `<option${round === q.round ? " selected" : ""}>${escape(round)}</option>`).join("")}</select></div><div class="field"><label for="mock-question">QUESTION BANK</label><select id="mock-question">${company.questions.map(item => `<option value="${item.id}"${q.id === item.id ? " selected" : ""}>${escape(item.round)} / ${escape(item.title)}</option>`).join("")}</select></div></div>
        <div class="mock-meta"><span class="tag">${escape(q.kind)}</span><span>${mode === "loop" ? `Guided round ${index + 1} / 6` : `Bank question ${index + 1} / ${sequence.length}`} &middot; ${q.minutes}-minute practice budget</span><span id="mock-progress">${completed}/6 guided questions self-scored</span></div>
        <div class="mock-layout"><section class="panel"><span class="eyebrow">${escape(q.round)}</span><h2>${escape(q.title)}</h2><p class="prompt">${escape(q.prompt)}</p><p>${escape(q.brief)}</p><details class="mock-source"><summary>Question source and evidence</summary><p><a href="${source.url}" target="_blank" rel="noopener noreferrer">${source.name} &rarr;</a><br>${escape(source.evidence)}<br>Question wording, fixtures, follow-ups, reference approach, and rubric here are adaptations/original practice, not a verbatim transcript.</p></details>
        ${q.followups.slice(0, Math.min(d.phase, 2)).map(([prompt], i) => `<section class="mock-followup"><span class="eyebrow">INTERVIEWER FOLLOW-UP ${i + 1}</span><p>${escape(prompt)}</p></section>`).join("")}
        <label for="mock-answer" class="eyebrow">YOUR ANSWER / CODE DRAFT (OR REHEARSE ALOUD)</label><textarea id="mock-answer" spellcheck="false" placeholder="Clarifying questions, approach, implementation or diagram notes, tests, and trade-offs...">${escape(d.answer)}</textarea><p id="mock-save-status" class="save-status" role="status">Saved drafts stay on this browser. Do not enter confidential interview material.</p>
        <div class="quiz-controls"><button class="button" type="button" data-mock-action="advance"${d.phase === 3 ? " disabled" : ""}>${d.phase < 2 ? `Reveal follow-up ${d.phase + 1}` : d.phase === 2 ? "Finish round and reveal rubric" : "Rubric revealed"}</button>${d.phase < 2 ? '<button class="button secondary" type="button" data-mock-action="review">Finish early / reveal rubric</button>' : ""}</div></section>
        <aside class="panel mock-clock-panel" aria-label="Round timer and preparation"><span class="eyebrow">ROUND CLOCK</span><output id="mock-clock" aria-label="Time remaining"></output><div class="quiz-controls"><button class="button secondary small-button" type="button" id="mock-timer" data-mock-action="timer"></button><button class="button text small-button" type="button" data-mock-action="reset-clock">Reset timer only</button></div><p id="mock-time-status" role="status"></p><h3>Before answering</h3><ol><li>Clarify contracts and success.</li><li>Think aloud or draft an implementation.</li><li>Test a small example and a failure.</li><li>Respond to the next follow-up before opening the rubric.</li></ol><p>There is no automatic correctness score. A partner can play the interviewer; self-scores measure your assessment, not hiring readiness.</p></aside></div>
        ${d.phase === 3 ? `<section class="panel mock-review" id="mock-review" tabindex="-1"><span class="eyebrow">REVIEW / ORIGINAL REFERENCE APPROACH</span><h2>Compare your reasoning, then score it.</h2><ol>${q.outline.map(line => `<li>${escape(line)}</li>`).join("")}</ol><details><summary>What the follow-ups are testing</summary><ul>${q.followups.map(([prompt, expected]) => `<li><strong>${escape(prompt)}</strong><br>${escape(expected)}</li>`).join("")}</ul></details><h3>Common failure modes</h3><ul>${q.pitfalls.map(line => `<li>${escape(line)}</li>`).join("")}</ul><p>For each dimension: <strong>0</strong> missing/incorrect, <strong>1</strong> partial or unsupported, <strong>2</strong> correct, specific, and demonstrated. Complete all five for a score; a total is not an interview pass prediction.</p><div class="mock-score-grid">${q.checks.map((check, i) => `<div class="field"><label for="mock-score-${i}">${i + 1}. ${escape(check)}</label><select id="mock-score-${i}" data-mock-score="${i}"><option value=""${d.scores[i] === null ? " selected" : ""}>Not scored</option>${[0, 1, 2].map(n => `<option value="${n}"${d.scores[i] === n ? " selected" : ""}>${n} / 2</option>`).join("")}</select></div>`).join("")}</div><p class="mock-score-result" id="mock-score-result" role="status">${scored ? `Self-score: ${total} / 10. Review the weakest dimensions before another attempt.` : "Self-score incomplete: score all five dimensions."}</p></section>` : ""}
        <div class="lesson-actions"><button class="button secondary" type="button" data-mock-action="previous"${index <= 0 ? " disabled" : ""}>&larr; Previous ${mode === "loop" ? "round" : "question"}</button><button class="button" type="button" data-mock-action="next"${index === sequence.length - 1 ? " disabled" : ""}>Next ${mode === "loop" ? "round" : "question"} &rarr;</button></div>`;
      if (d.phase === 3) container.querySelector("#mock-review").insertAdjacentHTML("beforeend", '<button class="button secondary" type="button" data-mock-action="restart">Try again: keep draft, reset clock and self-scores</button>');
      updateClock();
    }
    function choose(id, nextMode = mode) {
      pause();
      questionId = id;
      mode = nextMode;
      paint();
      container.querySelector("h2").setAttribute("tabindex", "-1");
      container.querySelector("h2").focus({ preventScroll: true });
    }
    function click(event) {
      const button = event.target.closest("[data-mock-action]");
      if (!button || button.disabled) return;
      const action = button.dataset.mockAction;
      if (action === "timer") {
        if (started === null) {
          started = Date.now();
          interval = setInterval(updateClock, 1000);
        } else pause();
        updateClock();
      } else if (action === "reset-clock") {
        pause();
        draft().elapsed = 0;
        persist();
        updateClock();
      } else if (action === "advance" || action === "review") {
        draft().phase = action === "review" ? 3 : Math.min(3, draft().phase + 1);
        if (draft().phase === 3) pause();
        persist();
        paint();
        const target = draft().phase === 3 ? container.querySelector("#mock-review") : [...container.querySelectorAll(".mock-followup")].at(-1);
        target.setAttribute("tabindex", "-1");
        target.focus({ preventScroll: true });
      } else if (action === "restart") {
        pause();
        draft().phase = 0;
        draft().elapsed = 0;
        draft().scores = [null, null, null, null, null];
        persist();
        paint();
        container.querySelector("#mock-answer").focus();
      } else if (action === "loop") choose(company.plan[0], "loop");
      else if (action === "previous" || action === "next") {
        const ids = mode === "loop" ? company.plan : company.questions.map(q => q.id);
        choose(ids[ids.indexOf(questionId) + (action === "next" ? 1 : -1)]);
      }
    }
    function input(event) {
      if (event.target.id === "mock-answer") {
        draft().answer = event.target.value;
        persist();
      }
    }
    function change(event) {
      const target = event.target;
      if (target.id === "mock-question") choose(target.value, "practice");
      else if (target.id === "mock-round") choose(company.questions.find(q => q.round === target.value).id, "practice");
      else if (target.dataset.mockScore !== undefined) {
        draft().scores[Number(target.dataset.mockScore)] = target.value === "" ? null : Number(target.value);
        persist();
        const d = draft();
        container.querySelector("#mock-score-result").textContent = d.scores.every(n => n !== null) ? `Self-score: ${d.scores.reduce((a, n) => a + n, 0)} / 10. Review the weakest dimensions before another attempt.` : "Self-score incomplete: score all five dimensions.";
        container.querySelector("#mock-progress").textContent = `${company.plan.filter(id => drafts[id]?.scores.every(n => n !== null)).length}/6 guided questions self-scored`;
      }
    }
    paint();
    container.addEventListener("click", click);
    container.addEventListener("input", input);
    container.addEventListener("change", change);
    window.addEventListener("pagehide", pause);
    return () => {
      pause();
      disposed = true;
      container.removeEventListener("click", click);
      container.removeEventListener("input", input);
      container.removeEventListener("change", change);
      window.removeEventListener("pagehide", pause);
    };
  }
  return { companies, sources, validDrafts, validRoute, mount };
})();
