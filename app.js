"use strict";

(() => {
  const KEY = "field-guide-v1";
  const chapterCount = GUIDE.lessons.length;
  const questionCount = GUIDE.lessons.reduce((sum, lesson) => sum + lesson.questions.length, 0);
  const examCount = chapterCount * 2;
  const passingCount = Math.ceil(examCount * .8);
  const main = document.getElementById("main");
  const search = document.getElementById("search");
  document.querySelector(".skip-link").addEventListener("click", event => {
    event.preventDefault();
    main.focus();
    main.scrollIntoView();
  });
  let state = { completed: [], quizzes: {}, notes: {}, studio: "", exams: [], coding: {} };
  let codingIndex = 0;
  let codingTrack = "pytorch";
  const codingExercises = () => codingTrack === "spark" ? SPARK_LAB : CODING_LAB;
  let activeExam = null;
  let examAnswers = {};
  let examResult = null;
  let filter = "all";
  let query = "";
  let storageAvailable = true;
  const escape = value => String(value).replace(/[&<>"']/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  const lessonById = id => GUIDE.lessons.find(lesson => lesson.id === id);
  const percent = () => Math.round(state.completed.length / GUIDE.lessons.length * 100);
  const warning = message => {
    const element = document.getElementById("storage-warning");
    element.textContent = message;
    element.hidden = false;
  };
  const isPlain = value => value !== null && typeof value === "object" && !Array.isArray(value);
  function validState(value) {
    if (!isPlain(value) || !Array.isArray(value.completed) || !isPlain(value.quizzes) || !isPlain(value.notes) || typeof value.studio !== "string" || !Array.isArray(value.exams)) return false;
    if (!value.completed.every(id => typeof id === "string" && lessonById(id)) || new Set(value.completed).size !== value.completed.length) return false;
    if (!Object.entries(value.notes).every(([id, note]) => lessonById(id) && typeof note === "string")) return false;
    if (value.coding !== undefined && (!isPlain(value.coding) || !Object.entries(value.coding).every(([id, answers]) => {
      const exercise = [...CODING_LAB, ...SPARK_LAB].find(item => item.id === id);
      return exercise && Array.isArray(answers) && answers.length === exercise.slots.length && answers.every(answer => typeof answer === "string");
    }))) return false;
    if (!Object.entries(value.quizzes).every(([id, result]) => {
      const lesson = lessonById(id);
      return lesson && isPlain(result) && Number.isInteger(result.score) && result.score >= 0 && result.score <= lesson.questions.length && Array.isArray(result.answers) && result.answers.length === lesson.questions.length && result.answers.every((answer, i) => Number.isInteger(answer) && answer >= 0 && answer < lesson.questions[i][1].length);
    })) return false;
    return value.exams.every(result => isPlain(result) && Number.isInteger(result.score) && Number.isInteger(result.total) && result.total > 0 && result.score >= 0 && result.score <= result.total && typeof result.date === "string" && !Number.isNaN(Date.parse(result.date)));
  }
  try {
    const saved = localStorage.getItem(KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (!validState(parsed)) throw new Error("Saved progress has an unrecognized format.");
      state = { ...parsed, coding: parsed.coding || {} };
    }
  } catch (error) {
    storageAvailable = false;
    warning(`Saved progress could not be loaded: ${error.message} This session still works, but changes will not persist. Export your notes before leaving.`);
  }
  function save() {
    if (!storageAvailable) return false;
    try {
      localStorage.setItem(KEY, JSON.stringify(state));
      return true;
    } catch (error) {
      storageAvailable = false;
      warning(`Progress could not be saved: ${error.message} Export your notes before leaving; changes now exist only in this tab.`);
      return false;
    }
  }
  function updateProgress() {
    document.getElementById("sidebar-progress-label").textContent = `${state.completed.length} of ${chapterCount} chapters`;
    document.querySelector(".nav-count").textContent = chapterCount;
    document.getElementById("sidebar-percent").textContent = `${percent()}%`;
    document.getElementById("sidebar-progress-bar").style.width = `${percent()}%`;
  }
  function card(lesson) {
    const complete = state.completed.includes(lesson.id);
    return `<a class="chapter-card" href="#lesson/${lesson.id}">
      <div class="card-meta"><span>${escape(GUIDE.stages[lesson.stage])}</span><span>${lesson.time} MIN READ</span></div>
      <div class="chapter-icon" aria-hidden="true">${lesson.icon}</div>
      <h3>${escape(lesson.title)}</h3><p>${escape(lesson.subtitle)}</p>
      <div class="tag-row">${lesson.tags.slice(0, 3).map(tag => `<span class="tag">${escape(tag)}</span>`).join("")}</div>
      <div class="card-footer"><span>${complete ? '<strong>✓ Completed</strong>' : `CHAPTER ${lesson.icon} <span aria-hidden="true">&middot;</span> 3-QUESTION QUIZ`}</span><span class="arrow" aria-hidden="true">&rarr;</span></div>
    </a>`;
  }
  function dataControls() {
    return `<div class="data-controls"><p>Progress, coding answers, and notes are saved on this browser and origin. Export a backup before switching devices or ports. No personal profile is included.</p><button class="button secondary small-button" id="export-data">Export progress</button><button class="button secondary small-button danger" id="reset-data">Reset progress</button><div id="reset-confirm" hidden></div><div id="export-preview" class="export-preview" hidden></div></div>`;
  }
  function home() {
    const next = GUIDE.lessons.find(lesson => !state.completed.includes(lesson.id)) || GUIDE.lessons[0];
    const quizCount = Object.keys(state.quizzes).length;
    const best = state.exams.length ? `${Math.round(Math.max(...state.exams.map(exam => exam.score / exam.total)) * 100)}%` : "Not yet";
    main.innerHTML = `<div class="page-heading"><div><span class="eyebrow">LEARN WITH INTENTION. INTERVIEW WITH CONFIDENCE.</span><h1>Build intuition. Ship better systems.</h1><p class="intro">A field guide for your next Applied Scientist / MLE interview.<br>From first principles to end-to-end evaluation.</p></div><div class="date-label">APPLIED ML LEARNING PATH<br>FOUNDATIONS &rarr; PRODUCTION &rarr; AGENTS</div></div>
      <section class="hero" aria-label="Continue your learning"><div class="hero-copy"><span class="eyebrow">${state.completed.length ? "KEEP BUILDING YOUR INTUITION" : "THE BIG PICTURE, ONE CHAPTER AT A TIME"}</span><h2>Understand the model.<br>Think beyond the model.</h2><p>Connect transformers, alignment, retrieval, and agents to real production decisions. Learn the why. Practice the how.</p><a class="button" href="#lesson/${next.id}">${state.completed.length ? "Continue learning" : "Start the journey"} <span aria-hidden="true">&rarr;</span></a><span class="hero-note">Next: ${escape(next.title)}</span></div><div class="hero-art" aria-hidden="true"><div class="orbit"></div><div class="orbit two"></div><span class="diagram-spark">✧</span><div class="diagram-stack"><div class="diagram-node">FOUNDATIONS</div><div class="diagram-node">SYSTEMS + EVALUATION</div><div class="diagram-node">AGENTIC THINKING</div></div><span class="art-caption">INTUITION → IMPLEMENTATION → IMPACT</span></div></section>
      <section class="stats" aria-label="Your learning statistics"><div class="stat"><span class="stat-icon" aria-hidden="true">▤</span><div><strong>${chapterCount}</strong><small>In-depth chapters</small></div></div><div class="stat"><span class="stat-icon" aria-hidden="true">◷</span><div><strong>${percent()}%</strong><small>Your path completed</small></div></div><div class="stat"><span class="stat-icon" aria-hidden="true">◇</span><div><strong>${quizCount} / ${chapterCount}</strong><small>Concept quizzes taken</small></div></div><div class="stat"><span class="stat-icon" aria-hidden="true">✓</span><div><strong>${best}</strong><small>Best final assessment</small></div></div></section>
      <div class="section-heading"><div><h2>A solid foundation changes everything.</h2><p>Your first three chapters. Build the mental model before the toolkit.</p></div><a href="#curriculum">Explore all chapters &rarr;</a></div>
      <div class="chapter-grid">${GUIDE.lessons.slice(0, 3).map(card).join("")}</div>
      <div class="personal-strip"><span class="strip-icon" aria-hidden="true">✦</span><div><h3>Practice decisions, not just definitions.</h3><p>Practice search, SLMs, RAG, ranking, GPU systems, and distributed data.</p></div><a href="#practice">Enter interview studio &rarr;</a></div>
      <section class="section-spacer"><div class="section-heading"><div><h2>Don't stop at the architecture diagram.</h2><p>Learn the complete lifecycle, then prove it works.</p></div></div><div class="lab-grid"><a class="panel" href="#lesson/rag"><span class="eyebrow">END-TO-END RAG</span><h2>From source to supported answer.</h2><p>Ingestion → permissions → retrieval → reranking → grounding → component evaluation → online quality → freshness monitoring.</p><span class="completion-label">Explore the RAG lifecycle &rarr;</span></a><a class="panel" href="#lesson/agent-eval"><span class="eyebrow">END-TO-END AGENTS</span><h2>From intent to verified outcome.</h2><p>Task → tools → state → execution → recovery → trajectory evaluation → final-state checks → safe rollout.</p><span class="completion-label">Explore agent evaluation &rarr;</span></a></div></section>${dataControls()}`;
  }
  function curriculum() {
    const searchText = query.trim().toLowerCase();
    const matches = GUIDE.lessons.filter(lesson => {
      const text = `${lesson.title} ${lesson.subtitle} ${lesson.tags.join(" ")} ${lesson.objective} ${lesson.sections.map(section => section.title + " " + section.body.replace(/<[^>]+>/g, " ")).join(" ")}`.toLowerCase();
      return (filter === "all" || lesson.stage === Number(filter)) && (!searchText || text.includes(searchText));
    });
    main.innerHTML = `<div class="page-heading"><div><span class="eyebrow">A CONNECTED CURRICULUM</span><h1>From first principles to production.</h1><p class="intro">${chapterCount} chapters · ${questionCount} concept questions · a ${examCount}-question final.<br>Read in order, or go straight to the gap you want to close.</p></div></div><div class="filter-bar" aria-label="Filter by stage">${["all", ...GUIDE.stages.map((_, i) => i)].map((stage, i) => `<button class="filter-chip ${String(stage) === filter ? "active" : ""}" data-filter="${stage}" aria-pressed="${String(stage) === filter}">${i === 0 ? "All chapters" : escape(GUIDE.stages[stage])}</button>`).join("")}</div>${searchText ? `<p class="muted search-heading" role="status">${matches.length} chapter${matches.length === 1 ? "" : "s"} matching “${escape(query)}”</p>` : ""}${matches.length ? GUIDE.stages.map((stage, index) => {
      const group = matches.filter(lesson => lesson.stage === index);
      return group.length ? `<section><div class="stage-header"><span class="stage-index">0${index + 1}</span><div><h2>${stage}</h2><small>${["Build the mental model", "Adapt the behavior", "Retrieve, serve, and measure", "Act, verify, and defend", "Distributed data fundamentals"][index]}</small></div></div><div class="chapter-grid">${group.map(card).join("")}</div></section>` : "";
    }).join("") : '<div class="empty-state"><h2>No matching chapters.</h2><p class="muted">Try a broader term like attention, retrieval, or evaluation, or select All chapters.</p></div>'}`;
  }
  function quizMarkup(questions, prefix, saved, exam = false) {
    return questions.map((question, i) => `<fieldset class="question"><legend><span class="question-number">${String(i + 1).padStart(2, "0")}</span>${escape(question[0])}</legend>${question[1].map((option, j) => `<label class="option"><input type="radio" name="${prefix}-${i}" value="${j}" ${saved && saved[i] === j ? "checked" : ""}><span>${escape(option)}</span></label>`).join("")}<div class="feedback" id="${prefix}-feedback-${i}" hidden></div>${exam ? "" : ""}</fieldset>`).join("");
  }
  function lessonPage(lesson) {
    const saved = state.quizzes[lesson.id];
    const index = GUIDE.lessons.indexOf(lesson);
    main.innerHTML = `<div class="lesson-layout"><article><header class="lesson-header"><span class="eyebrow">CHAPTER ${lesson.icon} / ${escape(GUIDE.stages[lesson.stage])}</span><h1>${escape(lesson.title)}</h1><p class="intro">${escape(lesson.subtitle)}</p><div class="lesson-meta">${lesson.time} min read <span>&middot;</span> ${lesson.sections.length} sections <span>&middot;</span> 3-question concept check</div><div class="tag-row">${lesson.tags.map(tag => `<span class="tag">${escape(tag)}</span>`).join("")}</div><div class="objective"><strong>By the end:</strong> ${escape(lesson.objective)}</div></header><div class="lesson-body">${lesson.sections.map((section, i) => `<section class="lesson-section" id="section-${i}"><h2>${section.title}</h2>${section.body}</section>`).join("")}<section class="sources"><h3>Read the original ideas</h3>${lesson.sources.map(([label, url]) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${escape(label)} ↗</a>`).join("")}</section><section class="quiz" id="concept-check"><span class="eyebrow">PAUSE. RETRIEVE. REINFORCE.</span><h2>Check your intuition.</h2><p class="quiz-intro">Answer all three before checking. Explanations appear after submission; retry as often as you like. Chapter completion is self-reported, not a passing grade.</p><form id="lesson-quiz" data-lesson="${lesson.id}">${quizMarkup(lesson.questions, "lesson", saved ? saved.answers : null)}<p class="inline-error" id="quiz-error" role="alert" hidden></p><div id="quiz-result" aria-live="polite"></div><div class="quiz-controls"><button class="button" type="submit">Check answers &rarr;</button><button class="button secondary" type="button" id="retry-quiz" ${saved ? "" : "hidden"}>Try again</button></div></form></section><div class="lesson-actions"><button class="button" id="complete-lesson" data-id="${lesson.id}">${state.completed.includes(lesson.id) ? "✓ Completed · Mark incomplete" : "Mark chapter complete ✓"}</button>${index < GUIDE.lessons.length - 1 ? `<a class="button secondary" href="#lesson/${GUIDE.lessons[index + 1].id}">Next chapter &rarr;</a>` : '<a class="button secondary" href="#exam">Take the final assessment &rarr;</a>'}</div></div></article><aside class="lesson-aside" aria-label="Chapter contents and notes"><h3>IN THIS CHAPTER</h3>${lesson.sections.map((section, i) => `<a href="#section-${i}" data-section="${i}">${section.title}</a>`).join("")}<a href="#concept-check" data-section="quiz">Concept check</a><div class="notes-area"><label for="lesson-notes">YOUR MARGIN NOTES</label><textarea id="lesson-notes" data-id="${lesson.id}" placeholder="An intuition, a question, a detail from your project...">${escape(state.notes[lesson.id] || "")}</textarea><div class="save-status" id="note-status" aria-live="polite">Private · saved on this browser</div></div></aside></div>`;
    if (saved) showFeedback(lesson.questions, saved.answers, "lesson");
  }
  function showFeedback(questions, answers, prefix) {
    let score = 0;
    questions.forEach((question, i) => {
      const correct = answers[i] === question[2];
      if (correct) score++;
      const inputs = main.querySelectorAll(`input[name="${prefix}-${i}"]`);
      inputs.forEach(input => {
        input.disabled = true;
        input.closest(".option").classList.toggle("correct", Number(input.value) === question[2]);
        input.closest(".option").classList.toggle("incorrect", Number(input.value) === answers[i] && !correct);
      });
      const feedback = document.getElementById(`${prefix}-feedback-${i}`);
      feedback.hidden = false;
      feedback.className = `feedback${correct ? "" : " wrong"}`;
      feedback.textContent = `${correct ? "Correct. " : `Not quite. Correct answer: ${question[1][question[2]]}. `}${question[3]}`;
    });
    if (prefix === "lesson") {
      document.getElementById("quiz-result").innerHTML = `<div class="quiz-result ${score === questions.length ? "" : "needs-work"}"><strong>${score} / ${questions.length} correct.</strong> ${score === questions.length ? "Nice work. Can you explain the reasoning out loud?" : "Review the explanations, then try again without looking."}</div>`;
      document.querySelector('#lesson-quiz button[type="submit"]').hidden = true;
      document.getElementById("retry-quiz").hidden = false;
    }
    return score;
  }
  function lab() {
    main.innerHTML = `<div class="page-heading"><div><span class="eyebrow">MAKE THE EQUATIONS TANGIBLE</span><h1>A small lab. A better intuition.</h1><p class="intro">Change the inputs, predict the result, then explain why it changed.<br>These are educational calculations, not production capacity forecasts.</p></div></div><div class="lab-grid"><section class="panel"><span class="eyebrow">01 / ATTENTION SANDBOX</span><h2>Where does attention go?</h2><p>One query, three key scores [2, 1, 0]. Watch softmax redistribute attention as you change temperature or mask the last key.</p><div class="field"><label for="temperature">Temperature: <output id="temp-value">1.0</output></label><input id="temperature" type="range" min="0.1" max="3" step="0.1" value="1"></div><label class="check-line"><input id="mask-key" type="checkbox">Mask the third key before softmax</label><div id="attention-bars" class="attention-bars" aria-live="polite"></div><div class="lab-output"><strong id="attention-sum">100%</strong><p>Probability mass over unmasked keys. This isolates softmax; real attention also includes learned Q/K projections and head-dimension scaling.</p></div></section><section class="panel"><span class="eyebrow">02 / KV MEMORY ESTIMATOR</span><h2>What does context cost?</h2><p>Ideal cache size = 2 × batch × tokens × layers × KV heads × head width × bytes. Query heads do not replace KV heads in this formula.</p><div class="form-row"><div class="field"><label for="kv-batch">Concurrent sequences (batch)</label><input id="kv-batch" type="number" min="1" max="1024" value="1" required></div><div class="field"><label for="kv-tokens">Tokens per sequence</label><input id="kv-tokens" type="number" min="1" max="1000000" value="4096" required></div></div><div class="form-row"><div class="field"><label for="kv-layers">Transformer layers</label><input id="kv-layers" type="number" min="1" max="1024" value="32" required></div><div class="field"><label for="kv-heads">KV heads</label><input id="kv-heads" type="number" min="1" max="1024" value="8" required></div></div><div class="form-row"><div class="field"><label for="kv-width">Head dimension</label><input id="kv-width" type="number" min="1" max="4096" value="128" required></div><div class="field"><label for="kv-bytes">Cache precision</label><select id="kv-bytes"><option value="2">BF16 / FP16 · 2 bytes</option><option value="1">INT8 / FP8 · 1 byte</option><option value="4">FP32 · 4 bytes</option></select></div></div><div id="kv-output" class="lab-output" aria-live="polite"></div></section><section class="panel"><span class="eyebrow">03 / RANKING METRICS</span><h2>Order changes the outcome.</h2><p>Enter graded relevance labels (0–3) in retrieved order. This demo treats those documents as the complete judged pool, so it evaluates ordering only—not retrieval recall.</p><div class="field"><label for="ranking-labels">Comma-separated relevance labels (max 30)</label><input id="ranking-labels" value="0, 2, 1" autocomplete="off"></div><div id="ranking-output" class="lab-output" aria-live="polite"></div><p class="section-spacer">Gain = 2<sup>rel</sup> − 1. Discount = log<sub>2</sub>(rank + 1). With no relevant documents, NDCG is undefined here rather than silently treated as a perfect score.</p></section><section class="panel"><span class="eyebrow">04 / DPO LOSS EXPLORER</span><h2>Feel the preference margin.</h2><p>m is the chosen minus rejected log-probability improvement relative to the reference policy. Loss = −log sigmoid(βm).</p><div class="field"><label for="dpo-margin">Relative margin: <output id="margin-value">0.0</output></label><input id="dpo-margin" type="range" min="-10" max="10" step=".1" value="0"></div><div class="field"><label for="dpo-beta">Beta: <output id="beta-value">0.1</output></label><input id="dpo-beta" type="range" min=".1" max="2" step=".1" value=".1"></div><div id="dpo-output" class="lab-output" aria-live="polite"></div><p class="section-spacer">A lower pairwise loss is not evidence of more truthful answers. Evaluate grounded correctness and label quality separately.</p></section></div>`;
    updateLabs();
    main.insertAdjacentHTML("beforeend", '<div class="personal-strip"><span class="strip-icon" aria-hidden="true">⌨</span><div><h3>Now write the PyTorch.</h3><p>Six guided coding exercises, hints, live code previews, and a runnable notebook with tests.</p></div><a href="#coding">Open coding lab &rarr;</a></div>');
  }
  function exerciseCode(exercise, answers) {
    return exercise.scaffold.replace(/__SLOT_(\d+)__/g, (_, index) => answers[Number(index)]?.trim() || "...");
  }
  function coding() {
    const exercises = codingExercises();
    const exercise = exercises[codingIndex];
    const spark = codingTrack === "spark";
    const runtime = spark ? "PySpark" : "PyTorch";
    const notebook = spark ? "spark-exercises.ipynb" : "pytorch-exercises.ipynb";
    const answers = state.coding[exercise.id] || exercise.slots.map(() => "");
    main.innerHTML = `<div class="page-heading"><div><span class="eyebrow">${runtime.toUpperCase()} / LEARN BY WRITING</span><h1>${spark ? "Think in data. Write the transformation." : "From an equation to working code."}</h1><p class="intro">${exercises.length} exercises. Fill the missing expressions, inspect the assembled code, and explain the result before opening the solution.</p></div></div><div class="filter-bar"><a class="filter-chip ${spark ? "" : "active"}" href="#coding">PyTorch coding</a><a class="filter-chip ${spark ? "active" : ""}" href="#spark-coding">Spark add-on</a></div><div class="coding-notice"><strong>Two environments. An honest boundary.</strong><p>Browser checks compare supported reference expressions; they do not execute Python or prove semantic equivalence. The notebook runs real ${runtime} assertions in your own compatible environment. No code is sent to a service.</p><a class="button secondary small-button" href="${notebook}" download>Download the runnable notebook &darr;</a><a class="button text small-button" href="${notebook}" target="_blank" rel="noopener">Open notebook file ↗</a></div><div class="coding-selector"><label for="coding-exercise">CHOOSE AN EXERCISE</label><select id="coding-exercise">${exercises.map((item, index) => `<option value="${index}" ${index === codingIndex ? "selected" : ""}>${index + 1}. ${escape(item.title)}</option>`).join("")}</select><span class="saved-label">${codingIndex + 1} of ${exercises.length}</span></div><div class="coding-layout"><section class="panel"><span class="eyebrow">${escape(exercise.level)}</span><h2>${escape(exercise.title)}</h2><p>${escape(exercise.objective)}</p><form id="coding-form">${exercise.slots.map((slot, index) => `<div class="coding-field"><label for="coding-slot-${index}"><span>${index + 1}</span> ${escape(slot.label)}</label><textarea id="coding-slot-${index}" data-coding-slot="${index}" spellcheck="false" autocapitalize="off" autocomplete="off" placeholder="${escape(slot.placeholder)}" aria-describedby="coding-feedback-${index}">${escape(answers[index])}</textarea><p id="coding-feedback-${index}" class="coding-feedback" hidden></p><details><summary>Need a hint?</summary><p>${escape(slot.hint)}</p></details></div>`).join("")}<div id="coding-result" role="status" aria-live="polite"></div><div class="quiz-controls"><button class="button" type="submit">Check expressions &rarr;</button><button class="button secondary" type="button" id="reset-exercise">Clear this exercise</button></div><div id="note-status" class="save-status" aria-live="polite">Expressions save on this browser</div></form></section><section class="panel coding-preview"><span class="eyebrow">LIVE PYTHON PREVIEW / NOT EXECUTED</span><pre class="code-block"><code id="assembled-code">${escape(exerciseCode(exercise, answers))}</code></pre><h3>Expected behavior</h3><p>${escape(exercise.expected)}</p><details><summary>Inspect the real notebook tests</summary><pre class="code-block"><code>${escape(exercise.tests)}</code></pre></details><details><summary>Reveal the reference solution</summary><pre class="code-block"><code>${escape(exerciseCode(exercise, exercise.slots.map(slot => slot.answers[0])))}</code></pre><p>${escape(exercise.explanation)}</p></details><a class="button text" href="#lesson/${exercise.lesson}">Review the related chapter &rarr;</a></section></div><div class="lesson-actions"><button class="button secondary" id="previous-exercise" ${codingIndex === 0 ? "disabled" : ""}>&larr; Previous exercise</button><button class="button" id="next-exercise" ${codingIndex === exercises.length - 1 ? "disabled" : ""}>Next exercise &rarr;</button></div><details class="section-spacer"><summary>How to run the notebook and use the checks</summary><p>Open the notebook in VS Code with a compatible Jupyter/Python kernel. ${spark ? "Install a compatible PySpark distribution and Java runtime; configure JAVA_HOME as required by that Spark release. The notebook uses a local[2] session and classic Spark APIs, not Spark Connect." : "Install a compatible PyTorch build if needed. The exercises use CPU tensors and need no checkpoint download or CUDA device."} Fill each ellipsis, then run its separate test cell. Reference solutions appear at the end as text, not automatically executed.</p><p>The assertions cover toy inputs and selected failure cases, not every production input. A different valid expression may require notebook execution to verify. No private data is needed.</p></details>`;
  }
  function checkCoding() {
    const exercise = codingExercises()[codingIndex];
    const answers = state.coding[exercise.id] || exercise.slots.map(() => "");
    const normalize = text => text.replace(/\s+/g, "");
    let matches = 0;
    exercise.slots.forEach((slot, index) => {
      const matched = slot.answers.some(answer => normalize(answer) === normalize(answers[index]));
      if (matched) matches++;
      const feedback = document.getElementById(`coding-feedback-${index}`);
      feedback.hidden = false;
      feedback.className = `coding-feedback ${matched ? "match" : "no-match"}`;
      feedback.textContent = matched ? "Reference expression matched." : !answers[index].trim() ? "Fill this expression before checking." : "Not a supported reference form. It may still be valid—compare the solution or run the notebook tests.";
    });
    document.getElementById("coding-result").innerHTML = `<div class="quiz-result ${matches === exercise.slots.length ? "" : "needs-work"}"><strong>${matches} / ${exercise.slots.length} expressions match.</strong> ${matches === exercise.slots.length ? "Now explain the tensor shapes and run the actual notebook tests. This was an expression check, not Python execution." : "Use the hints, inspect your assembled code, and try again. No Python was executed."}</div>`;
  }
  function updateLabs() {
    const temperature = Number(document.getElementById("temperature").value);
    const masked = document.getElementById("mask-key").checked;
    const scores = [2, 1, 0].map((score, i) => masked && i === 2 ? 0 : Math.exp((score - 2) / temperature));
    const sum = scores.reduce((a, b) => a + b, 0);
    document.getElementById("temp-value").textContent = temperature.toFixed(1);
    document.getElementById("attention-bars").innerHTML = scores.map((score, i) => `<div class="attention-row"><span>Key ${i + 1}</span><div class="attention-bar"><span style="width:${score / sum * 100}%"></span></div><strong>${(score / sum * 100).toFixed(1)}%</strong></div>`).join("");
    const ids = ["kv-batch", "kv-tokens", "kv-layers", "kv-heads", "kv-width"];
    const fields = ids.map(id => document.getElementById(id));
    const output = document.getElementById("kv-output");
    if (fields.some(field => !field.validity.valid || !Number.isInteger(Number(field.value)))) {
      output.innerHTML = "<p class='inline-error'>Enter whole numbers within the displayed field limits; all dimensions must be positive.</p>";
    } else {
      const bytes = 2 * fields.reduce((product, field) => product * Number(field.value), 1) * Number(document.getElementById("kv-bytes").value);
      output.innerHTML = `<strong>${bytes < 1024 ** 3 ? (bytes / 1024 ** 2).toFixed(1) + " MiB" : (bytes / 1024 ** 3).toFixed(2) + " GiB"}</strong><p>${bytes.toLocaleString("en-US")} bytes. Excludes model weights, activations, workspaces, fragmentation, and quantization metadata. Large values are arithmetic estimates, not feasible configurations.</p>`;
    }
    const raw = document.getElementById("ranking-labels").value.trim();
    const labels = raw.split(",").map(label => label.trim());
    const rankingOutput = document.getElementById("ranking-output");
    if (!raw || labels.length > 30 || labels.some(label => !/^[0-3]$/.test(label))) {
      rankingOutput.innerHTML = "<p class='inline-error'>Use 1–30 comma-separated integers between 0 and 3, with no empty entries.</p>";
    } else {
      const relevance = labels.map(Number);
      const dcg = list => list.reduce((total, rel, i) => total + (2 ** rel - 1) / Math.log2(i + 2), 0);
      const actual = dcg(relevance);
      const ideal = dcg([...relevance].sort((a, b) => b - a));
      rankingOutput.innerHTML = `<strong>${ideal === 0 ? "Undefined" : (actual / ideal).toFixed(3)}</strong><p>NDCG@${labels.length} · DCG ${actual.toFixed(3)} · IDCG ${ideal.toFixed(3)}${ideal === 0 ? " · No relevant documents in this judged pool." : ""}</p>`;
    }
    const margin = Number(document.getElementById("dpo-margin").value);
    const beta = Number(document.getElementById("dpo-beta").value);
    const z = -beta * margin;
    const loss = Math.max(0, z) + Math.log1p(Math.exp(-Math.abs(z)));
    document.getElementById("margin-value").textContent = margin.toFixed(1);
    document.getElementById("beta-value").textContent = beta.toFixed(1);
    document.getElementById("dpo-output").innerHTML = `<strong>${loss.toFixed(4)}</strong><p>Pairwise DPO loss · scaled margin ${(beta * margin).toFixed(2)}. Zero margin always gives log(2) ≈ 0.6931.</p>`;
  }
  function practice() {
    main.innerHTML = `<div class="page-heading"><div><span class="eyebrow">YOUR DESIGN DECISIONS, UNDER THE MICROSCOPE</span><h1>The interview studio.</h1><p class="intro">Use these synthetic scenarios to practice clear reasoning, evaluation, and trade-offs. They do not describe any individual's employment history.</p></div></div><div class="studio-grid">${GUIDE.scenarios.map(scenario => `<section class="panel scenario"><div class="scenario-meta"><span>${escape(scenario.kind).toUpperCase()}</span><span>${scenario.time}</span></div><h2>${escape(scenario.title)}</h2><p class="prompt">${escape(scenario.prompt)}</p><details><summary>Open the answer rubric</summary><ul>${scenario.checks.map(check => `<li>${escape(check)}</li>`).join("")}</ul></details></section>`).join("")}<section class="panel scenario"><div class="scenario-meta"><span>SELF-ASSESSMENT</span><span>10 MIN</span></div><h2>Make every claim defensible.</h2><p class="prompt">Rehearse aloud. Score each dimension from 0 (missing) to 2 (specific and correct). Aim for 8/10 or better with no technical-correctness gap.</p><ul class="review-list"><li>Problem framing and constraints</li><li>Technical correctness and derivation</li><li>Evidence and measurement</li><li>Alternatives and trade-offs</li><li>Failure handling and operational ownership</li></ul></section></div><section class="studio-note"><h2>Your rehearsal notebook.</h2><p class="intro">Draft a two-minute story. Include baseline, your decision, evidence, and one limitation. Don't paste confidential work details into a publicly shared version.</p><label for="studio-notes" class="eyebrow">PRIVATE NOTES</label><textarea id="studio-notes" placeholder="Situation → decision → experiment → result → what I would change...">${escape(state.studio)}</textarea><div id="note-status" class="save-status" aria-live="polite">Saved on this browser only</div></section>${dataControls()}`;
  }
  function shuffled(list) {
    const copy = [...list];
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  }
  function startExam() {
    activeExam = shuffled(GUIDE.lessons.flatMap(lesson => shuffled(lesson.questions).slice(0, 2).map(question => ({ question, lesson: lesson.id, stage: lesson.stage }))));
    examAnswers = {};
    examResult = null;
    examPage();
    main.focus();
    window.scrollTo(0, 0);
  }
  function examPage() {
    if (!activeExam) {
      const latest = state.exams[state.exams.length - 1];
      main.innerHTML = `<div class="page-heading"><div><span class="eyebrow">RETRIEVE. CONNECT. EXPLAIN.</span><h1>The final checkpoint.</h1><p class="intro">Test your conceptual breadth across the complete learning path. Then return to the gaps—not just the score.</p></div></div><div class="exam-intro"><section class="panel"><h2>${examCount} questions. One connected picture.</h2><p>Two randomly selected questions from every chapter, shuffled each attempt. This reuses the concept-check question bank, so it is a learning assessment—not an independent certification.</p><ul class="exam-rules"><li>All ${examCount} questions must be answered before submission.</li><li>Pass threshold: 80%, meaning at least ${passingCount} / ${examCount}.</li><li>No timer. Plan for about 30 minutes without notes.</li><li>Explanations and a chapter-level review plan follow submission.</li><li>Answers stay in this tab while navigating; refreshing discards an unfinished attempt.</li></ul><button class="button" id="start-exam">Begin assessment &rarr;</button>${latest ? `<div class="exam-history">Last attempt: ${latest.score} / ${latest.total} (${Math.round(latest.score / latest.total * 100)}%) · ${escape(new Date(latest.date).toLocaleDateString())}<br>${state.exams.length} saved attempt${state.exams.length === 1 ? "" : "s"} · best ${Math.round(Math.max(...state.exams.map(result => result.score / result.total)) * 100)}%</div>` : ""}</section><section class="panel"><span class="eyebrow">WHAT THIS COVERS</span><h2>Not just model trivia.</h2><p>Transformers, PyTorch, SFT, DPO, PPO, FlashAttention, CUDA, inference, retrieval, RAG and evaluation, agent design and evaluation, and system design.</p><p>A high score is a useful signal. It does not replace coding practice, open-ended derivation, or defending real project decisions.</p><a class="button secondary" href="#practice">Practice the deep dives &rarr;</a></section></div>`;
      return;
    }
    main.innerHTML = `<div class="page-heading"><div><span class="eyebrow">FINAL ASSESSMENT</span><h1>${examResult ? "Feedback worth studying." : "Trust the intuition you've built."}</h1><p class="intro">${examCount} questions · all ${chapterCount} chapters · 80% to pass</p></div></div><div id="exam-summary" aria-live="polite"></div><section class="quiz"><div class="exam-topline"><span id="exam-answered">${Object.keys(examAnswers).length} / ${examCount} answered</span><span class="tag">Untimed · browser-local</span></div><form id="exam-form">${quizMarkup(activeExam.map(item => item.question), "exam", examAnswers, true)}<p class="inline-error" id="exam-error" role="alert" hidden></p><button class="button" type="submit" ${examResult ? "hidden" : ""}>Submit assessment &rarr;</button></form></section>`;
    if (examResult) renderExamResult();
  }
  function renderExamResult() {
    const score = showFeedback(activeExam.map(item => item.question), examAnswers, "exam");
    const failed = activeExam.filter((item, i) => examAnswers[i] !== item.question[2]);
    const reviewIds = [...new Set(failed.map(item => item.lesson))];
    const passed = score / activeExam.length >= .8;
    document.getElementById("exam-summary").innerHTML = `<section class="panel"><span class="eyebrow">${passed ? "CHECKPOINT PASSED" : "YOUR NEXT STUDY PLAN"}</span><h2>${score} / ${examCount} · ${Math.round(score / examCount * 100)}%</h2><p>${passed ? "A strong conceptual foundation. Now practice explaining and implementing the ideas without answer choices." : "Not there yet—and now you know where to focus. Review the chapters below, then try another attempt."}</p><div class="exam-domain">${GUIDE.stages.map((stage, index) => {
      const items = activeExam.map((item, i) => ({ ...item, i })).filter(item => item.stage === index);
      const correct = items.filter(item => examAnswers[item.i] === item.question[2]).length;
      return `<div class="domain-score">${stage}: <strong>${correct} / ${items.length}</strong></div>`;
    }).join("")}</div>${reviewIds.length ? `<h3>Review these chapters</h3><ul class="review-list">${reviewIds.map(id => `<li><a href="#lesson/${id}">${escape(lessonById(id).title)}</a></li>`).join("")}</ul>` : '<p>All correct. Try an open-ended design scenario next.</p>'}<div class="quiz-controls"><button class="button" id="retake-exam">Start a new attempt</button><a class="button secondary" href="#practice">Practice a project story</a></div></section>`;
  }
  function render(focus = true) {
    const hash = location.hash.slice(1) || "home";
    const [page, id] = hash.split("/");
    const lesson = page === "lesson" ? lessonById(id) : null;
    const validPage = ["home", "curriculum", "lab", "coding", "spark-coding", "practice", "exam"].includes(page) || lesson;
    const nextTrack = page === "spark-coding" ? "spark" : "pytorch";
    if ((page === "coding" || page === "spark-coding") && nextTrack !== codingTrack) {
      codingTrack = nextTrack;
      codingIndex = 0;
    }
    document.querySelectorAll("[data-nav]").forEach(link => {
      const active = link.dataset.nav === (lesson ? "curriculum" : page === "spark-coding" ? "coding" : page);
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    const labels = { home: "Overview", curriculum: "Learning path", lab: "Interactive lab", coding: "PyTorch coding", "spark-coding": "Spark coding", practice: "Interview studio", exam: "Final assessment" };
    document.getElementById("breadcrumb").textContent = lesson ? `Chapter ${lesson.icon}` : labels[page] || "Not found";
    document.title = `${lesson ? lesson.title : labels[page] || "Not found"} · Field Guide`;
    if (!validPage) main.innerHTML = '<div class="empty-state"><h1>This page is not in the guide.</h1><a class="button" href="#home">Back to overview</a></div>';
    else if (lesson) lessonPage(lesson);
    else ({ home, curriculum, lab, coding, "spark-coding": coding, practice, exam: examPage })[page]();
    updateProgress();
    if (focus) {
      main.focus({ preventScroll: true });
      window.scrollTo(0, 0);
    }
  }
  main.addEventListener("click", event => {
    const element = event.target.closest("button, a");
    if (!element) return;
    if (element.dataset.filter !== undefined) {
      filter = element.dataset.filter;
      curriculum();
    } else if (element.dataset.section !== undefined) {
      event.preventDefault();
      document.getElementById(element.dataset.section === "quiz" ? "concept-check" : `section-${element.dataset.section}`).scrollIntoView({ behavior: "smooth" });
    } else if (element.id === "complete-lesson") {
      const id = element.dataset.id;
      state.completed = state.completed.includes(id) ? state.completed.filter(value => value !== id) : [...state.completed, id];
      save();
      updateProgress();
      element.textContent = state.completed.includes(id) ? "✓ Completed · Mark incomplete" : "Mark chapter complete ✓";
    } else if (element.id === "retry-quiz") {
      delete state.quizzes[document.getElementById("lesson-quiz").dataset.lesson];
      save();
      lessonPage(lessonById(location.hash.split("/")[1]));
      document.getElementById("concept-check").scrollIntoView();
    } else if (element.id === "next-exercise" || element.id === "previous-exercise") {
      codingIndex += element.id === "next-exercise" ? 1 : -1;
      coding();
      main.focus();
      window.scrollTo(0, 0);
    } else if (element.id === "reset-exercise") {
      delete state.coding[codingExercises()[codingIndex].id];
      save();
      coding();
    } else if (element.id === "start-exam" || element.id === "retake-exam") startExam();
    else if (element.id === "export-data") {
      const json = JSON.stringify(state, null, 2);
      const preview = document.getElementById("export-preview");
      preview.hidden = false;
      preview.innerHTML = `<label for="progress-json">If your browser blocks downloads, copy this JSON backup:</label><textarea id="progress-json" readonly spellcheck="false">${escape(json)}</textarea>`;
      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "field-guide-progress.json";
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } else if (element.id === "reset-data") {
      const confirm = document.getElementById("reset-confirm");
      confirm.hidden = false;
      confirm.className = "confirm-reset";
      confirm.innerHTML = 'Delete all saved notes, coding answers, chapter progress, quiz results, and exam history? <button class="button secondary small-button danger" id="confirm-reset">Yes, delete study data</button><button class="button secondary small-button" id="cancel-reset">Cancel</button>';
    } else if (element.id === "cancel-reset") document.getElementById("reset-confirm").hidden = true;
    else if (element.id === "confirm-reset") {
      try {
        localStorage.removeItem(KEY);
        storageAvailable = true;
        document.getElementById("storage-warning").hidden = true;
      } catch (error) {
        warning(`Saved data could not be deleted: ${error.message} Clear this site's storage in your browser settings.`);
        return;
      }
      state = { completed: [], quizzes: {}, notes: {}, studio: "", exams: [], coding: {} };
      activeExam = null;
      examAnswers = {};
      examResult = null;
      render(false);
    }
  });
  main.addEventListener("input", event => {
    const target = event.target;
    if (target.dataset.codingSlot !== undefined) {
      const exercise = codingExercises()[codingIndex];
      const answers = state.coding[exercise.id] || exercise.slots.map(() => "");
      answers[Number(target.dataset.codingSlot)] = target.value;
      state.coding[exercise.id] = answers;
      document.getElementById("assembled-code").textContent = exerciseCode(exercise, answers);
      document.getElementById("coding-result").innerHTML = "";
      document.querySelectorAll(".coding-feedback").forEach(element => { element.hidden = true; });
      document.getElementById("note-status").textContent = save() ? "Expressions saved on this browser" : "Not saved · export before leaving";
    } else if (target.id === "lesson-notes" || target.id === "studio-notes") {
      if (target.id === "lesson-notes") state.notes[target.dataset.id] = target.value;
      else state.studio = target.value;
      document.getElementById("note-status").textContent = save() ? "Saved on this browser" : "Not saved · export before leaving";
    } else if (location.hash === "#lab") updateLabs();
  });
  main.addEventListener("change", event => {
    if (event.target.id === "coding-exercise") {
      codingIndex = Number(event.target.value);
      coding();
    } else if (event.target.name && event.target.name.startsWith("exam-") && activeExam && !examResult) {
      examAnswers[Number(event.target.name.slice(5))] = Number(event.target.value);
      document.getElementById("exam-answered").textContent = `${Object.keys(examAnswers).length} / ${examCount} answered`;
    }
  });
  main.addEventListener("submit", event => {
    event.preventDefault();
    const form = event.target;
    if (form.id === "coding-form") {
      checkCoding();
      return;
    }
    if (form.id !== "lesson-quiz" && form.id !== "exam-form") return;
    if (form.id === "exam-form" && examResult) return;
    const lesson = form.id === "lesson-quiz" ? lessonById(form.dataset.lesson) : null;
    const questions = lesson ? lesson.questions : activeExam.map(item => item.question);
    const prefix = lesson ? "lesson" : "exam";
    const formData = new FormData(form);
    const answers = questions.map((_, i) => formData.get(`${prefix}-${i}`));
    const error = document.getElementById(lesson ? "quiz-error" : "exam-error");
    if (answers.some(answer => answer === null)) {
      error.hidden = false;
      error.textContent = `Answer all ${questions.length} questions before submitting.`;
      const missing = answers.findIndex(answer => answer === null);
      form.querySelector(`input[name="${prefix}-${missing}"]`).focus();
      return;
    }
    error.hidden = true;
    const numericAnswers = answers.map(Number);
    if (lesson) {
      const score = showFeedback(questions, numericAnswers, prefix);
      state.quizzes[lesson.id] = { score, answers: numericAnswers };
      save();
      document.getElementById("quiz-result").scrollIntoView({ block: "center" });
    } else {
      examAnswers = Object.fromEntries(numericAnswers.map((answer, i) => [i, answer]));
      const score = questions.filter((question, i) => numericAnswers[i] === question[2]).length;
      examResult = { score, total: questions.length, date: new Date().toISOString() };
      state.exams.push(examResult);
      save();
      examPage();
      window.scrollTo(0, 0);
    }
  });
  search.addEventListener("input", () => {
    query = search.value;
    if (location.hash !== "#curriculum") location.hash = "curriculum";
    else curriculum();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "/" && !event.ctrlKey && !event.metaKey && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
      event.preventDefault();
      search.focus();
    }
  });
  window.addEventListener("hashchange", () => render(document.activeElement !== search));
  render(false);
})();
