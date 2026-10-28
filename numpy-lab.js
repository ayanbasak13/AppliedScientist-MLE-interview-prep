"use strict";
const NUMPY_LAB = [
  {
    "id": "numpy-indexing",
    "title": "Distinguish slices, paired indices, and rectangular selection",
    "level": "01 / INDEXING & OWNERSHIP",
    "objective": "Select from a fixed [4,5] array while preserving the difference between a basic-slice view and advanced-index copies.",
    "scaffold": "import numpy as np\n\ndef select_regions(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n    if a.shape != (4, 5):\n        raise ValueError(\"This fixture requires shape (4,5)\")\n    sliced = __SLOT_0__\n    paired = __SLOT_1__\n    rectangle = __SLOT_2__\n    return sliced, paired, rectangle",
    "slots": [
      {"label": "Rows 1 and 3, columns 1 and 3, with basic slices", "placeholder": "a[start:stop:step, ...]", "answers": ["a[1:4:2, 1:5:2]", "a[1::2, 1::2]"], "hint": "Stops are exclusive. Both dimensions use basic slicing, which shares storage here."},
      {"label": "Select coordinate pairs (0,1) and (3,4)", "placeholder": "a[[...], [...]]", "answers": ["a[[0, 3], [1, 4]]"], "hint": "Equal-shaped integer index arrays are paired elementwise, not expanded into a grid."},
      {"label": "Select every combination of rows [0,3] and columns [1,4]", "placeholder": "a[np.ix_(...)]", "answers": ["a[np.ix_([0, 3], [1, 4])]"], "hint": "ix_ creates broadcastable index arrays for an outer rectangular selection."}
    ],
    "expected": "For arange(20).reshape(4,5): slice [[6,8],[16,18]], paired [1,19], rectangle [[1,4],[16,19]]. Only the basic slice shares array storage.",
    "explanation": "Basic slicing and advanced indexing have different ownership semantics. Coordinate arrays must broadcast; lengths 3 and 2 do not broadcast as a pair. Use np.ix_ when you want the Cartesian product.",
    "tests": "a = np.arange(20).reshape(4, 5)\nsliced, paired, rectangle = select_regions(a)\nnp.testing.assert_array_equal(sliced, [[6, 8], [16, 18]])\nnp.testing.assert_array_equal(paired, [1, 19])\nnp.testing.assert_array_equal(rectangle, [[1, 4], [16, 19]])\nassert np.shares_memory(a, sliced)\nassert not np.shares_memory(a, paired)\nassert not np.shares_memory(a, rectangle)\nsliced[0, 0] = -9\nassert a[1, 1] == -9\nrectangle[0, 0] = -7\nassert a[0, 1] == 1\nprint('PASS: values, shapes, pairing, outer selection, view/copy behavior')",
    "lesson": "numpy"
  },
  {
    "id": "numpy-standardize",
    "title": "Standardize features with the correct reduction axis",
    "level": "02 / BROADCASTING & STATISTICS",
    "objective": "Fit per-feature training means/scales and preserve dimensions so [1,D] statistics broadcast over [N,D] samples.",
    "scaffold": "import numpy as np\n\ndef fit_standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:\n    x = np.asarray(x, dtype=np.float64)\n    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():\n        raise ValueError(\"Expected a finite nonempty [samples,features] array\")\n    mean = __SLOT_0__\n    scale = __SLOT_1__\n    scale = np.where(scale == 0, 1.0, scale)\n    standardized = __SLOT_2__\n    return standardized, mean, scale",
    "slots": [
      {"label": "Mean across samples, keeping a singleton sample axis", "placeholder": "x.mean(axis=..., keepdims=True)", "answers": ["x.mean(axis=0, keepdims=True)", "np.mean(x, axis=0, keepdims=True)"], "hint": "Rows are samples. Reduce axis 0 to obtain one statistic per feature."},
      {"label": "Population standard deviation with the same shape", "placeholder": "x.std(...)", "answers": ["x.std(axis=0, keepdims=True)", "np.std(x, axis=0, keepdims=True)"], "hint": "This exercise uses ddof=0. Constant features intentionally use scale 1 in the given policy."},
      {"label": "Center and scale without repeating statistics", "placeholder": "(x - ...) / ...", "answers": ["(x - mean) / scale"], "hint": "Broadcast [1,D] across [N,D]; do not tile the statistics into a full matrix."}
    ],
    "expected": "For [[1,2,7],[3,4,7],[5,6,7]], mean is [[3,4,7]]. The constant third feature becomes zero; other standardized columns have mean 0 and population std 1.",
    "explanation": "Constant-feature handling is an explicit policy, not an accidental divide-by-zero fallback. Fit statistics only on training data and reuse them for validation/test; recomputing on held-out data leaks information.",
    "tests": "x = np.array([[1., 2., 7.], [3., 4., 7.], [5., 6., 7.]])\noriginal = x.copy()\nz, mean, scale = fit_standardize(x)\nnp.testing.assert_allclose(mean, [[3, 4, 7]])\nassert mean.shape == scale.shape == (1, 3)\nnp.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-12)\nnp.testing.assert_allclose(z[:, :2].std(axis=0), 1)\nnp.testing.assert_array_equal(z[:, 2], 0)\nnp.testing.assert_array_equal(x, original)\nfor invalid in [np.empty((0, 3)), np.array([[np.nan]])]:\n    try:\n        fit_standardize(invalid)\n    except ValueError:\n        pass\n    else:\n        raise AssertionError('Invalid data must fail')\nprint('PASS: axis, shape, scaling, constant feature, unchanged input, invalid data')",
    "lesson": "numpy"
  },
  {
    "id": "numpy-softmax",
    "title": "Compute stable row probabilities in NumPy",
    "level": "03 / UFUNCS & NUMERICS",
    "objective": "Use reductions, broadcasting, and exp without overflowing on large finite logits.",
    "scaffold": "import numpy as np\n\ndef row_softmax(logits: np.ndarray) -> np.ndarray:\n    logits = np.asarray(logits, dtype=np.float64)\n    if logits.ndim < 1 or 0 in logits.shape or not np.isfinite(logits).all():\n        raise ValueError(\"Expected finite, nonempty logits\")\n    shifted = __SLOT_0__\n    weights = __SLOT_1__\n    return __SLOT_2__",
    "slots": [
      {"label": "Subtract each row's maximum", "placeholder": "logits - logits.max(...)", "answers": ["logits - logits.max(axis=-1, keepdims=True)", "logits - np.max(logits, axis=-1, keepdims=True)"], "hint": "Normalize the last axis independently; preserve it for broadcasting."},
      {"label": "Apply the exponential ufunc", "placeholder": "np....(shifted)", "answers": ["np.exp(shifted)"], "hint": "exp is an elementwise ufunc; np.vectorize is unnecessary."},
      {"label": "Divide by each row's sum", "placeholder": "weights / weights.sum(...)", "answers": ["weights / weights.sum(axis=-1, keepdims=True)"], "hint": "The last-axis sum must keep its dimension."}
    ],
    "expected": "Rows [1000,1001,1002] and [-1000,-999,-998] yield the same probabilities, approximately [0.0900,0.2447,0.6652].",
    "explanation": "This exercise requires finite, nonempty inputs and rejects masked infinities. Attention with forbidden keys needs a separate mask/all-masked-row contract. NumPy computes values, not an automatic gradient graph.",
    "tests": "x = np.array([[1000., 1001., 1002.], [-1000., -999., -998.]])\np = row_softmax(x)\nassert np.isfinite(p).all()\nnp.testing.assert_allclose(p.sum(axis=-1), 1)\nnp.testing.assert_allclose(p[0], p[1])\nnp.testing.assert_allclose(p[0], [0.0900305732, 0.2447284711, 0.6652409558])\nnp.testing.assert_allclose(row_softmax(np.zeros((2, 3, 4))), 0.25)\nfor invalid in [np.empty((0, 2)), np.array([np.nan])]:\n    try:\n        row_softmax(invalid)\n    except ValueError:\n        pass\n    else:\n        raise AssertionError('Invalid logits must fail')\nprint('PASS: large logits, shift invariance, normalization, batch axes, invalid inputs')",
    "lesson": "numpy"
  },
  {
    "id": "numpy-distances",
    "title": "Vectorize pairwise squared distances",
    "level": "04 / VECTORIZATION & MEMORY",
    "objective": "Map [N,D] points and [K,D] centers to [N,K] distances while tracking the [N,K,D] intermediate.",
    "scaffold": "import numpy as np\n\ndef squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:\n    x = np.asarray(x, dtype=np.float64)\n    centers = np.asarray(centers, dtype=np.float64)\n    if (x.ndim != 2 or centers.ndim != 2 or 0 in x.shape or 0 in centers.shape\n            or x.shape[1] != centers.shape[1]\n            or not np.isfinite(x).all() or not np.isfinite(centers).all()):\n        raise ValueError(\"Expected finite nonempty [N,D] and [K,D] arrays\")\n    difference = __SLOT_0__\n    return __SLOT_1__",
    "slots": [
      {"label": "Create point-center differences", "placeholder": "x[:, None, :] - ...", "answers": ["x[:, None, :] - centers[None, :, :]"], "hint": "[N,1,D] minus [1,K,D] broadcasts to [N,K,D]."},
      {"label": "Sum squared differences over features", "placeholder": "np.sum(..., axis=...)", "answers": ["np.sum(difference ** 2, axis=-1)", "(difference ** 2).sum(axis=-1)"], "hint": "Reduce D, not N or K."}
    ],
    "expected": "Points [[0,0],[3,4]] and centers [[0,0],[0,4]] produce [[0,16],[25,9]].",
    "explanation": "Vectorization removes Python loops, not memory costs: the difference array has N*K*D elements. Chunk large problems or consider the norm/matmul identity with cancellation safeguards. Float64 conversion avoids integer arithmetic overflow for these fixtures, not arbitrary floating-point overflow.",
    "tests": "x = np.array([[0, 0], [3, 4]], dtype=np.int16)\nc = np.array([[0, 0], [0, 4]], dtype=np.int16)\nd = squared_distances(x, c)\nnp.testing.assert_array_equal(d, [[0, 16], [25, 9]])\nnp.testing.assert_array_equal(np.diag(squared_distances(x, x)), 0)\nassert d.shape == (2, 2) and d.dtype == np.float64\ntry:\n    squared_distances(x, np.ones((3, 5)))\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Feature dimensions must match')\nprint('PASS: distances, axes, dtype, self-distance, dimension mismatch')",
    "lesson": "numpy"
  },
  {
    "id": "numpy-lstsq",
    "title": "Fit a line without explicitly inverting a matrix",
    "level": "05 / LINEAR ALGEBRA",
    "objective": "Construct a design matrix with an intercept and use a least-squares solver, including a rank-deficient fixture.",
    "scaffold": "import numpy as np\n\ndef fit_line(x: np.ndarray, y: np.ndarray) -> np.ndarray:\n    x = np.asarray(x, dtype=np.float64)\n    y = np.asarray(y, dtype=np.float64)\n    if (x.ndim != 1 or y.shape != x.shape or x.size < 2\n            or not np.isfinite(x).all() or not np.isfinite(y).all()):\n        raise ValueError(\"Expected matching finite vectors with at least two samples\")\n    design = __SLOT_0__\n    coefficients = __SLOT_1__\n    return coefficients",
    "slots": [
      {"label": "Create [intercept column, x column]", "placeholder": "np.column_stack(...)", "answers": ["np.column_stack([np.ones_like(x), x])", "np.column_stack((np.ones_like(x), x))"], "hint": "The coefficient order is intercept, then slope."},
      {"label": "Solve least squares and select the coefficient vector", "placeholder": "np.linalg.lstsq(...)[...]", "answers": ["np.linalg.lstsq(design, y, rcond=None)[0]"], "hint": "lstsq returns coefficients plus diagnostics; do not form inverse(X.T@X)."}
    ],
    "expected": "For x=[0,1,2] and y=[1,3,5], coefficients are [1,2]. With x=[1,1,1], y=[2,2,2], a minimum-norm solution still predicts 2.",
    "explanation": "Explicit inversion of the normal equations is unnecessary and can worsen conditioning. lstsq handles rectangular/rank-deficient designs; a square nonsingular system may instead use solve. Inspect rank, residuals and singular values in a real analysis.",
    "tests": "np.testing.assert_allclose(fit_line(np.array([0, 1, 2]), np.array([1, 3, 5])), [1, 2], atol=1e-12)\ncoefficients = fit_line(np.ones(3), np.full(3, 2.))\nnp.testing.assert_allclose(np.column_stack([np.ones(3), np.ones(3)]) @ coefficients, 2, atol=1e-12)\nnp.testing.assert_allclose(coefficients, [1, 1], atol=1e-12)\ntry:\n    fit_line(np.array([1, 2]), np.array([1]))\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Mismatched targets must fail')\nprint('PASS: intercept/slope, prediction, rank deficiency, invalid shape')",
    "lesson": "numpy"
  },
  {
    "id": "numpy-rng",
    "title": "Use a local random generator without changing global state",
    "level": "06 / RANDOMNESS & REPRODUCIBILITY",
    "objective": "Generate reproducible integer samples with a local Generator and explicit interval bounds.",
    "scaffold": "import numpy as np\n\ndef sample_ids(seed: int, population: int, size: int) -> np.ndarray:\n    if population <= 0 or size < 0:\n        raise ValueError(\"population must be positive and size nonnegative\")\n    rng = __SLOT_0__\n    return __SLOT_1__",
    "slots": [
      {"label": "Create an independent generator", "placeholder": "np.random....(seed)", "answers": ["np.random.default_rng(seed)"], "hint": "Avoid np.random.seed, which changes legacy global state."},
      {"label": "Sample integers in [0,population)", "placeholder": "rng.integers(...)", "answers": ["rng.integers(0, population, size=size)"], "hint": "The upper bound is exclusive. Sampling here is with replacement."}
    ],
    "expected": "The same seed/configuration repeats the same sequence in a fixed compatible environment. Results have shape [size] and each value lies in [0,population).",
    "explanation": "A seed is part of reproducibility, not a universal cross-version/hardware guarantee. Record generator, versions, call order and data order. Independent generators avoid disturbing randomness elsewhere in a program.",
    "tests": "before = np.random.get_state()\na = sample_ids(23, 100, 12)\nb = sample_ids(23, 100, 12)\nnp.testing.assert_array_equal(a, b)\nassert a.shape == (12,) and np.issubdtype(a.dtype, np.integer)\nassert ((a >= 0) & (a < 100)).all()\nafter = np.random.get_state()\nassert all(np.array_equal(left, right) for left, right in zip(before, after))\nassert sample_ids(23, 100, 0).shape == (0,)\ntry:\n    sample_ids(23, 0, 1)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Invalid population must fail')\nprint('PASS: local reproducibility, bounds, global-state isolation, empty/invalid sizes')",
    "lesson": "numpy"
  },
  {
    "id": "numpy-einsum",
    "title": "Contract features without mixing batch dimensions",
    "level": "07 / ADVANCED TENSOR AXES",
    "objective": "Produce [B,T,U] dot products from Q[B,T,D] and K[B,U,D], keeping batch and sequence roles distinct.",
    "scaffold": "import numpy as np\n\ndef batched_scores(q: np.ndarray, k: np.ndarray) -> np.ndarray:\n    if (q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0]\n            or q.shape[2] != k.shape[2]\n            or not np.isfinite(q).all() or not np.isfinite(k).all()):\n        raise ValueError(\"Expected finite compatible [B,T,D] and [B,U,D] arrays\")\n    return __SLOT_0__",
    "slots": [
      {"label": "Sum D while preserving B, T, and U", "placeholder": "np.einsum(...)", "answers": ["np.einsum(\"btd,bud->btu\", q, k)", "np.einsum('btd,bud->btu', q, k)", "q @ k.swapaxes(-1, -2)"], "hint": "b appears on both inputs and the output; d is the only contracted axis."}
    ],
    "expected": "Q shaped [2,2,3] and K shaped [2,3,3] produce [2,2,3], with no cross-batch comparisons.",
    "explanation": "einsum specifies axes explicitly; it is not automatically faster than matmul. Array.T reverses every axis for an n-D array, so use swapaxes on the last two dimensions when that is the intended batched transpose. Numerical accuracy still depends on dtype.",
    "tests": "q = np.arange(12., dtype=np.float64).reshape(2, 2, 3)\nk = np.arange(18., dtype=np.float64).reshape(2, 3, 3)\nresult = batched_scores(q, k)\nreference = (q[:, :, None, :] * k[:, None, :, :]).sum(axis=-1)\nassert result.shape == (2, 2, 3)\nnp.testing.assert_allclose(result, reference)\nchanged = k.copy()\nchanged[1] += 100\nnp.testing.assert_allclose(batched_scores(q, changed)[0], result[0])\ntry:\n    batched_scores(q, np.ones((3, 3, 3)))\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Batch mismatch must fail')\nprint('PASS: axis contraction, independent reference, batch isolation, invalid shapes')",
    "lesson": "numpy"
  },
  {
    "id": "numpy-impute",
    "title": "Fit NaN-aware medians without hiding invalid columns",
    "level": "08 / PREPROCESSING CONTRACTS",
    "objective": "Impute training NaNs using per-column medians while rejecting infinity and all-missing features.",
    "scaffold": "import numpy as np\n\ndef fit_impute(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:\n    x = np.asarray(x, dtype=np.float64)\n    if x.ndim != 2 or 0 in x.shape or np.isinf(x).any():\n        raise ValueError(\"Expected nonempty [N,D] data without infinity\")\n    if np.isnan(x).all(axis=0).any():\n        raise ValueError(\"Cannot fit a median for an all-missing feature\")\n    medians = __SLOT_0__\n    filled = __SLOT_1__\n    return filled, medians",
    "slots": [
      {"label": "Fit one NaN-aware training median per feature", "placeholder": "np.nanmedian(...)", "answers": ["np.nanmedian(x, axis=0)"], "hint": "Reduce the sample axis; reject all-missing columns before reduction."},
      {"label": "Replace only NaNs, broadcasting medians across rows", "placeholder": "np.where(...)", "answers": ["np.where(np.isnan(x), medians, x)"], "hint": "Use np.isnan, not x == np.nan. Preserve observed entries and return new storage."}
    ],
    "expected": "For [[1,NaN,4],[3,6,NaN],[NaN,10,8]], medians are [2,8,6] and filled data is [[1,8,4],[3,6,6],[2,10,8]].",
    "explanation": "NaN-aware functions do not decide a safe policy for an all-missing feature. This exercise fails explicitly rather than inventing a value. Fit only on training data; reuse the fitted medians for held-out transformations.",
    "tests": "x = np.array([[1., np.nan, 4.], [3., 6., np.nan], [np.nan, 10., 8.]])\noriginal = x.copy()\nfilled, medians = fit_impute(x)\nnp.testing.assert_array_equal(medians, [2, 8, 6])\nnp.testing.assert_array_equal(filled, [[1, 8, 4], [3, 6, 6], [2, 10, 8]])\nassert np.array_equal(x, original, equal_nan=True)\nassert not np.shares_memory(x, filled)\nfor invalid in [np.array([[np.nan], [np.nan]]), np.array([[np.inf]])]:\n    try:\n        fit_impute(invalid)\n    except ValueError:\n        pass\n    else:\n        raise AssertionError('Unusable feature values must fail')\nprint('PASS: medians, broadcasting, unchanged data, independent output, explicit errors')",
    "lesson": "numpy"
  }
];
