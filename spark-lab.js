"use strict";
const SPARK_LAB = [
  {
    "id": "spark-aggregate",
    "title": "Filter and aggregate with native expressions",
    "level": "01 / DATAFRAME BASICS",
    "objective": "Keep positive, non-null amounts and total them by user without collecting raw rows to Python.",
    "scaffold": "from pyspark.sql import functions as F\n\ndef user_totals(events):\n    valid = events.filter(__SLOT_0__)\n    return valid.groupBy(\"user_id\").agg(__SLOT_1__)",
    "slots": [
      {"label": "Keep positive amounts", "placeholder": "F.col(...)", "answers": ["F.col(\"amount\") > 0", "F.col('amount') > 0", "(F.col(\"amount\").isNotNull()) & (F.col(\"amount\") > 0)"], "hint": "SQL comparison with null is not true, so filtering amount > 0 also excludes nulls."},
      {"label": "Sum amounts with a stable output name", "placeholder": "F.sum(...).alias(...)", "answers": ["F.sum(\"amount\").alias(\"total\")", "F.sum('amount').alias('total')", "F.sum(F.col(\"amount\")).alias(\"total\")"], "hint": "Use a native aggregate rather than a Python loop over collected rows."}
    ],
    "expected": "For u1 amounts [10, -2, null, 5] and u2 amount [7], output totals are u1=15 and u2=7. Row ordering is unspecified.",
    "explanation": "DataFrame expressions remain in Spark's query plan. Filtering null/negative values follows an explicit business rule. The tiny test collects only a bounded two-row fixture; production results should not be collected wholesale.",
    "tests": "events = spark.createDataFrame([('u1', 10), ('u1', -2), ('u1', None), ('u1', 5), ('u2', 7)], 'user_id string, amount long')\nresult = {row.user_id: row.total for row in user_totals(events).collect()}\nassert result == {'u1': 15, 'u2': 7}\nassert set(user_totals(events).columns) == {'user_id', 'total'}\nprint('PASS: native filter, null handling, aggregation, output schema')",
    "lesson": "spark"
  },
  {
    "id": "spark-join",
    "title": "Preserve the left-side grain",
    "level": "02 / JOIN CORRECTNESS",
    "objective": "Join a small unique user dimension to events, preserve unmatched events, and keep the intended row count.",
    "scaffold": "from pyspark.sql import functions as F\n\ndef attach_segment(events, users):\n    # Contract: users contains at most one row per non-null user_id.\n    return events.join(__SLOT_0__, on=\"user_id\", how=__SLOT_1__)",
    "slots": [
      {"label": "Hint that this tiny test dimension can be broadcast", "placeholder": "F....(users)", "answers": ["F.broadcast(users)"], "hint": "The fixture is small. A production broadcast decision needs a measured size and memory budget."},
      {"label": "Preserve every event, including unmatched users", "placeholder": "A join-type string", "answers": ["\"left\"", "'left'", "\"left_outer\"", "'left_outer'"], "hint": "An inner join would discard events with no matching dimension record."}
    ],
    "expected": "Three input events remain three output rows. u3 has a null segment because it does not exist in the unique user dimension.",
    "explanation": "Broadcasting changes a possible physical execution strategy, not join semantics. Uniqueness is an explicit input contract; the separate test demonstrates row multiplication if it is violated rather than silently dropping duplicates.",
    "tests": "events = spark.createDataFrame([(1, 'u1'), (2, 'u2'), (3, 'u3')], 'event_id long, user_id string')\nusers = spark.createDataFrame([('u1', 'A'), ('u2', 'B')], 'user_id string, segment string')\nrows = attach_segment(events, users).orderBy('event_id').collect()\nassert len(rows) == 3\nassert [row.segment for row in rows] == ['A', 'B', None]\nduplicates = users.unionByName(users.filter(F.col('user_id') == 'u1'))\nassert attach_segment(events, duplicates).count() == 4\nprint('PASS: left preservation, unmatched nulls, duplicate-key failure demonstration')",
    "lesson": "spark"
  },
  {
    "id": "spark-window",
    "title": "Choose the latest record deterministically",
    "level": "03 / WINDOW FUNCTIONS",
    "objective": "Keep one event per user with an explicit tie-breaker, not an arbitrary dropDuplicates selection.",
    "scaffold": "from pyspark.sql import Window\nfrom pyspark.sql import functions as F\n\ndef latest_per_user(events):\n    window = Window.partitionBy(\"user_id\").orderBy(__SLOT_0__)\n    ranked = events.withColumn(\"rn\", __SLOT_1__)\n    return ranked.filter(F.col(\"rn\") == 1).drop(\"rn\")",
    "slots": [
      {"label": "Order newest timestamp first, then largest unique event ID", "placeholder": "F.col(...).desc(), ...", "answers": ["F.col(\"event_time\").desc(), F.col(\"event_id\").desc()", "F.col('event_time').desc(), F.col('event_id').desc()"], "hint": "Timestamp ties need a stable secondary key. The fixture uses integer event times and unique event IDs."},
      {"label": "Number rows within the window", "placeholder": "F....().over(window)", "answers": ["F.row_number().over(window)"], "hint": "row_number assigns exactly one row number 1 within each nonempty group."}
    ],
    "expected": "u1 keeps event 3 rather than event 2 when timestamps tie; u2 keeps its only event.",
    "explanation": "A deterministic order makes repeated runs reproducible for this input contract. rank or dense_rank can retain multiple tied records; dropDuplicates does not specify which record wins. Define a null-ordering policy when production timestamps can be null.",
    "tests": "events = spark.createDataFrame([('u1', 1, 10), ('u1', 2, 20), ('u1', 3, 20), ('u2', 4, 5)], 'user_id string, event_id long, event_time long')\nresult = {row.user_id: row.event_id for row in latest_per_user(events).collect()}\nassert result == {'u1': 3, 'u2': 4}\nassert 'rn' not in latest_per_user(events).columns\nprint('PASS: per-user window, deterministic tie break, helper-column cleanup')",
    "lesson": "spark"
  },
  {
    "id": "spark-partition",
    "title": "Partition without changing the data",
    "level": "04 / EXECUTION & DATA PARITY",
    "objective": "Redistribute a bounded fixture by key while preserving every row and duplicate.",
    "scaffold": "def partition_by_user(events, partitions: int):\n    if partitions <= 0:\n        raise ValueError(\"partitions must be positive\")\n    return __SLOT_0__",
    "slots": [
      {"label": "Redistribute into the requested count by user_id", "placeholder": "events.repartition(...)", "answers": ["events.repartition(partitions, \"user_id\")", "events.repartition(partitions, 'user_id')"], "hint": "repartition can redistribute by a key and usually incurs a shuffle. coalesce is not a general keyed redistribution substitute."}
    ],
    "expected": "The result has the requested three partitions for this explicit repartition operation, and its multiset of rows equals the input, including duplicates.",
    "explanation": "Partitioning is an execution choice, not a deduplication rule or an ordering guarantee. Equal row counts alone are insufficient to establish parity; compare multiset differences on controlled data. Production parity checks should avoid unnecessarily collecting large data.",
    "tests": "events = spark.createDataFrame([('u1', 1), ('u1', 1), ('u2', 2)], 'user_id string, value long')\nresult = partition_by_user(events, 3)\nassert result.rdd.getNumPartitions() == 3\nassert events.exceptAll(result).count() == 0\nassert result.exceptAll(events).count() == 0\ntry:\n    partition_by_user(events, 0)\nexcept ValueError:\n    pass\nelse:\n    raise AssertionError('Invalid partition count must be rejected')\nprint('PASS: partition count, bidirectional multiset parity, invalid input')",
    "lesson": "spark"
  }
];
