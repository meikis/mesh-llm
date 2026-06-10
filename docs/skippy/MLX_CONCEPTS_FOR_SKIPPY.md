# MLX Concepts To Steal For Skippy

This plan captures useful distributed-parallelism ideas from MLX and translates
them into Skippy split-serving work. The goal is not to add an MLX backend or
to implement tensor parallelism in Skippy. The goal is to improve Skippy's
existing layer pipeline by borrowing the parts of MLX that map cleanly onto
GGUF stage serving.

Reference source snapshot:

| Repo | Commit | Relevant surface |
| --- | --- | --- |
| `ml-explore/mlx` | `968d264f2903d578e699c4452a4dbf48633921aa` | distributed ops, Ring transport, launch docs |
| `ml-explore/mlx-lm` | `df48987708fc90b8fac72fb6db7538c5dbb1077d` | `sharded_load`, pipeline mixin, server request sync |

## Boundary

In scope:

- transport-aware Skippy stage ordering;
- large activation-frame transfer improvements;
- package/artifact planning improvements inspired by metadata-first loading;
- readout-stage topology experiments inspired by MLX rank ordering;
- request and cache coordination ideas that apply to Skippy stage sessions.

Out of scope:

- adding an MLX serving backend;
- reimplementing MLX tensor parallelism;
- adding all-reduce collectives to Skippy decode;
- changing the public OpenAI frontend ownership away from `openai-frontend`;
- replacing GGUF layer packages with safetensors.

Tensor parallelism can stay as research context. It is not a Skippy work item
until there is a concrete llama.cpp staged-runtime design for tensor-sharded
weights, collectives, KV layout, correctness certification, and mixed-version
mesh behavior.

## What MLX Does That Matters

### Pipeline Rank Ordering

MLX pipeline mode assigns rank `0` to the final layer range and the highest rank
to the first layer range. Forward execution receives from `rank + 1`, computes
local layers, sends to `rank - 1`, then gathers the final hidden state. This is
the inverse of Skippy's public-facing mental model, where stage `0` owns the
first layers and the OpenAI-facing driver.

Skippy should not blindly reverse stage numbering, because stage `0` also owns
tokenization, session control, and public routing. The useful idea is to treat
the readout/final layer owner as a first-class topology role rather than only
the tail of a chain.

### Neighbor-Only Pipeline Shape

MLX Ring over TCP only supports `send`/`recv` between direct neighbors. Pipeline
models are shaped to match that constraint: activation flow is adjacent-rank
only. Skippy already uses an activation chain, but the planner currently has
room to become more explicit about ordered edge quality.

The useful idea is not the ring itself; it is the discipline of making the
pipeline topology match what the transport is good at.

### Metadata-First Artifact Loading

MLX `sharded_load` downloads non-weight metadata first, lazy-builds the model,
applies pipeline pruning, then uses `model.safetensors.index.json` to fetch only
the local rank's weight files. Skippy layer packages already provide per-layer
artifacts, but the same two-phase planning shape is worth making explicit:

1. resolve package metadata and layer inventory;
2. plan stages using local cache and peer transfer signals;
3. fetch only the artifacts needed for the accepted topology;
4. publish routability only after the selected artifacts and stages are ready.

### Multi-Connection Large Transfers

MLX Ring can stripe large payloads over multiple neighbor sockets. Decode
activations are small enough that striping would add complexity without much
gain, but prefill activation frames can be large and dominate total transfer.

Skippy should evaluate striping only for large prefill activation frames and
artifact transfer, with decode left on the simple single-frame path.

### Rank-0 Request Broadcast

The MLX HTTP server lets rank `0` dequeue requests, then broadcasts serialized
request data to all ranks through collectives so every process participates in
the same generation loop. Skippy stages already receive typed stage messages,
so this does not map directly. The useful idea is synchronized request epochs:
every stage should agree on request/session/generation boundaries before work is
allowed to overlap aggressively.

## Workstream 1: Transport-Aware Stage Ordering

Objective: make the Skippy topology planner score ordered stage edges, not just
individual nodes.

Implementation tasks:

1. Extend topology inputs with pairwise edge measurements:
   - RTT;
   - sustained large-frame throughput when available;
   - optional operator-provided fabric label such as `lan`, `thunderbolt`, or
     `wan`;
   - whether direct prediction return is available for the edge from final
     stage to driver-facing stage.
2. Add an edge-cost model in `skippy-topology`:
   - decode edge cost: latency dominated;
   - prefill edge cost: bytes divided by measured or estimated throughput;
   - artifact edge cost: cold-path transfer cost, not decode hot path.
3. Make candidate plans order-aware:
   - choose stage permutations that minimize serialized decode edge cost;
   - preserve memory and family-boundary constraints;
   - keep fewer physical stages preferred when they satisfy residency.
4. Emit diagnostics:
   - chosen ordered edges;
   - rejected lower-cost orders and the reason they failed;
   - estimated decode network floor;
   - estimated prefill transfer pressure.

Acceptance evidence:

- `skippy-topology` unit tests show the same node set can be ordered
  differently when edge costs differ.
- Split doctor reports ordered stage edges and their edge-cost reasons.
- A two-node and four-node split still produce the same model outputs as the
  current planner under identical boundaries.

## Workstream 2: Large Prefill Activation-Frame Striping

Objective: reduce prefill transfer time for large activation frames without
making decode more fragile.

Implementation tasks:

1. Add a negotiated stage capability for striped activation frames.
2. Keep the existing single-frame protocol as the compatibility default.
3. Add a new large-frame path for prefill only:
   - split frames above a configurable byte threshold;
   - send chunks over multiple streams or connections;
   - reassemble by request id, session id, frame sequence, and chunk index;
   - enforce max decoded activation bytes before allocation.
4. Keep decode activations single-path unless measurements prove otherwise.
5. Add backpressure:
   - per-request maximum in-flight chunks;
   - per-stage maximum buffered reassembly bytes;
   - timeout and cleanup for incomplete frames.
6. Add telemetry:
   - striped frame count;
   - chunk count;
   - reassembly wait time;
   - bytes sent per stream;
   - fallback count to single-frame mode.

Acceptance evidence:

- `skippy-protocol` tests cover chunk header parsing, out-of-order chunks,
  duplicate chunks, timeout cleanup, and max-size rejection.
- `skippy-server` tests cover single-frame fallback when a peer does not
  advertise striping.
- `skippy-bench` can compare no striping versus striping on a long-prompt
  corpus and report TTFT plus per-stage forward-write time.

## Workstream 3: Metadata-First Package Planning

Objective: make Skippy's package resolution explicitly two-phase and avoid
downloading or materializing artifacts before a topology is accepted.

Implementation tasks:

1. Separate package metadata resolution from artifact materialization:
   - manifest;
   - layer inventory;
   - tokenizer/projector sidecar inventory;
   - artifact sizes and hashes.
2. Feed artifact presence into topology planning:
   - local cached bytes;
   - missing bytes;
   - peer-transfer eligibility;
   - Hugging Face fallback eligibility.
3. After the topology is accepted, materialize only selected stage artifacts.
4. Preserve original model layer indexes in every cache key and diagnostic.
5. Report cold-start plan cost:
   - local bytes already present;
   - peer-transfer bytes;
   - remote-download bytes;
   - expected materialized stage bytes.

Acceptance evidence:

- Package-only certification can run metadata planning without materializing
  unselected stage artifacts.
- Split doctor shows why a node was selected when it had a better cached slice
  despite lower raw availability score.
- Materialized-stage cache cleanup still treats stage GGUFs as derived cache.

## Workstream 4: Readout-Stage Topology Experiment

Objective: test whether making the final/readout stage a first-class role can
reduce decode hot-path latency or simplify direct prediction return.

Candidate designs:

| Design | Shape | Risk |
| --- | --- | --- |
| Current Skippy | stage `0` owns first layers and public driver; final stage returns prediction directly to stage `0` | Known path; direct return still pays final-to-stage0 hop |
| MLX-like rank role | driver-facing process coordinates, but rank/readout `0` owns final layers | Requires careful session/control separation |
| Split driver/readout | driver-facing stage owns OpenAI/session control; readout stage owns logits and direct client token return path | More protocol work; clearer role separation |

Implementation tasks:

1. Add topology role names distinct from numeric stage index:
   - `driver`;
   - `embedding`;
   - `intermediate`;
   - `readout`.
2. Teach diagnostics to print roles and layer ranges separately.
3. Build a benchmark-only prototype before changing production routing:
   - same layers and peers as the current direct-return path;
   - compare decode network floor, TTFT, and total tok/s;
   - record failure and cancellation behavior.
4. Decide whether the role model belongs in production topology planning.

Acceptance evidence:

- Benchmark report compares current direct-return topology with the role-based
  prototype on the same hosts and model.
- The prototype preserves OpenAI streaming order and cancellation behavior.
- If results do not beat current generation 3 direct return, close this as an
  experiment and keep only the role terminology if it improved diagnostics.

## Workstream 5: Request And Cache Epoch Coordination

Objective: borrow MLX's synchronized-request discipline without replacing
Skippy's typed stage protocol.

Implementation tasks:

1. Add explicit request epoch fields to stage diagnostics:
   - request id;
   - session id;
   - generation/config epoch;
   - cache checkpoint generation.
2. Assert that overlapped prefill, restore, trim, and decode frames cannot cross
   incompatible epochs.
3. Add a stage-local debug mode that records the last N epoch transitions per
   session.
4. Use epoch diagnostics in split doctor bundles when cache or cancellation
   errors occur.

Acceptance evidence:

- Tests reject stale restore/trim messages that target an older checkpoint
  generation when newer decode work has started.
- A mini-agent tool-call loop with cache reuse still passes with epoch debug
  enabled.

## Do Not Do Yet: Tensor Parallelism

MLX tensor parallelism shards every transformer block and uses all-reduce in the
hot path. That is a different runtime architecture from Skippy's stage pipeline.
For Skippy, tensor parallelism would require:

- tensor-sharded GGUF loading;
- llama.cpp graph changes for collectives;
- KV-cache layout changes;
- per-layer all-reduce transport;
- new correctness certification;
- new mixed-version mesh compatibility rules.

Do not start this work under the MLX concept-stealing effort. If tensor
parallelism becomes a goal later, it should have its own design document and
explicit protocol-compatibility review.

## Suggested Order

1. Transport-aware stage ordering.
2. Metadata-first package planning.
3. Large prefill activation-frame striping.
4. Request/cache epoch diagnostics.
5. Readout-stage role experiment.

This order improves the existing Skippy path first, keeps protocol risk
contained, and avoids committing to a new backend or tensor-parallel runtime
before the pipeline has absorbed the low-risk MLX lessons.
