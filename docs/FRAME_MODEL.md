# Qwen execution frame model

## Scope

This document defines the minimal execution model for Qwen work scheduled with Euhedral-Execution
0.0.7. The `prepare/embed` stage performs one synchronous Q3 token-embedding lookup into a
per-submission BF16 hidden-state buffer. Transformer layers, CUDA streams, graph capture, KV-cache
allocation, batching, sampling, and model-serving APIs remain out of scope.

The selected dependency is:

```text
io.euhedral-execution:euhedral-core:0.0.7
```

The dependency is currently an implementation dependency of the inference `core` module. It should
be promoted to an API dependency only if a later public inference API exposes Euhedral frame or
scheduler types directly.

## Verified Euhedral-Execution 0.0.7 primitives

The design is based on the published 0.0.7 source artifact, not on a newer checkout.

### `AbstractFrame`

`AbstractFrame` is the scheduled unit. It carries an immutable identity hash, a mutable routing hash,
an optional origin, an optional shared kill switch, and finalization hooks. `AbstractExecutor` checks
liveness, invokes the executor body, and then calls exactly one normal or error finalization path for
caught `Exception` values. Arbitrary JVM `Error` values are outside that boundary.

A frame is an ownership-bearing object, not merely a message. After it is recycled or republished,
the prior owner must not access it.

### `PipelineFrame`

`PipelineFrame<T>` supplies a typed reusable chain. Each transformation and the terminal consumer is
a distinct scheduled frame. A successor is published only after the current stage completes.

- `fanOut` permits a stage to route independently across workers.
- `fanIn` keeps a stage on stable ordered routing.
- Builder definitions are immutable and reusable.
- Filtering, cancellation, and failures terminate the whole chain.
- The root frame owns chain completion and recycling.

The chain is sequential for one input even when its stages use `fanOut`: only one stage of that chain
is eligible at a time. `fanOut` affects placement and concurrency between independent inputs; it does
not make transformer layers for one request concurrent.

### `PipelineRunner`

`PipelineRunner<I>` owns admission, a reusable frame manager, a partitioned ingest queue, and
accepted-chain accounting. `submit` provides a `CompletableFuture<PipelineFrame.Outcome>` with
`SUCCESS`, `FILTERED`, `CANCELLED`, or `FAILED` status. `run` avoids that future but does not expose an
end-to-end result signal.

The runner has one submission owner. Its lifecycle lock coordinates admission and close, but does
not make frame checkout multi-consumer safe. An inference facade that accepts concurrent callers
must therefore serialize admission onto one owner thread or maintain one runner per submission
owner. Callers must not invoke one runner concurrently.

`completeGracefully` closes admission and completes after all accepted chains finish. `complete`
closes admission, trips the runner-wide kill switch, and disconnects immediately. Queued cancellation
can be deferred while another thread owns the queue drain, and an executing body is not interrupted;
it observes cancellation cooperatively and finalizes after returning. The top-level lattice remains a
separate lifecycle owner.

### `FrameManager` and `FrameFactory`

`FrameManager` recycles frames through an MPSC return path with one checkout consumer. The
`FrameFactory` creates a frame on a pool miss and delegates reuse initialization to its caller-supplied
replacement callback. The factory does not reset arbitrary application fields. Every reusable frame
type must explicitly overwrite or clear all of its per-run state. Recycling is best effort; a rejected
recycler offer may allow the frame to become garbage collectable.

Application state must be cleared before recycler publication. A pooled execution frame must not own
persistent GPU allocations whose lifetime extends beyond one submission.

### `ControlPlaneLattice`

`ControlPlaneLattice` is the JVM-wide scheduler owner. Adding a runner as an upstream starts the
lattice lazily, waits for readiness, and attaches the source. The lattice owns worker/shard topology
and must be closed by the runtime owner. Queue empty, runner completion, and lattice termination are
different events.

## Selected model

Use a reusable `PipelineFrame<QwenExecutionContext>` chain created through a `PipelineRunner`. Do not
introduce a Qwen-specific `AbstractFrame` subclass initially.

One pipeline submission represents one inference quantum for one sequence:

- a prompt chunk during prefill; or
- one target-model decode step, possibly over a small token block when verification is added.

A whole request is not one long-running frame. Persistent sequence state survives across submissions,
while each submission carries an exclusive lease on that state until its chain reaches a terminal
outcome.

```text
caller or admission owner
  -> QwenExecutionContext
  -> PipelineRunner.submit
  -> prepare and embed
  -> transformer layer 0
  -> transformer layer 1
  -> ...
  -> transformer layer N - 1
  -> final norm and LM head
  -> terminal result capture
  -> PipelineFrame.Outcome
```

Do not create separate scheduling frames for RMS normalization, attention projections, GDN
projections, individual experts, or individual CUDA kernels. Those are operations within one layer
stage. This keeps scheduling granularity at a meaningful model boundary and avoids queueing dozens of
small dependent tasks per layer.

## Runtime objects

The execution skeleton is implemented in
`core/src/main/java/io/euhedral_execution/inference/core/scheduling`. The first real GPU operation
is the synchronous Q3 embedding lookup; transformer-layer execution and remaining runtime policies
are intentionally deferred.

### `QwenExecutionPlan`

An immutable plan is built once after `QwenWeights` has loaded. It owns no request state. Because
`QwenWeights` exposes arrays, the implementation snapshots the layer array and layer-type configuration
once during construction. The plan retains that model snapshot for its lifetime and exposes layer
weights only through an immutable list view. The plan captures:

- the loaded `QwenWeights`;
- a non-owning `QwenExecutionGpu` reference for request allocations and synchronous embedding;
- one preselected layer operation per `QwenLayerWeights` entry;
- the reusable base pipeline builder.

The plan does not own request workspaces or sequence state. Each context allocates and releases its own
hidden-state workspace while the plan retains model weights for the model lifetime.

Layer selection occurs during plan construction, not on every frame. The plan binds each layer index to
attention or GDN from the already assembled weight records. Dense and MoE FFN data remains part of the
same layer's loaded weights but is not executed by this skeleton. This avoids re-reading model topology
in the scheduled path.

### `QwenSequenceState`

Sequence state persists across prefill and decode submissions. It is owned by the request coordinator
except while a submitted chain holds its exclusive execution lease. It contains only state needed
across inference quanta:

- sequence identity;
- current logical token position;
- generated-token history or an external token-history handle;
- KV-cache ownership;
- any persistent recurrent GDN state;
- request-local cancellation state; and
- terminal request status.

GPU addresses remain opaque `long` values. They are never represented as dereferenceable Java memory
segments. Sequence state owns KV and recurrent allocations until request completion; the pooled frame
chain does not.

Only one chain may mutate a `QwenSequenceState` at a time. The chain receives an unforgeable execution
lease, which is required for KV/recurrent placeholder writes and for releasing the token-position lease.
Cancellation remains a coordinator request that may target the active chain. A later decode submission is
admitted only after the prior outcome has published. Parallelism is across independent sequence states,
not across dependent layers of one sequence.

The state is published as one immutable snapshot through an `AtomicReference`. Claims, cancellation,
terminal transitions, and lease-protected mutations use compare-and-set loops; no intrinsic monitor or
blocking lock is used. This makes the multi-field transitions linearizable while retaining lock-free
progress under contention. Lease release resolves a concurrent cancellation request in the same CAS that
releases ownership, so cancellation cannot slip between a separate check and token-position publication.

### `QwenExecutionContext`

The context is a per-submission mutable envelope. It is not pooled independently from its enclosing
pipeline chain in the minimal design. It references:

- the immutable execution plan;
- the exclusively leased sequence state;
- execution kind (`PREFILL` or `DECODE`);
- a copied array of input token IDs and the starting position;
- the context-owned hidden-state workspace, allocated before embedding and retained through the terminal
  consumer;
- current trace state; and
- a placeholder result record until transformer-layer execution produces logits.

The workspace contains only the BF16 hidden-state buffer sized as token count times
`QwenConfig.hiddenSize()`. The temporary device token-ID buffer is freed by the preparation stage after
the synchronous embedding call. The context must not own model weights or sequence-lifetime state.
The `QwenExecutionRunner` closes the workspace after terminal outcome publication on success,
cancellation, or failure. If a later implementation stores a context
inside a recycled custom frame, every per-run reference and result field must be overwritten or
cleared before reuse. A runner verifies that the context belongs to its plan before claiming sequence
state, so a context cannot be run through a different model topology.

### `QwenExecutionRunner`

The skeleton wraps Euhedral's `PipelineRunner` with a submission-owner adapter. The adapter preserves
the underlying `PipelineRunner` for scheduling, but binds sequence-lease release to the returned
`PipelineFrame.Outcome` future. This keeps a successor from claiming the sequence between terminal
callback execution and outcome publication.

## Pipeline construction

Build the immutable pipeline after model loading:

1. A single preparation/embedding stage validates the context and compact token-embedding descriptor,
   checks token IDs and cancellation, allocates the hidden-state workspace, uploads temporary token IDs,
   launches Q3 lookup against the GPU-resident embedding weights, synchronizes, and leaves the hidden
   state resident for its terminal consumer.
2. Add one stage for every `QwenLayerWeights` entry in index order. The stage closure captures the
   preselected layer operation from `QwenExecutionPlan`.
3. Add one final stage for final normalization and the LM head.
4. Use the terminal consumer to capture the completed placeholder result while retaining the sequence
   lease. `QwenExecutionRunner` releases or terminalizes that lease only after Euhedral publishes the
   raw `PipelineFrame.Outcome`.

The initial placement should use `fanOut` for computational stages. Dependencies within a chain remain
sequential, while independent sequences may use different workers. In 0.0.7, the recycler-backed root
is routed separately from `routeChain`: its constructor marks an unordered root and `FrameFactory`
randomizes that root on creation and replacement. Later stages are randomized by `routeChain`. Thus a
first stage declared with `fanOut` is still unordered, although its routing setup follows a different
path from successor stages. `fanIn` should be introduced only for a measured or correctness-driven
serialization requirement, such as a resource that truly has one owner. It must not be used merely to
express transformer layer order; the chain already expresses that order.

The admission adapter calls `QwenExecutionRunner.submit(context)`, which delegates to
`PipelineRunner.submit(context)`. The returned outcome future is the authoritative chain-completion
signal and the point at which the sequence lease is released. A nonblocking continuation maps:

- `SUCCESS` to the result stored by the terminal consumer;
- `CANCELLED` to request cancellation;
- `FILTERED` to an explicit rejected/omitted result if filtering is later used; and
- `FAILED` to the recorded failure.

The continuation may run inline on a Euhedral finalizer. It must not block, wait for lattice progress,
or submit concurrently through the same single-owner runner. Follow-up decode admission must be
handed back to the admission owner.

## Scheduling and ownership

### Admission

A single admission owner is responsible for runner checkout and publication. Concurrent external
requests are first placed into an application-level MPSC admission queue or dispatched to distinct
runner owners. The owner acquires the sequence execution lease before calling `submit`.

A successful submit transfers chain ownership to Euhedral. The caller retains only the outcome future
and request-level state that is safe to observe after completion.

### Stage execution

A stage borrows the context and its sequence lease for the duration of the callback. It may mutate
only the buffers and sequence fields assigned to that request. It must not retain a pipeline frame or
context reference after returning.

The synchronous `prepare/embed` stage checks request-local cancellation before work and after the CUDA
synchronization boundary. Temporary token IDs are freed in a `finally` path; a failure or cancellation
also closes the context-owned hidden-state workspace. A cancellation detected at a safe boundary uses
Euhedral's shared `AbstractFrame.CANCEL_SIGNAL`, allowing the pipeline to report `CANCELLED` rather than
a generic failure.

### Terminal ownership

The current terminal consumer writes the placeholder result while retaining the sequence execution lease.
When the supplied callback succeeds, Euhedral publishes the raw outcome first; the
`QwenExecutionRunner` continuation then releases the lease and completes the returned outcome future.
If the callback throws, the context records the pending failure and clears the result; after Euhedral
publishes `FAILED`, the continuation marks the sequence failed and releases the active lease. For
cancellation and other stage failures, the same continuation finalizes the pending terminal state and
closes the context-owned workspace:

- on success, it commits the new logical position and transfers the result to the caller;
- on cancellation or failure, it rolls back or invalidates partially updated request state according
  to the eventual kernel contract;
- for each published terminal outcome, it releases per-submission workspace and the sequence
  execution lease; and
- on request completion, it frees request-owned KV and recurrent allocations.

Model allocations remain owned by `QwenWeights` and its model-runtime owner for the model lifetime.

`AbstractExecutor` does not catch arbitrary JVM `Error` values. Such an error can bypass pipeline
finalization, outcome publication, and accepted-chain decrement. The runtime therefore needs a fatal
error and abandonment policy outside the normal outcome path: stop new admission, terminate or mark
pending requests failed, and release or quarantine request-owned resources whose completion cannot be
proven. Graceful completion alone cannot recover a chain abandoned by an uncaught `Error`.

## MTP

MTP is optional and is represented by a separate reusable pipeline definition. Create it only when
`QwenWeights.mtp()` is non-null and the configuration declares MTP.

The base runner remains responsible for target-model prefill, decode, and later target verification.
The MTP runner consumes the same exclusively leased sequence state to produce draft candidates. The
request coordinator serializes base and MTP submissions for one sequence. No MTP frame or runner is
created when MTP is absent.

This separation avoids conditional MTP branches in every base-model stage and keeps MTP ownership
explicit. The exact proposal/verification loop remains outside this document because kernel and KV
semantics have not yet been implemented.

## Synchronous CUDA boundary

The frame model is explicitly synchronous: the embedding stage completes only when its output is safe
for the successor or terminal consumer, so it synchronizes CUDA before returning. The embedding kernel
reads the compact Q3 row-split `tokenEmbedding` directly from its existing GPU allocation and writes
BF16 hidden-state vectors to the context workspace. It does not copy model weights to host memory.
Simply launching asynchronous CUDA work and returning would violate the contract because
`AbstractExecutor` immediately invokes `PipelineFrame.doFinally`, which publishes the successor.

Asynchronous CUDA therefore requires a different design: a custom frame/finalization or continuation
source must retain ownership until an event or callback transfers completion back to the scheduler.
Stock `PipelineFrame` cannot defer successor publication after its stage callback returns. That custom
ownership protocol is intentionally outside the first model.

CUDA streams should be request- or sequence-owned if `fanOut` permits stages to move between CPU
workers. A per-worker stream would require explicit cross-stream dependencies whenever a sequence
moves. The first embedding operation uses synchronous CUDA calls and introduces no events or deferred
frame completion.

## Lifecycle

The runtime owner creates objects in this order:

1. Load `QwenWeights` and construct the immutable execution plan.
2. Create the base runner and optional MTP runner.
3. Obtain the JVM-wide `ControlPlaneLattice` and attach the runners as upstream sources.
4. Admit inference quanta through the single admission owner.

Shutdown proceeds in the opposite ownership direction:

1. Stop accepting external requests.
2. Call `completeGracefully` on each runner.
3. Wait for application-level request outcomes, not queue emptiness.
4. Release sequence-owned GPU state and then model-owned weights.
5. Close the lattice only when this runtime owns the JVM-wide lattice lifecycle.

Immediate runner completion is cancellation, not graceful drain. Lattice shutdown is not a substitute
for request outcome accounting.

## Deliberate non-goals

This design does not yet specify or implement:

- transformer-layer CUDA kernels, streams, events, or graph capture;
- tensor-parallel, pipeline-parallel, or multi-GPU execution;
- continuous batching or batch formation;
- KV-cache layout, paging, eviction, or prefix reuse;
- normalization, logits processing, or sampling;
- MTP acceptance rules;
- backpressure limits for inference admission;
- public serving APIs; or
- a custom `AbstractFrame` or `AbstractExecutor` implementation.

Those concerns should extend the ownership model above rather than bypassing it. In particular, GPU
asynchrony must not make a successor runnable before its inputs are complete, and pooled frames must
never become owners of model- or sequence-lifetime device allocations.
