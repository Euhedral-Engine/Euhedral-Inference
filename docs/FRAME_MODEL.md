# Qwen instruction and frame model

Euhedral-Inference defines operations and owns model/runtime state. Euhedral-Execution schedules
individual ready operations. A frame is an instruction, not a prebuilt request pipeline or a
transformer-layer control loop.

## State and ownership

- `QwenWeights` supplies model-lifetime, GPU-resident weights. Frames borrow handles; a quantum never
  releases model weights. The runtime owner must not unload weights while any frame can use them.
- `QwenExecutionPlan` is an immutable description of instruction kinds, weight handles, dependency
  edges, and output dimensions. Its published instructions and edges cannot be mutated by callers.
  Constructing it does not allocate a quantum or GPU workspace.
- `QwenSequenceState` is shared request-lifetime state. An atomic lease protects sequence mutation,
  and cancellation may race with frame completion. Independent frames reading the same normalized
  buffer do not gain independent sequence mutation rights.
- `QwenExecutionContext` owns one inference quantum: copied token IDs, the sequence lease, per-node
  dependency counters, outstanding-frame count, failure/cancellation outcome, and its GPU workspace.
  Intermediate buffers have distinct addresses for the embedding, normalized activation, and each
  projection. They are retained until all admitted frames finish; no per-access buffer lock is used.
- `QwenExecutionRunner` implements `LatticeSource` directly. It admits quanta and queues each ready
  context in the MPSC partition belonging to its immutable instruction. Worker completion and
  admission may enqueue concurrently; neither path creates a frame. The scheduler controls `pull`
  and `request`, which have exclusive entry. On that serialized source path, the runner checks out a
  pooled frame and supplies the queued quantum context. `request` synchronously drains ready work and
  pushes frames directly to Euhedral. `pull` delivers frames to its consumer without accumulating
  demand or pushing downstream. Empty requests do not leave outstanding credits; Euhedral must request
  again to service later arrivals. Both calls bulk-drain ready work without a per-frame demand counter.
  The Euhedral-owned `pull` consumer and downstream `push` do not throw.

## Instruction availability

The concrete `EmbeddingFrame`, `RmsNormFrame`, and `LinearFrame` each live in their own source file
under `scheduling/frames`. A quantum initially queues its embedding operation context. Each frame
invokes a standalone CUDA operation and synchronizes before reporting completion. `QwenWorkGenerator`
follows precomputed successor edges only; it decrements dependency counters and queues each newly
ready operation context. It does not walk the full model or call a layer method that drives the next
operation. Independent projections from one normalized activation receive separate `LinearFrame`
instances and output buffers.

```text
embedding output -> RmsNormFrame -> LinearFrame (projection A)
                                  -> LinearFrame (projection B)
```

All three frame types borrow their weight handles and buffer addresses. Their frame-local
completion state has one worker owner; shared dependency counts and terminal ownership use atomics.
Each successor's work reservation is recorded before enqueue, while the completing frame retains its
own reservation through fan-out; inline execution therefore cannot finalize a workspace while a
sibling instruction remains pending. CUDA operations do not import Euhedral types.

`QwenWorkGenerator` owns a Euhedral `FrameManager` for each fixed instruction: this keeps a linear
frame's weight binding immutable even when several `LinearFrame` instructions exist. Checkout passes
only the new quantum context; no per-checkout wrapper or instruction copy is created. `getOrCreate`
runs only on the serialized `pull`/`request` path, so there is no manager lock on frame checkout.
Successful and failed finalization enqueue successor contexts, clear the completed quantum reference,
and then return the frame to its manager. Undelivered frames are cleared and recycled on the source
path if their quantum is cancelled. The manager's MPSC return queue accepts concurrent worker returns.
Reuse is best-effort when its bounded pool is full; execution does not depend on a frame being retained.
Frame references expire at finalization: callers must not retain or invoke a recycled frame, since the
manager can issue the same object for another quantum.

The default compact-model plan currently performs embedding only. The explicit norm/projection
constructor is an operator slice for verified, semantically matched inputs; it is not a claim that
an incomplete transformer layer or a first-layer FFN projection can run directly after embedding.
A full model plan requires genuine attention/GDN/FFN dependencies and corresponding operators.

## Completion boundary

All GPU operations remain synchronous. An operation's output becomes ready only after CUDA
synchronization. On success, a terminal consumer may read the still-live GPU buffers; then the
workspace and temporary token-ID buffer are released and the sequence lease is committed. The
caller receives a copy of the completion future, so it cannot publish a false terminal outcome.
Failure or cancellation prevents new successors, but already admitted frames still finalize before buffer
release. Ready contexts that have not been materialized are reaped only when the source next services
`pull` or `request`; without another source call, a cancelled quantum can remain outstanding. A failed
token-ID free is retried at terminal cleanup without discarding its address.
External cancellation is cooperative and does not interrupt an executing GPU call.
Cancellation that wins before the sequence lease is claimed produces a cancelled quantum, even if
it arrives between initial validation and the claim. Workspace allocations begin only after the
quantum owns the workspace object, retaining partial allocations for terminal cleanup or later retry.

`completeGracefully` closes new admission and signals source completion after every accepted quantum
has finalized. Queue emptiness alone is not a completion signal. `LatticeSource.complete()` currently
uses the same resource-safe graceful behavior. An executor abandoned by an uncaught JVM `Error` or a
permanently unavailable scheduler can leave a quantum pending; neither case can safely free buffers
can still be in use. Fatal shutdown and forced abandonment require a separate ownership policy.

The native RMSNorm and Q3-linear wrappers bind the calling worker to its CUDA device before their
first kernel load and before each launch. This permits CPU workers to move a ready frame between
threads while the current implementation remains single-device; multi-device context selection and
cross-device model residency are not implemented.

The architecture allows a future scheduler to defer GPU capacity, choose a CPU-capable operation,
coalesce ready frames, or run another sequence without changing instruction dependencies. It does
not yet implement transformer layers, KV cache, GDN state, MTP, attention, sampling, batching, CUDA
Graphs, or asynchronous GPU completion.
