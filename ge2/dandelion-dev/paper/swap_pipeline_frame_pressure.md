# Swap Pipeline Frame Pressure

Assume one GPU has 9 physical partition frames:

```text
4 visible compute frames + 5 non-visible frames = 9 total frames
```

The hidden frames are not extra compute width. They are frames that hold the next admitted partitions before the current compute state finishes.

## The Important Distinction

For one transition with `a` admitted partitions:

```text
while computing state t:
  visible current partitions        = 4
  hidden preloaded next partitions  = a
  free frames                       = 5 - a

after publishing hidden -> visible:
  visible next partitions           = 4
  stale dirty old partitions        = a
  free frames                       = 5 - a
```

That is why `a = 3` looks okay if you only inspect one transition:

```text
4 visible + 3 stale + 2 free = 9
```

But a fully pipelined steady state also wants to preload the following transition while those stale frames are still being written back:

```text
visible next partitions + stale dirty old partitions + hidden preloads for following state <= 9

4 + a_current + a_next <= 9
```

If we cap all transitions to the same maximum `a`, then:

```text
4 + a + a <= 9
2a <= 5
a <= 2
```

## Case A: 2-Admit Transition Fully Pipelines

```text
State t compute in progress

frames: [V0] [V1] [V2] [V3] [H4] [H5] [F6] [F7] [F8]
roles:   visible current       next preload       free
```

At the boundary, the two hidden frames become visible, and the two evicted visible frames become stale:

```text
State t+1 compute in progress, old dirty frames writing back

frames: [V0] [V1] [S2] [S3] [V4] [V5] [F6] [F7] [F8]
roles:   kept visible  stale     admitted visible  free
```

Now the next prefetch can still start immediately:

```text
State t+1 compute in progress, old dirty frames writing back, next preload active

frames: [V0] [V1] [S2] [S3] [V4] [V5] [H6] [H7] [F8]
roles:   visible       stale     visible       next preload  spare
```

No dependency between stale writeback and next preload exists. This is the robust fully-overlapped case.

## Case B: 3-Admit Transition Has Frame Pressure

```text
State t compute in progress

frames: [V0] [V1] [V2] [V3] [H4] [H5] [H6] [F7] [F8]
roles:   visible current       next preload          free
```

At the boundary, the three hidden frames become visible, and the three evicted visible frames become stale:

```text
State t+1 compute in progress, old dirty frames writing back

frames: [V0] [S1] [S2] [S3] [V4] [V5] [V6] [F7] [F8]
roles:   kept  stale stale stale admitted visible    free
```

Only two free frames remain. If the next transition also needs three admits, the next preload cannot fully start:

```text
Wanted for next transition:

frames: [V0] [S1] [S2] [S3] [V4] [V5] [V6] [H7] [H8] [?]
roles:   visible/stale/visible                    only 2 hidden slots, need 3
```

The missing frame must come from one stale writeback finishing. So `a = 3` can work only if stale writeback releases a frame early enough.

That means `a = 3` is timing-dependent, not structurally guaranteed.

## Timeline View

### `a <= 2`: steady-state pipeline

```text
time --->

compute state t:       [==============================]
preload t+1 hidden:    [------]                         hidden count <= 2

publish t+1:                              |

compute state t+1:                         [==============================]
writeback stale t:                         [------]                       stale count <= 2
preload t+2 hidden:                         [------]                       hidden count <= 2

peak frames during state t+1 compute:
  4 visible + 2 stale + 2 hidden + 1 spare = 9
```

### `a = 3`: pipeline has a possible bubble

```text
time --->

compute state t:       [==============================]
preload t+1 hidden:    [------]                         hidden count = 3

publish t+1:                              |

compute state t+1:                         [==============================]
writeback stale t:                         [-----------]                  stale count = 3
preload t+2 hidden:                         [---- two frames ----][wait][third frame]

peak requested frames during state t+1 compute:
  4 visible + 3 stale + 3 hidden = 10

available:
  9
```

## Practical Reading

The statement "swap visible with hidden and write stale back while training is busy" is correct.

The missing detail is that the same compute window also needs to preload the next hidden set. If stale writeback and next preload are both alive, they compete for the same 5 non-visible frames.

So:

```text
a <= 2  => fully pipelined by construction
a = 3   => can be hidden only if stale writeback finishes before the next preload needs the third frame
a = 4   => even more timing-dependent and usually exposed
```

For a scheduler whose goal is "swap fully overlapped, no bubbles", use `max_admits_per_transition <= 2` unless the runtime gets more non-visible frames or changes the writeback mechanism.

## FB86M p32/q4 Scheduling Target

Assume:

```text
FB embedding table              = 32 GiB
embedding partition size        = 1 GiB
optimizer-state partition size  = 1 GiB
logical partition payload       = 2 GiB
GPU memory                      = 24 GiB
reserved GPU memory             ~= 5 GiB
usable frame memory             ~= 19 GiB

resident logical frames         = floor(19 / 2) = 9
visible compute frames          = 4
non-visible frames              = 5
```

The best fully-pipelined target is therefore:

```text
4 visible compute frames
2 hidden preload frames
2 stale writeback frames
1 spare frame
= 9 total frames
```

The scheduling rule should be:

```text
every transition keeps 2 partitions
every transition evicts 2 partitions
every transition admits 2 partitions
```

Equivalently:

```text
|V_t| = 4
|V_t intersect V_{t+1}| = 2
|V_{t+1} - V_t| = 2
```

This is the 2-admit q4 scheduler.

## Pair-Chain View

A useful way to think about the schedule is that every visible state is made from two 2-partition pairs:

```text
state t:     A_t   + A_{t+1}
state t+1:   A_{t+1} + A_{t+2}
state t+2:   A_{t+2} + A_{t+3}
```

Each transition keeps one pair and admits one new pair:

```text
S0 = {0, 1 | 8, 9}
S1 = {8, 9 | 16, 17}
S2 = {16,17 | 24,25}
```

At each boundary:

```text
kept partitions:   2
evicted stale:     2
admitted hidden:   2
```

The frame picture during steady-state compute is:

```text
[V] [V] [V] [V] [S] [S] [H] [H] [F]
 compute visible   stale    hidden  spare
```

This is the first p32 layout that makes full overlap structurally possible on a 24 GiB GPU.

## Difference From p16 GPU-Aware

For p16, each embedding partition is about 2 GiB and each optimizer-state partition is another 2 GiB:

```text
p16 logical partition payload = 4 GiB
usable frame memory           ~= 19 GiB
resident logical frames       = floor(19 / 4) = 4
```

That leaves only the 4 visible compute frames:

```text
[V] [V] [V] [V]
```

There is no room for hidden preload frames or stale writeback frames. So p16 GPU-aware scheduling is mostly a reordering problem: it can reduce bad transitions, but it cannot make partition swap fully pipelined.

For p32:

```text
p32 logical partition payload = 2 GiB
resident logical frames       = 9
```

Now scheduling becomes a state-generation problem. The scheduler should generate states that satisfy the 2-admit pipeline constraint, not just reorder an existing CUSTOM template.

## Can Single-GPU p32/q4 Have 84 Transitions?

Short answer:

```text
No, not for the fully-pipelined 2-admit scheduler.
No, also not as an exact q4 cover if "84 transitions" means 85 states.
```

There are 32 partitions. For q4, each state covers at most:

```text
C(4, 2) = 6 unordered off-diagonal partition pairs
```

The full FB p32 problem has:

```text
C(32, 2) = 496 unordered off-diagonal partition pairs
```

A pure capacity lower bound would be:

```text
ceil(496 / 6) = 83 states
```

But that bound is too weak. The stronger Schonheim bound for covering all pairs with 4-sets is:

```text
ceil(32 / 4 * ceil(31 / 3)) = 8 * 11 = 88 states
```

So any exact q4 state cover needs at least:

```text
88 states
87 transitions
```

That already rules out 84 transitions if it means:

```text
84 transitions = 85 states
```

For the 2-admit pipeline constraint, the lower bound is even larger. The first state can cover 6 unordered pairs. Each later state keeps 2 partitions and admits 2 new partitions. The kept pair is already present, so the next state can add at most:

```text
C(4,2) - C(2,2) = 6 - 1 = 5 new unordered pairs
```

Therefore:

```text
states >= 1 + ceil((496 - 6) / 5)
       = 1 + ceil(490 / 5)
       = 99 states

transitions >= 98
```

So the realistic single-GPU p32/q4 target is:

```text
2-admit fully-pipelined scheduler:
  lower bound: 99 states / 98 transitions
  practical target: ~100-115 states
```

Current CUSTOM p32/q4 has 104 states, but it includes some 3-admit transitions. The new scheduler should aim to stay near the same state count while removing those 3-admit spikes:

```text
current CUSTOM-like shape:
  ~104 states
  some 3-admit transitions
  pipeline is timing-dependent

new 2-admit q4 shape:
  ~100-115 states
  max 2 admits per transition
  pipeline is frame-safe by construction
```

## Proposed New Scheduler: 2-Admit Work-Balanced Cover

The new scheduler should not be "CUSTOM with a different order". It should generate the 4-partition states directly with the GPU frame budget as a hard constraint.

Name used here:

```text
TWO_ADMIT_Q4
```

Core idea:

```text
every state has 4 visible partitions
every transition keeps exactly 2 partitions
every transition admits exactly 2 partitions
all buckets are still assigned exactly once
```

So every transition looks like this:

```text
before:  {a, b, c, d}
after:   {c, d, e, f}

kept:    {c, d}
evicted: {a, b}
admit:   {e, f}
```

The visible state is still q4. We are not changing the model semantics or the bucket definition. We are only controlling which q4 states are visited and in what order.

### Why This Can Be Better Than CUSTOM

CUSTOM p32/q4 is good at producing a valid cover, but it is not designed around the 9-frame pipeline:

```text
current CUSTOM p32/q4:
  valid coverage
  ~104 states
  mostly 2-admit transitions
  some 3-admit transitions
```

Those 3-admit transitions are the problem. They create this peak request:

```text
4 visible + 3 stale + 3 hidden = 10 frames
```

But the GPU has only 9 logical frames. So the runtime must rely on timing: stale writeback must release a frame before the next preload needs it.

The proposed scheduler removes that dependency:

```text
new TWO_ADMIT_Q4:
  valid coverage
  ~100-115 states
  all transitions <= 2 admits
  4 visible + 2 stale + 2 hidden + 1 spare = 9 frames
```

The expected win is not a huge reduction in state count. The expected win is lower exposed swap overhead and fewer boundary spikes.

### What The Scheduler Must Optimize

Hard constraints:

```text
1. |V_t| = 4
2. |V_t intersect V_{t+1}| = 2
3. every directed bucket (i,j) is assigned exactly once
4. no bucket is dropped
5. no bucket is trained twice
6. no transition admits more than 2 partitions
7. state work is not allowed to collapse to near-zero
```

The last constraint matters because a 2-admit walk can otherwise create many late states with little fresh work. That would preserve accuracy but waste launch/setup/swap overhead.

Soft objective, in priority order:

```text
1. maximize fresh uncovered bucket work per state
2. minimize worst-state underfill
3. minimize edge-count imbalance between states
4. minimize total admitted partitions
5. minimize host H2D/D2H bytes
6. maximize peer/local reuse if running multi-GPU
7. minimize state count
```

State count is last because a 99-state schedule with bad work imbalance can be slower than a 108-state schedule with balanced states and hidden swaps.

### Beam-Search Construction

The simplest practical builder is a beam search over q4 states.

Candidate state:

```text
V = {p0, p1, p2, p3}
```

Transition rule:

```text
next V must share exactly 2 partitions with current V
```

At each step, score candidate next states by:

```text
score(next) =
    fresh_edge_work(next)
  - repeated_pair_penalty(next)
  - state_underfill_penalty(next)
  - edge_imbalance_penalty(next)
  - future_dead_end_penalty(next)
```

The scheduler stops when all 32 x 32 directed buckets are assigned.

Pseudocode:

```text
uncovered = all directed buckets
beam = initial q4 states

while uncovered is not empty:
  new_beam = []
  for partial_schedule in beam:
    current = partial_schedule.last_state
    for next_state sharing exactly 2 partitions with current:
      fresh = buckets in next_state that are still uncovered
      if fresh is too small:
        continue
      candidate = partial_schedule + next_state
      assign fresh buckets to next_state
      score candidate
      new_beam.push(candidate)
  beam = best K candidates

return best complete schedule
```

### What A Good Result Looks Like

For FB p32/q4 single-GPU:

```text
states:              100-115
transitions:         99-114
max admits:          2
hidden frames used:  <= 2
stale frames used:   <= 2
bucket coverage:     exact once
state work:          balanced by real edge counts
```

The comparison should be:

```text
CUSTOM p32/q4:
  ~104 states
  84 two-admit transitions
  19 three-admit transitions
  frame-safe only when stale writeback finishes fast enough

TWO_ADMIT_Q4:
  ~100-115 states
  all transitions two-admit
  frame-safe by construction
  should reduce boundary spikes and exposed swap
```

The new scheduler is better only if the measured epoch time improves. The expected improvement comes from making the partition pipeline stable, not from reducing the number of states.

## What Is The Actual Benefit?

The goal is epoch time:

```text
epoch_time =
    useful_compute_time
  + redundant_state_compute_time
  + exposed_swap_time
  + boundary/setup_overhead
  + multi-GPU tail/imbalance
```

The proposed scheduler mainly targets:

```text
exposed_swap_time
```

It does not magically remove all swaps. It makes each swap small enough to fit the pipeline.

### Current p32/q4 CUSTOM-Like Trace

Observed p32/q4 trace shape:

```text
states:         104
transitions:    103
2-admit rounds: 84
3-admit rounds: 19
total admits:   225 logical partitions
```

With a 2 GiB logical partition:

```text
2-admit transition:
  H2D preload     = 2 * 2 GiB = 4 GiB
  D2H stale       = 2 * 2 GiB = 4 GiB
  total movement  = 8 GiB
  frame request   = 4 visible + 2 hidden + 2 stale = 8 frames

3-admit transition:
  H2D preload     = 3 * 2 GiB = 6 GiB
  D2H stale       = 3 * 2 GiB = 6 GiB
  total movement  = 12 GiB
  frame request   = 4 visible + 3 hidden + 3 stale = 10 frames
```

The GPU has only 9 frames. So the 3-admit transitions are the risky ones.

### Proposed TWO_ADMIT_Q4 Benefit

If the new scheduler keeps the state count close to current CUSTOM:

```text
states:       about 100-115
transitions:  about 99-114
max admits:   2
```

Then the benefit is:

```text
1. max swap size drops
   from 3 admitted partitions to 2 admitted partitions

2. peak frame request drops
   from 10 frames to 8 frames plus 1 spare

3. exposed swap bubbles should drop
   because hidden preload and stale writeback can run together

4. boundary spikes should shrink
   because the scheduler removes 3-admit transitions

5. total transfer may drop if state count stays near 104
```

For example, if the new schedule also used 104 states:

```text
current admits:      225
2-admit admits:      103 transitions * 2 = 206
saved admits:        19 logical partitions

saved H2D:           19 * 2 GiB = 38 GiB
saved D2H:           19 * 2 GiB = 38 GiB
saved total traffic: 76 GiB
```

That is not the main benefit, but it helps.

The main benefit is that every transition becomes pipeline-safe:

```text
4 visible + 2 stale + 2 hidden + 1 spare = 9 frames
```

### What If The New Scheduler Adds More States?

This is the trade-off.

If the new scheduler needs too many states, it can lose:

```text
more states => more transitions
more transitions => more boundary/setup cost
more states => more redundant compute or small-state overhead
```

So the scheduler must not optimize only for `max_admits <= 2`. It must optimize:

```text
minimize exposed epoch time
```

Good target:

```text
100-110 states
all transitions <= 2 admits
balanced edge work per state
```

Risky target:

```text
125+ states
all transitions <= 2 admits
but many weak/underfilled states
```

The second one may have cleaner swaps but worse epoch time.

### Decision Rule

TWO_ADMIT_Q4 is better than CUSTOM only if:

```text
saved_exposed_swap_time >
  extra_state_compute_time
+ extra_boundary_overhead
+ extra_tail_imbalance
```

So the comparison should report:

```text
states
transitions
max_admits
total_admits
total H2D/D2H bytes
max H2D/D2H bytes per transition
state edge-work balance
predicted exposed swap time
measured epoch time
```

If CUSTOM's 3-admit transitions are already fully hidden, TWO_ADMIT_Q4 may not help much. If those transitions create the observed swap spikes, TWO_ADMIT_Q4 is exactly the right fix.

### Bigger Alternative: q8

There is another way to reduce swap count: increase visible capacity to q8.

But q8 changes the memory picture:

```text
8 visible frames + no hidden/stale room
```

That can reduce states a lot, but it exposes swap unless each q8 state has enough compute to hide boundary transfer. So q8 is a separate ablation:

```text
q4 TWO_ADMIT_Q4:
  more states
  smaller swaps
  fully pipeline-safe

q8 lifted-q4:
  fewer states
  larger exposed swaps
  only good if compute per state hides transfer
```

## What If We Reserve Less Memory And Allow 3-Admit?

This can work in theory, but it changes the safety margin.

With p32:

```text
embedding partition      = 1 GiB
optimizer partition      = 1 GiB
logical partition frame  = 2 GiB
GPU memory               = 24 GiB
```

Current conservative budget:

```text
reserved                 = 5 GiB
usable frame memory      = 19 GiB
logical frames           = floor(19 / 2) = 9

9 frames = 4 visible + 2 stale + 2 hidden + 1 spare
```

For fully-overlapped 3-admit, the minimum frame request is:

```text
4 visible + 3 stale + 3 hidden = 10 frames
```

That requires:

```text
10 frames * 2 GiB = 20 GiB frame memory
reserved <= 24 - 20 = 4 GiB
```

So reducing reserve from 5 GiB to 4 GiB can make 3-admit possible by frame count:

```text
reserved                 = 4 GiB
usable frame memory      = 20 GiB
logical frames           = 10

10 frames = 4 visible + 3 stale + 3 hidden
```

But this has no spare frame:

```text
4 visible + 3 stale + 3 hidden + 0 spare = 10
```

The robust 3-admit version needs one spare:

```text
4 visible + 3 stale + 3 hidden + 1 spare = 11 frames
```

That requires:

```text
11 frames * 2 GiB = 22 GiB frame memory
reserved <= 24 - 22 = 2 GiB
```

That is usually too tight because the process still needs memory for:

```text
CUDA context
PyTorch allocator fragmentation
edge batches and negative-sampling tensors
temporary index/map tensors
async copy/staging tensors
peer-relay source scratch if multi-GPU
kernel workspaces
```

So the options are:

```text
reserve 5 GiB:
  9 frames
  2-admit is robust
  3-admit is not frame-safe

reserve 4 GiB:
  10 frames
  3-admit can fit with zero spare
  risky but worth testing as an ablation

reserve 2 GiB:
  11 frames
  3-admit with one spare
  likely too risky on 24 GiB cards
```

The scheduler decision changes if 10 frames are stable:

```text
9 frames:
  best scheduler target = TWO_ADMIT_Q4

10 frames:
  best scheduler target = THREE_ADMIT_Q4
```

For `THREE_ADMIT_Q4`, every transition should keep one partition and admit three:

```text
before:  {a, b, c, d}
after:   {d, e, f, g}

kept:    {d}
evicted: {a, b, c}
admit:   {e, f, g}
```

This has a better theoretical state lower bound than 2-admit:

```text
first state covers 6 unordered pairs
each next 3-admit state can add at most 6 new unordered pairs

states >= 1 + ceil((496 - 6) / 6)
       = 1 + ceil(490 / 6)
       = 83 states
```

But the general q4 cover lower bound is still:

```text
88 states
```

So a practical 3-admit q4 target is:

```text
88-104 states
max admits = 3
larger swaps than 2-admit
fewer states than 2-admit
only valid if 10 frames are stable at runtime
```

This gives the real trade-off:

```text
TWO_ADMIT_Q4:
  9 frames
  smaller swaps
  more states
  safer overlap

THREE_ADMIT_Q4:
  10 frames minimum
  larger swaps
  fewer states
  no spare unless reserve <= 2 GiB
```

Recommended experiment:

```text
1. keep reserve ~= 5 GiB and test TWO_ADMIT_Q4
2. reduce reserve to ~= 4 GiB and test THREE_ADMIT_Q4
3. compare measured epoch time, OOM rate, boundary spikes, and stale backlog
```

If 10 frames are stable, `THREE_ADMIT_Q4` may be faster because it can reduce state count and total transitions. If 10 frames causes allocator pressure or copy-staging fallbacks, `TWO_ADMIT_Q4` will be more reliable.

## 4 GiB Reserve Plan

The code does not expose a direct "reserve exactly 4 GiB" option. The effective knob is the number of extra physical frames:

```text
physical_frames = buffer_capacity + GEGE_FRAME_CACHE_HIDDEN_FRAMES
```

For q4:

```text
buffer_capacity = 4
```

So:

```text
GEGE_FRAME_CACHE_HIDDEN_FRAMES=5  => 9 physical frames
GEGE_FRAME_CACHE_HIDDEN_FRAMES=6  => 10 physical frames
```

For FB p32, one logical frame is:

```text
1 GiB embedding + 1 GiB optimizer state = 2 GiB
```

Therefore:

```text
10 frames * 2 GiB = 20 GiB
24 GiB GPU - 20 GiB frames = 4 GiB remaining
```

So the 4 GiB reserve experiment is:

```bash
export GEGE_FRAME_CACHE_HIDDEN_FRAMES=6
```

Keep the rest of the frame-cache pipeline enabled:

```bash
export GEGE_SINGLE_GPU_GPU_AWARE_CUSTOM=1
export GEGE_SINGLE_GPU_ASYNC_ADMIT_PRELOAD=1
export GEGE_FRAME_CACHE_HIDDEN_ONLY_PRELOAD=1
export GEGE_FRAME_CACHE_DELAYED_STALE_WRITEBACK=1
export GEGE_FRAME_CACHE_PRIORITIZED_WRITEBACK=1
export GEGE_FRAME_CACHE_SERIALIZE_ADMIT_H2D=1
export GEGE_EMPTY_CACHE_AROUND_SWAP=0
export GEGE_SYNC_BEFORE_SWAP=0
```

### Phase 1: Prove 10 Frames Are Stable

Do not change the scheduler first. Run current p32/q4 CUSTOM-like schedule with 10 frames:

```text
num_partitions = 32
buffer_capacity = 4
GEGE_FRAME_CACHE_HIDDEN_FRAMES = 6
```

What to check:

```text
1. no CUDA OOM
2. no async stage memory fallback
3. no preload miss swaps
4. no partial preload swaps
5. stale_backlog_after_publish_max <= 3
6. boundary spikes shrink or at least do not grow
7. epoch metrics remain unchanged within normal stochastic noise
```

The useful log fields are:

```text
[perf][epoch][frame_cache]
  preload_miss_swaps
  partial_preload_swaps
  delayed_stale_writeback_swaps
  async_evict_in_flight_before_swap_swaps
  reserved_preload_frames_avg
  free_frames_before_swap_avg
  free_frames_after_publish_avg
  stale_backlog_after_publish_max
```

If this phase fails, the 4 GiB reserve is not enough and the scheduler should stay with `TWO_ADMIT_Q4`.

### Phase 2: Build THREE_ADMIT_Q4

If 10 frames are stable, the better scheduler target becomes:

```text
THREE_ADMIT_Q4
```

Hard transition rule:

```text
|V_t| = 4
|V_t intersect V_{t+1}| = 1
|V_{t+1} - V_t| = 3
```

Example:

```text
before: {a, b, c, d}
after:  {d, e, f, g}

kept:   {d}
evict:  {a, b, c}
admit:  {e, f, g}
```

Frame model:

```text
4 visible + 3 stale + 3 hidden = 10 frames
```

This uses all available frame memory, so the scheduler should also prefer transitions where stale writeback is expected to finish quickly. There is no spare frame.

### Expected Scheduler Shape

For p32/q4:

```text
general q4 cover lower bound: 88 states
current CUSTOM-like schedule: 104 states
THREE_ADMIT_Q4 target:        88-104 states
```

A good result would be:

```text
states:              88-100
transitions:         87-99
max admits:          3
hidden frames used:  <= 3
stale frames used:   <= 3
bucket coverage:     exact once
state work:          balanced by real edge counts
```

### Accuracy-Preserving Rules

The scheduler must not change the training objective. Accuracy is preserved only if:

```text
1. every directed bucket (src_partition, dst_partition) is assigned exactly once
2. no directed bucket is dropped
3. no directed bucket is trained twice
4. diagonal buckets are still handled exactly as current q4 semantics require
5. negative sampling and filtering are unchanged
6. partition IDs are only reordered, not remapped incorrectly
7. embedding and optimizer-state storage use the same buffer-state sequence
8. dirty/stale writeback completes before a partition version is reused from host
9. peer relay, if used, copies the newest version of the partition
```

The scheduler can change when a bucket is trained, but not whether it is trained or what examples it contributes.

The implementation should validate:

```text
coverage_count[bucket] == 1 for all 32 * 32 buckets
max_admits_per_transition <= 3
max_live_frames <= 10
state_edge_work_min >= threshold
```

### Decision Rule For 4 GiB Reserve

Use `THREE_ADMIT_Q4` only if:

```text
10-frame runtime is stable
and
state_count reduction + smaller epoch overhead beats the risk of zero spare frame
```

Compare three runs:

```text
A. current CUSTOM p32/q4, 9 frames
B. current CUSTOM p32/q4, 10 frames
C. new THREE_ADMIT_Q4, 10 frames
```

Report:

```text
epoch time
MRR/Hits or validation metric
states/transitions
total admitted partitions
max admitted partitions
total H2D/D2H bytes
max swap bytes
preload_miss_swaps
partial_preload_swaps
stale_backlog_after_publish_max
CUDA OOM or allocator fallback events
```

If B improves over A, the extra frame helps even before a new scheduler. If C improves over B with unchanged accuracy, then `THREE_ADMIT_Q4` is the right direction.

### Initial 10-Frame Smoke Test

Run:

```text
run name: fb86m_p32_q4_10frames_3e_20260501_153144
config:   freebase86m_32p_epoch158_20260428.yaml
epochs:   3
GPU:      single RTX 3090
frames:   buffer_capacity=4 visible + GEGE_FRAME_CACHE_HIDDEN_FRAMES=6 = 10 total
schedule: current CUSTOM p32/q4
log:      /home/smansou2/codex_runs/exp_logs/fb86m_p32_q4_10frames_3e_20260501_153144_train.log
```

Observed:

```text
epoch  runtime_ms  swap_update_ms  hidden_publish_parts  preload_miss  partial_preload  stale_backlog_after_max
1      111885      24176.920       225                   0             0                3
2      109078      21555.875       226                   0             0                3
3      108752      21524.934       226                   0             0                3

avg_runtime_s:       109.905
avg_edges_per_sec:   5441908.3
swap_count_per_epoch: 103
```

Interpretation:

```text
10-frame p32/q4 did not OOM for 3 epochs.
No preload misses occurred.
No partial preloads occurred.
No visible fallback installs occurred.
The stale backlog reached 3, which is expected for current CUSTOM transitions that admit up to 3 partitions.
```

This validates the first step of the 4 GiB reserve plan: the current scheduler can run with 10 logical frames. It does not yet prove accuracy, because evaluation was disabled for this timing run, and it does not yet prove that `THREE_ADMIT_Q4` is better, because the scheduler was still current CUSTOM.

## Current p32/q4 Bounded Scheduler Correctness

The selected p32/q4 bounded schedule is:

```text
GEGE_BOUNDED_GREEDY_COVER_Q4=1
GEGE_MINMAX_BUCKET_ASSIGNMENT=1
```

The schedule works with:

```text
p = 32 partitions
q = 4 visible partitions per state
directed buckets = p * p = 1024
```

A state is a 4-subset:

```text
S_t subset {0, ..., 31}
|S_t| = 4
```

A directed bucket `(i,j)` is legal in state `S_t` iff:

```text
i in S_t and j in S_t
```

Accuracy is preserved by bucket ownership:

```text
every directed bucket (i,j) is assigned to exactly one legal state
```

The state set may cover the same partition pair more than once. That is not an accuracy problem. The assignment layer still chooses one owner state for each bucket.

### Verified p32/q4 Numbers

The actual hard-coded `p32_q4_bounded_greedy_cover_state_order()` has:

```text
states:                         93
transitions:                    92
all states size 4:              true
all partition IDs in [0,31]:     true
missing directed buckets:        0 / 1024
first-cover exact assignments:   1024 / 1024
max admits per transition:       3
admit histogram:                 {2: 48, 3: 44}
total hidden publishes:          228
```

The unordered pair coverage histogram is:

```text
covered once:  444 pairs
covered twice: 43 pairs
covered 3x:    8 pairs
covered 4x:    1 pair
total:         496 pairs
```

So every unordered off-diagonal partition pair is covered by at least one state, and every directed bucket is assignable to at least one legal state.

### Why There Are 92 Transitions

The first visible state is loaded before the epoch starts. Transitions are only boundaries between consecutive states:

```text
state0 -> state1
state1 -> state2
...
state91 -> state92
```

Therefore:

```text
transitions = states - 1 = 93 - 1 = 92
```

### Why There Are 228 Hidden Publishes

For one transition:

```text
admits = next_visible - current_visible
evicts = current_visible - next_visible
keeps  = current_visible intersect next_visible
```

Since both states have size 4:

```text
admits = evicts = 4 - |current_visible intersect next_visible|
```

The schedule has:

```text
48 transitions with overlap 2 -> 2 admits each
44 transitions with overlap 1 -> 3 admits each
```

So:

```text
hidden_publish_parts = 48 * 2 + 44 * 3
                     = 96 + 132
                     = 228
```

This is the number of hidden/preloaded partition frames published into visible slots per epoch. It is not the number of states.

Example:

```text
state0 = {6, 17, 27, 28}
state1 = {3, 12, 25, 28}

keep:  {28}
admit: {3, 12, 25}   -> 3 hidden publishes
evict: {6, 17, 27}   -> 3 stale writebacks
```

Next:

```text
state1 = {3, 12, 25, 28}
state2 = {12, 21, 22, 28}

keep:  {12, 28}
admit: {21, 22}      -> 2 hidden publishes
evict: {3, 25}       -> 2 stale writebacks
```

## State Count Lower Bounds

The naive directed-bucket lower bound is:

```text
ceil(p^2 / q^2)
```

For p32/q4:

```text
ceil(32^2 / 4^2) = ceil(1024 / 16) = 64
```

This is too weak because off-diagonal directed buckets come in unordered partition pairs. To train both `(i,j)` and `(j,i)`, some state must contain both `i` and `j`.

For q4:

```text
unordered pairs to cover = C(p,2)
pairs covered per state  = C(4,2) = 6
```

The pair-capacity lower bound is:

```text
ceil(C(p,2) / C(4,2))
```

For p32:

```text
ceil(C(32,2) / 6) = ceil(496 / 6) = 83
```

This is still not tight.

The stronger Schonheim lower bound for covering all pairs with q-sets is:

```text
B >= ceil(p / q * ceil((p - 1) / (q - 1)))
```

For p32/q4:

```text
B >= ceil(32 / 4 * ceil(31 / 3))
  = ceil(8 * 11)
  = 88 states
```

Equivalently, each partition must appear with the other 31 partitions. One q4 state containing partition `x` pairs `x` with at most 3 other partitions, so each partition must appear in at least:

```text
ceil(31 / 3) = 11 states
```

Across 32 partitions:

```text
32 * 11 = 352 partition appearances
```

Each state contributes 4 appearances:

```text
4B >= 352
B >= 88
```

The exact covering number is known:

```text
C(32,4,2) = 88
```

Reference: La Jolla Covering Repository, `C(32,4,2)=88`, lower bound Schonheim:
https://ljcr.dmgordon.org/cover/show_cover.php?k=4&t=2&v=32

So:

```text
88 states = mathematical optimum for coverage alone
93 states = current practical bounded ordered schedule
```

The current 93-state schedule is correct, but it is not proven state-count optimal.

## Ordered Pipeline Lower Bounds

Coverage alone ignores transition order. The pipeline also imposes a maximum admit count:

```text
a_t = |S_{t+1} - S_t|
```

For q4:

```text
overlap_t = |S_t intersect S_{t+1}|
a_t = 4 - overlap_t
```

If the maximum admit cap is `a`, then every transition must keep at least:

```text
q - a
```

partitions.

The first state can cover at most:

```text
C(q,2)
```

unordered partition pairs. Each later state can add at most:

```text
C(q,2) - C(q - a, 2)
```

new unordered pairs, because the kept partitions already co-occurred in the previous state.

Therefore an ordered lower bound is:

```text
B >= 1 + ceil((C(p,2) - C(q,2)) /
              (C(q,2) - C(q - a,2)))
```

For p32/q4 with a 2-admit fully frame-safe schedule:

```text
a = 2
C(4,2) - C(2,2) = 6 - 1 = 5

B >= 1 + ceil((496 - 6) / 5)
  = 1 + 98
  = 99 states
```

So a strict 2-admit p32/q4 schedule cannot have 88 or 93 states. It needs at least 99 states.

For p32/q4 with a 3-admit schedule:

```text
a = 3
C(4,2) - C(1,2) = 6 - 0 = 6

B >= 1 + ceil((496 - 6) / 6)
  = 83 states
```

The coverage lower bound of 88 dominates, so 3-admit scheduling can in principle reach the 88-state coverage optimum. Our current 93-state schedule is 5 states above that optimum but has a bounded transition path and balanced bucket assignment.

## Arbitrary Partition Count p

For q4 and arbitrary partition count `p`, the correctness requirements do not depend on `p` being a power of two:

```text
1. every state has exactly 4 distinct partition IDs in [0, p-1]
2. every unordered pair {i,j}, i != j, is contained in at least one state
3. every directed bucket (i,j) is assigned to exactly one compatible state
4. every transition satisfies the chosen admit cap
5. every state has enough assigned edge work to justify a kernel/setup boundary
```

The general coverage lower bound is:

```text
B_cover(p,4) >= ceil(p / 4 * ceil((p - 1) / 3))
```

The general ordered lower bound with admit cap `a` is:

```text
B_ordered(p,4,a) >=
  1 + ceil((C(p,2) - 6) / (6 - C(4 - a,2)))
```

The scheduler lower bound is:

```text
B >= max(B_cover, B_ordered)
```

Examples:

```text
p=32, q=4, a=3:
  B_cover   >= 88
  B_ordered >= 83
  lower     >= 88
  current   = 93

p=32, q=4, a=2:
  B_cover   >= 88
  B_ordered >= 99
  lower     >= 99

p=43, q=4, a=3:
  B_cover   >= ceil(43/4 * ceil(42/3)) = ceil(10.75 * 14) = 151
  B_ordered >= 1 + ceil((903 - 6) / 6) = 151
  lower     >= 151
```

The exact covering number is also known for p43:

```text
C(43,4,2) = 151
```

Reference: La Jolla Covering Repository, `C(43,4,2)=151`, lower bound Schonheim:
https://ljcr.dmgordon.org/cover/show_cover.php?k=4&t=2&v=43

For arbitrary `p`, exact covering numbers are not always guaranteed by a simple formula. The safe planner should:

```text
1. compute the lower bounds
2. generate or import a covering state set
3. order it under the admit cap
4. assign buckets exactly once
5. validate all invariants before training
```

## CUSTOM Power-of-Two Limitation

There are two different things called "CUSTOM" in the code path:

```text
canonical CUSTOM template
bounded greedy q4 schedule selected through CUSTOM ordering
```

The canonical CUSTOM template has a power-of-two limitation. The code path computes `log2(num_partitions)` and asserts:

```text
2^log2l == num_partitions
```

So canonical CUSTOM is unsafe for partition counts like:

```text
p = 43
```

The lifted q4 CUSTOM path is also restricted. It requires the coarse partition count to be a power of two:

```text
coarse_num_partitions = num_partitions / (buffer_capacity / 4)
coarse_num_partitions must be a positive power of two
```

The selected bounded greedy q4 scheduler is different. In `getBoundedGreedyCoverEdgeBucketOrdering`:

```text
if p == 32 and q == 4:
  use hard-coded p32_q4_bounded_greedy_cover_state_order()
else:
  build_greedy_cover_state_set(p, 4)
  reorder_states_for_bounded_admits(..., max_admits=3)
```

So the bounded greedy q4 idea does not mathematically require `p` to be a power of two. However, the current best 93-state schedule is p32-specific. For p43 or any other `p`, the code uses the generic greedy cover plus bounded reorder. That is general, but it is not yet a formally optimal or heavily tuned planner.

The generic builder enumerates all q4 candidates:

```text
C(p,4)
```

For p43:

```text
C(43,4) = 123410 candidate states
```

That is still practical. For much larger `p`, this may need a sampled/beam construction rather than full enumeration.

## General Multi-GPU Formalization

For `G` GPUs, the schedule should be represented as super-rounds:

```text
S_{r,g} = visible state for round r on GPU g
```

where:

```text
r = 0, ..., R-1
g = 0, ..., G-1
|S_{r,g}| = q = 4
```

The accuracy invariant is global:

```text
every directed bucket (i,j) is assigned to exactly one compatible state S_{r,g}
```

The lane-local admit count is:

```text
a_{r,g} = |S_{r+1,g} - S_{r,g}|
```

The pipeline constraint is:

```text
a_{r,g} <= a_max for all r,g
```

The structural frame bound is:

```text
q visible + a_current stale + a_next hidden <= total_frames
```

If all transitions are capped at the same `a_max`, then:

```text
q + 2*a_max <= total_frames
```

For q4:

```text
9 frames:  4 + 2*a <= 9   -> a <= 2
10 frames: 4 + 2*a <= 10  -> a <= 3
```

The 10-frame `a=3` case has no spare frame:

```text
4 visible + 3 stale + 3 hidden = 10
```

The run shows this can work for p32 on one RTX 3090 with zero preload misses, but it leaves less slack than the 9-frame 2-admit design.

### Multi-GPU Round Constraints

Within one super-round, prefer disjoint visible states:

```text
S_{r,g1} intersect S_{r,g2} = empty
```

when possible. This is possible only if:

```text
G * q <= p
```

For p32/q4:

```text
G=2 -> 8 visible partitions per round, disjoint possible
G=4 -> 16 visible partitions per round, disjoint possible
```

For p43/q4:

```text
G=4 -> 16 visible partitions per round, disjoint possible
```

Disjointness is not an accuracy requirement. It is a residency and work-balance objective. The hard accuracy requirement remains exact bucket assignment.

### Multi-GPU Lower Bounds

If the state cover needs `B` total q4 states, then the number of super-rounds satisfies:

```text
R >= ceil(B / G)
```

This is only a wall-clock lower bound. It ignores lane continuity and disjoint grouping.

The number of lane-local transitions is approximately:

```text
B - G
```

assuming every GPU lane receives at least one state. Hidden publishes are:

```text
sum over lanes and lane transitions of |S_{r+1,g} - S_{r,g}|
```

So multi-GPU reduces wall-clock rounds, but it does not make the admit/update count disappear. It changes which admits can be overlapped or served by peer relay.

### Peer Relay Formalization

Let:

```text
U_r = union over GPUs of S_{r,g}
```

For GPU `g`, a partition admitted at `r+1` is:

```text
x in S_{r+1,g} - S_{r,g}
```

It is peer-relay eligible if another GPU had it in the previous round:

```text
x in U_r and x not in S_{r,g}
```

Then the scheduler can copy `x` from a sibling GPU instead of loading it from host storage.

The multi-GPU score should count:

```text
host_admit_bytes = admits not peer-relay eligible
peer_admit_bytes = admits peer-relay eligible
```

with host bytes more expensive than peer bytes.

### Bad vs Good Multi-GPU Extension

Bad extension:

```text
round 0: GPU0=state0, GPU1=state1, GPU2=state2, GPU3=state3
round 1: GPU0=state4, GPU1=state5, GPU2=state6, GPU3=state7
```

This blindly stripes a single-GPU order. GPU0 now transitions `state0 -> state4`, GPU1 transitions `state1 -> state5`, etc. Those transitions were not optimized, so they may admit 4 partitions and break the pipeline.

Good extension:

```text
choose states, round groups, and lane assignment together
```

Hard constraints:

```text
1. |S_{r,g}| = 4
2. every directed bucket assigned exactly once globally
3. |S_{r+1,g} - S_{r,g}| <= a_max for every GPU lane
4. frame pressure satisfies q + 2*a_max <= total_frames
5. state work does not collapse to tiny tail states
```

Soft objectives:

```text
1. minimize max host admits per GPU per round
2. maximize peer-relay eligible admits
3. balance edge work across GPUs within each round
4. prefer disjoint visible states within each round
5. minimize total hidden publishes
6. minimize state count after the above are satisfied
```

### Implementation Plan For General p and Multi-GPU

The planner should be split into four stages:

```text
1. cover generation
   input: p, q
   output: candidate q-sets covering all unordered pairs

2. bucket assignment
   input: q-sets and bucket edge counts
   output: exact one-owner assignment for every directed bucket

3. ordered lane scheduling
   input: assigned q-sets, G, a_max, frame count
   output: super-round/lane schedule S_{r,g}

4. validation
   check exact coverage, max admits, frame pressure, peer handoffs,
   state work balance, and bucket ownership uniqueness
```

For current code, this means:

```text
single-GPU p32:
  BOUNDED_GREEDY_COVER_Q4 works and is tested

single-GPU arbitrary p:
  generic greedy cover + bounded reorder exists, but must be validated per p

multi-GPU:
  current stateflow lane matching and peer handoff machinery exists,
  but BOUNDED_GREEDY_COVER_Q4 is currently enabled only for single-GPU.
  The next implementation step is to allow the bounded/general q4 state set
  to feed the multi-GPU lane matcher, then add a hard max-admit constraint
  and peer-relay-aware score.
```

## Dataset-Independent Fixed-Size Partition Formalization

The fixed-size partition story should be presented independently of Freebase.
Each dataset gets a different number of partitions because the partition size is fixed.

For dataset `D`, define:

```text
n_D      = number of entities/nodes
d        = embedding dimension
b        = bytes per embedding scalar
S_emb    = target embedding partition size in bytes
lambda   = logical frame multiplier including optimizer state
           current embedding + optimizer setup: lambda = 2
```

The embedding table size is:

```text
E_D = n_D * d * b
```

The number of fixed-size partitions is:

```text
p_D = ceil(E_D / S_emb)
```

The logical GPU frame size is:

```text
S_frame = lambda * S_emb
```

For the current Freebase p32 setup:

```text
E_D ~= 32 GiB
S_emb = 1 GiB
lambda = 2

p_D = 32
S_frame = 2 GiB
```

The scheduler should take `p_D` as an input. It should not assume that `p_D` is a power of two.

### Memory-Derived Admit Cap

For one GPU:

```text
M_gpu      = physical GPU memory
M_reserve  = memory reserved for CUDA/PyTorch/batches/temp tensors
F          = floor((M_gpu - M_reserve) / S_frame)
q          = visible partitions per compute state
H          = F - q non-visible frames
```

The steady-state pipeline has:

```text
q visible frames
a_current stale writeback frames
a_next hidden preload frames
```

The structural frame constraint is:

```text
q + a_current + a_next <= F
```

If all transitions are capped by the same admit count `a_max`:

```text
q + 2*a_max <= F
```

So:

```text
a_max = floor((F - q) / 2)
```

If we require one spare frame:

```text
a_max_spare = floor((F - q - 1) / 2)
```

For the current 24 GiB GPU, p32, 1 GiB embedding partitions, 1 GiB optimizer partitions:

```text
M_gpu = 24 GiB
S_frame = 2 GiB
q = 4

reserve 5 GiB:
  F = floor((24 - 5) / 2) = 9
  a_max = floor((9 - 4) / 2) = 2

reserve 4 GiB:
  F = floor((24 - 4) / 2) = 10
  a_max = floor((10 - 4) / 2) = 3
```

This is the clean way to present the fixed-size partition argument:

```text
fixed partition size -> p_D changes by dataset
fixed GPU memory     -> F changes by partition size and reserve
F and q              -> admit cap
admit cap            -> scheduler family
```

### Dataset-Independent Scheduling Problem

For dataset `D`, let:

```text
P_D = {0, ..., p_D - 1}
```

A visible compute state is:

```text
S_t subset P_D
|S_t| = q
```

The directed bucket set is:

```text
B_D = P_D x P_D
|B_D| = p_D^2
```

A bucket `(i,j)` is compatible with state `S_t` iff:

```text
i in S_t and j in S_t
```

The scheduler produces:

```text
state sequence: S_0, S_1, ..., S_{B-1}
bucket owner map: phi(i,j) in {0, ..., B-1}
```

Correctness constraints:

```text
1. |S_t| = q for every state
2. for every bucket (i,j), i and j are both in S_{phi(i,j)}
3. phi is a function, so every bucket has exactly one owner
4. every bucket in P_D x P_D is owned
5. transition admits satisfy |S_{t+1} - S_t| <= a_max
```

These constraints preserve accuracy because they only change when a bucket is processed, not whether it is processed.

### General Lower Bounds

Coverage lower bound:

```text
B >= C(p_D, q, 2)
```

where `C(v,k,t)` is the covering number: the minimum number of `k`-sets needed so every `t`-set is contained in at least one block.

For our bucket scheduler:

```text
v = p_D partitions
k = q visible partitions per state
t = 2 pair coverage
```

A general lower bound is the Schonheim bound:

```text
B_cover >= ceil(p_D / q * ceil((p_D - 1) / (q - 1)))
```

The simple pair-capacity lower bound is weaker:

```text
B_pair >= ceil(C(p_D,2) / C(q,2))
```

The ordered pipeline lower bound with admit cap `a_max` is:

```text
B_ordered >= 1 + ceil((C(p_D,2) - C(q,2)) /
                      (C(q,2) - C(q - a_max,2)))
```

The scheduler must satisfy:

```text
B >= max(B_cover, B_ordered)
```

For q4 this simplifies to:

```text
B_cover >= ceil(p_D / 4 * ceil((p_D - 1) / 3))

a_max = 3:
  B_ordered >= 1 + ceil((C(p_D,2) - 6) / 6)

a_max = 2:
  B_ordered >= 1 + ceil((C(p_D,2) - 6) / 5)
```

Examples:

```text
p_D = 32, q = 4, a_max = 3:
  B_cover   >= 88
  B_ordered >= 83
  lower     >= 88
  exact C(32,4,2) = 88
  current schedule = 93

p_D = 43, q = 4, a_max = 3:
  B_cover   >= 151
  B_ordered >= 151
  lower     >= 151
  exact C(43,4,2) = 151
```

### Multi-GPU Generalization

For `G` GPUs, use super-rounds:

```text
S_{r,g} subset P_D
|S_{r,g}| = q
```

where:

```text
r = round index
g = GPU/lane index
```

The bucket owner map becomes:

```text
phi(i,j) = (r,g)
```

Correctness is still global:

```text
every directed bucket (i,j) is assigned to exactly one compatible state S_{r,g}
```

Pipeline safety is lane-local:

```text
|S_{r+1,g} - S_{r,g}| <= a_max
```

The minimum number of super-rounds is:

```text
R >= ceil(B / G)
```

where `B` is the required number of q-states. This is only a lower bound because it ignores:

```text
lane continuity
same-round disjointness
edge-work balance
peer relay opportunities
```

Same-round disjointness is desirable when:

```text
G * q <= p_D
```

because it avoids wasting visible capacity on duplicated partitions in the same round.

Peer relay is modeled by:

```text
U_r = union_g S_{r,g}
```

If GPU `g` needs partition `x` at round `r+1`:

```text
x in S_{r+1,g} - S_{r,g}
```

then `x` is peer-relay eligible if:

```text
x in U_r and x not in S_{r,g}
```

So the multi-GPU objective should minimize:

```text
host_admit_bytes + peer_admit_bytes * peer_cost_factor
```

where:

```text
peer_cost_factor < 1
```

because peer copy should be cheaper than host reload.

### Presentation Summary

The fixed-size partition scheduler can be presented as:

```text
Given:
  dataset D
  fixed embedding partition size S_emb
  GPU memory and reserve
  visible width q
  GPU count G

Derive:
  p_D        = ceil(embedding_table_bytes(D) / S_emb)
  S_frame    = lambda * S_emb
  F          = floor((M_gpu - M_reserve) / S_frame)
  a_max      = floor((F - q) / 2)

Construct:
  q-state cover of p_D partitions
  ordered under max admit a_max
  bucket owner map with exact once semantics
  optional multi-GPU lane assignment with peer relay

Validate:
  all p_D^2 directed buckets assigned exactly once
  no incompatible bucket assignment
  max lane admit <= a_max
  frame pressure <= F
  state and round work balanced
```

### Implemented Multi-GPU Bounded Scheduler

The bounded scheduler is now split into two cases:

```text
G = 1:
  build one ordered q-state walk

G > 1:
  build q-state super-rounds directly
```

For multi-GPU, the scheduler does not reuse the single-GPU 93-state walk by simple striping. That striping can put the same visible partition on two GPUs in the same round, which is not safe for exact optimizer-state ownership. Instead it synthesizes rounds:

```text
round r:
  GPU 0: S_{r,0}
  GPU 1: S_{r,1}
  ...
  GPU G-1: S_{r,G-1}
```

with hard constraints:

```text
|S_{r,g}| = q = 4
S_{r,g} cap S_{r,h} = empty, for g != h
|S_{r+1,g} - S_{r,g}| <= a_max
all p^2 directed buckets assigned exactly once
```

For the current `p=32`, `q=4`, `a_max=3` configuration, the validator results are:

```text
2 GPUs:
  states = 94
  rounds = 47
  directed buckets = 1024 exact-once
  max lane admits = 3

3 GPUs:
  states = 93
  rounds = 31
  directed buckets = 1024 exact-once
  max lane admits = 3

4 GPUs:
  states = 96
  rounds = 24
  directed buckets = 1024 exact-once
  max lane admits = 3
```

The multi-GPU state count is not required to stay at the single-GPU 93 states. The 93-state walk is optimized for one lane. Multi-GPU adds another hard constraint: states in the same super-round must be visible-partition disjoint. That constraint can require a different cover with a few more states, but it reduces epoch time by running states in parallel while keeping the swap pipeline safe.

In code, the path is:

```text
GEGE_BOUNDED_GREEDY_COVER_Q4=1
requested_active_devices > 1
  -> getBoundedGreedyCoverMultiGpuEdgeBucketOrdering(...)
  -> compileMultiGpuStateflowPlan(...)
  -> validateStateflowPlanExactSemantics(...)
```

The scheduler first builds the multi-GPU rounds, then Stateflow accepts the input rounds as a candidate before trying older regrouping heuristics. This is important: older regrouping can destroy the lane-local overlap guarantee.
