# Process Discovery from Sensor Streams via Rule Synthesis

## Problem

Process mining discovers process models from event logs recorded by information systems.
A fundamental assumption is that events are discrete, named activities — high-level abstractions such as *"Part Assembled"* or *"Patient Admitted"* — that carry clear semantic meaning within a business context.

When process mining meets the Internet of Things, this assumption breaks down.
IoT devices produce continuous or high-frequency numerical sensor streams: temperature readings, vibration amplitudes, pressure curves.
These streams do not contain activities.
To apply classical process discovery, practitioners must first transform raw sensor data into activity-labeled event logs through a pipeline of segmentation, feature extraction, clustering, and labeling.

This abstraction pipeline is the central bottleneck.
It requires substantial domain knowledge to define meaningful segments, to select discriminative features, and — most critically — to assign activity labels that faithfully represent what the physical system is doing.
The choice of abstraction determines what the discovered model can express.
A poor abstraction renders the entire downstream analysis meaningless, yet validating the abstraction itself is largely a manual, expert-driven effort.
In practice, this means that process mining over IoT data is only as good as the domain expert's ability to hand-craft the right event abstraction — a requirement that limits scalability, reproducibility, and applicability across domains.

## Formal Background

The theoretical foundation of process discovery rests on a language-theoretic view.
A generative system $G$ (e.g., an information system executing a business process) produces behavior that can be characterized as a language $\mathcal{L}(G)$ over an alphabet of activities.
An event log $L$ is a finite sample of this language: $L \subseteq \mathcal{L}(G)$.
A discovery algorithm takes $L$ and produces a process model $M$ whose language $\mathcal{L}(M)$ approximates $\mathcal{L}(G)$.
Quality criteria — fitness, precision, generalization, simplicity — measure how well $\mathcal{L}(M)$ relates to $\mathcal{L}(G)$ given the sample $L$.

The key insight is that $M$ is an abstraction of reality.
It does not reproduce $G$ exactly; it captures enough structure so that replaying $M$ produces behavior consistent with what $G$ would produce.
The alphabet of activities is given, and the discovery algorithm's task is to find the ordering constraints among them.

## Idea

We propose to transfer this language-theoretic framework directly to sensor streams by replacing the hand-crafted activity alphabet with an automatically synthesized rule alphabet.

The generative system is now a sensor network $G_s$ that produces signals in a continuous signal space $\mathcal{S}$.
A sensor log $\sigma$ is a finite, sampled recording of this signal space: $\sigma \subseteq \mathcal{S}$.
The missing piece is the alphabet: in classical process mining, activities are given; in the sensor setting, they must be derived.

Our key idea is to use automated rule synthesis to construct this alphabet.
Rules are predicates over sensor variables that characterize temporal segments of the sensor stream — for example, *"temperature $\in [60, 80]$ $\wedge$ vibration $> 1.5$"* or *"pressure is monotonically increasing $\wedge$ flow rate $< 0.2$"*.
Each synthesized rule corresponds to a distinguishable physical regime in the sensor data.
These rules take the role of activities: they are the letters of the alphabet over which traces are formed.

Once the rule alphabet is established, the temporal sequence of rule activations in the sensor log yields traces in the classical sense.
Standard process discovery algorithms can then be applied to these traces to produce a process model $M_s$ whose language $\mathcal{L}(M_s)$ approximates the behavior of the sensor network, just as a classically discovered model approximates the behavior of an information system.

The goal of the discovered model is the same as in classical process mining: if we replay $M_s$, it should generate behavior similar to what the sensor network produces.
The difference is that the alphabet is not prescribed by domain experts but derived from the data itself.

## Solution Outline

The approach consists of three steps:

1. **Rule Synthesis.** Given a sensor log $\sigma$, automatically synthesize a set of rules $\mathcal{R} = \{r_1, \ldots, r_k\}$ that partition or characterize the temporal segments of $\sigma$. Each rule $r_i$ is a predicate over sensor variables. The synthesis combines change point detection, abductive segment merging, and SyGuS-based predicate synthesis (detailed below). The result is an alphabet $\Sigma = \mathcal{R}$.

2. **Trace Construction.** Translate the sensor log $\sigma$ into a set of traces over $\Sigma$. Each time point (or segment) in $\sigma$ is mapped to the rule $r_i \in \mathcal{R}$ that holds at that point. Consecutive time points where the same rule holds are merged into a single event. The resulting sequence of rule activations forms a trace. Repeated recordings or natural cycles in the data provide multiple traces, constituting an event log $L_s$ over $\Sigma$.

3. **Process Discovery.** Apply a standard discovery algorithm (e.g., Inductive Miner, ILP Miner, or region-based synthesis) to $L_s$ to obtain a process model $M_s$. The semantics of $M_s$ are standard Petri net semantics: places represent intermediate states between physical regimes, transitions represent regime changes, and the firing sequence corresponds to the temporal evolution of the sensor network.

## The Synthesis Problem

The central technical challenge is Step 1: synthesizing the rule alphabet $\mathcal{R}$ from a sensor log $\sigma$.
This section formalizes the synthesis problem and proposes a specification-driven approach that combines change point detection as a candidate generator, abductive reasoning for segment merging, and Syntax-Guided Synthesis (SyGuS) for predicate construction.

### Problem Statement

Let $\sigma = (v_1(t), \ldots, v_n(t))_{t \in [0, T]}$ be a multivariate sensor log over $n$ sensor variables sampled at discrete time points $t_0, t_1, \ldots, t_N$.
We seek a set of rules $\mathcal{R} = \{r_1, \ldots, r_k\}$, where each rule $r_i$ is a predicate over $v_1, \ldots, v_n$, satisfying the following correctness specification:

1. **Coverage.** Every time point is described by at least one rule:
$$\forall t \in \{t_0, \ldots, t_N\} : \bigvee_{i=1}^{k} r_i(\sigma(t))$$

2. **Mutual exclusivity.** No two rules hold simultaneously:
$$\forall t \in \{t_0, \ldots, t_N\}, \forall i \neq j : \neg(r_i(\sigma(t)) \wedge r_j(\sigma(t)))$$

3. **Temporal coherence.** Each rule holds over contiguous temporal segments rather than scattered individual time points.
Formally, for each rule $r_i$, the set $\{t \mid r_i(\sigma(t))\}$ consists of a small number of maximal contiguous intervals relative to the length of $\sigma$.

Together, properties (1) and (2) require that $\mathcal{R}$ partitions the sensor value space into $k$ regions.
Property (3) ensures that these regions correspond to temporally stable physical regimes rather than noise-level oscillations.

### Grammar

The expressiveness of the synthesized rules is controlled by a grammar $\mathcal{G}$ that defines the space of candidate predicates.
For sensor data, a physically motivated and computationally tractable grammar restricts rules to conjunctions of interval constraints over individual sensor variables:

$$r ::= c_1 \wedge \ldots \wedge c_m$$
$$c ::= v_i \leq \alpha \mid v_i \geq \alpha \mid v_i \in [\alpha, \beta]$$

where $\alpha, \beta \in \mathbb{R}$ are constants to be synthesized.
This grammar defines an axis-aligned hyperrectangular decomposition of the $n$-dimensional sensor space.
It is expressive enough to capture the regime structure of typical physical processes — where regimes are characterized by sensor variables falling within specific operating ranges — while keeping the synthesis problem decidable and the resulting rules directly interpretable as guards in a Data Petri Net.

The grammar can be extended to include rate-of-change predicates (e.g., $\dot{v}_i > \alpha$) or temporal operators if the application domain requires it, without changing the overall framework.

### Synthesis Procedure

Applying SyGuS directly to the full time series is computationally infeasible for realistic sensor logs.
We therefore decompose the synthesis into three phases, where change point detection provides candidate boundaries, abductive reasoning selects the minimal set of regimes that explains the observed segments, and SyGuS synthesizes the corresponding predicates.

**Phase 1: Change point detection (candidate generation).**
Apply a standard change point detection method (e.g., PELT, BOCPD, or variance-based segmentation) to identify temporal boundaries $\tau_0 = t_0, \tau_1, \ldots, \tau_m, \tau_{m+1} = t_N$ that partition the time series into $m+1$ segments.
This step is intentionally configured to over-segment: it is better to propose too many boundaries than too few, because the subsequent abductive step will merge segments that belong to the same regime.
The role of change point detection is solely to provide candidate segment boundaries, not to assign labels or semantics.

**Phase 2: Abductive segment merging.**
The over-segmentation from Phase 1 produces $m+1$ candidate segments $s_0, \ldots, s_m$.
For each segment $s_j$, compute the empirical value profile:
$$\text{profile}(s_j) = \left([\min_{t \in s_j} v_1(t), \max_{t \in s_j} v_1(t)], \ldots, [\min_{t \in s_j} v_n(t), \max_{t \in s_j} v_n(t)]\right)$$

The task is now an abductive one: given the set of observed segment profiles as observations and the grammar $\mathcal{G}$ as the theory, find the minimal set of hypotheses (regime classes) $E_1, \ldots, E_k$ such that every segment is explained by exactly one hypothesis.

Formally, the abductive problem is:

- **Observations:** The set of segment profiles $\{\text{profile}(s_0), \ldots, \text{profile}(s_m)\}$.
- **Theory:** The grammar $\mathcal{G}$ of admissible predicates. Two segments $s_a$ and $s_b$ are *compatible* — i.e., can be explained by the same hypothesis — if and only if there exists a predicate $p \in \mathcal{G}$ such that $p(\text{profile}(s_a)) = \top$ and $p(\text{profile}(s_b)) = \top$.
- **Abductive goal:** Find a partition of the segments into equivalence classes $E_1, \ldots, E_k$ such that:
  1. All segments within each class $E_i$ are pairwise compatible.
  2. The number of classes $k$ is minimal (parsimony criterion).

The parsimony criterion — minimizing $k$ — serves as the abductive preference: among all valid explanations, select the one with the fewest hypotheses.
This directly controls the granularity of the discovered model and encodes an Occam's razor principle: do not introduce a new regime unless the data forces it.

Computationally, this reduces to a minimum clique cover problem on a compatibility graph where segments are nodes and edges connect compatible pairs.
While minimum clique cover is NP-hard in general, the problem instances arising from sensor data are typically small (tens to hundreds of segments) and exhibit structure (profiles from the same physical regime cluster tightly) that makes them tractable in practice.

**Phase 3: SyGuS-based rule synthesis.**
For each pair of abductively determined equivalence classes $E_i, E_j$ with $i \neq j$, synthesize a discriminating predicate $d_{ij}$ in grammar $\mathcal{G}$ such that:
$$\forall s \in E_i : d_{ij}(\text{profile}(s)) = \top \quad \wedge \quad \forall s \in E_j : d_{ij}(\text{profile}(s)) = \bot$$

The rule for class $E_i$ is then constructed as:
$$r_i = \bigwedge_{j \neq i} d_{ij}$$

The SyGuS specification for each $d_{ij}$ is a finite set of positive and negative examples derived from the segment profiles.
Because the SyGuS instances operate on segment summaries rather than raw time points, the problem size is reduced from $N$ (number of time points, potentially millions) to $m$ (number of segments, typically tens to hundreds).

### Correctness

The three-phase decomposition preserves the partitioning properties as follows.

**Coverage** holds by construction: every time point belongs to some segment $s_j$, every segment belongs to some abductively determined equivalence class $E_i$, and every equivalence class is characterized by a rule $r_i$.

**Mutual exclusivity** is guaranteed by the discriminating predicates: for any two classes $E_i \neq E_j$, the predicate $d_{ij}$ separates them, so $r_i$ and $r_j$ cannot hold simultaneously on any segment profile that belongs to one of the classes.

**Temporal coherence** is inherited from the change point detection: since segments are contiguous intervals by construction, and each segment maps to exactly one rule via the abductive assignment, rule activations are contiguous.

The abductive step strengthens the correctness argument compared to heuristic grouping.
A heuristic equivalence-class assignment (e.g., profile-distance-based clustering) may group segments that are not separable by any predicate in $\mathcal{G}$, leading to a synthesis failure in Phase 3.
The abductive formulation prevents this: two segments are only placed in the same class if the grammar guarantees the existence of a covering predicate.
Conversely, two segments are only placed in different classes if they are not jointly coverable — ensuring that every class distinction corresponds to a real, synthesizable boundary in the predicate space.

Note that the change point detection step remains heuristic: a missed change point merges two segments, producing a coarser rule that covers a wider operating range.
This affects the granularity of the discovered model but not its correctness — the partitioning properties still hold over whatever segmentation is provided.
A spurious change point introduces an unnecessary segment boundary but, since the abductive step will merge both resulting sub-segments into the same equivalence class, it does not introduce a spurious rule.
This is a key advantage of the hybrid approach: over-segmentation in Phase 1 is harmless because the abductive merging in Phase 2 absorbs it.

### Why SyGuS over Heuristic Alternatives?

Decision trees, clustering algorithms, and other data-driven methods can produce axis-aligned partitions of the sensor space with similar discriminative power.
The choice of SyGuS is motivated by three properties that these alternatives lack:

First, the correctness specification (coverage, mutual exclusivity) is explicitly encoded and verified by the solver, not empirically checked on a test set.
The synthesis either produces a correct partitioning or reports that none exists in the given grammar — there is no silent failure mode.

Second, the synthesized rules are logical formulas that can be directly embedded as transition guards in a Data Petri Net.
No translation or approximation step is needed between the synthesis output and the process model.

Third, the grammar $\mathcal{G}$ is a modular extension point.
Enriching the rule language — for instance, adding rate-of-change predicates, temporal patterns, or cross-sensor correlations — requires only extending $\mathcal{G}$ without modifying the synthesis framework or the correctness specification.

### Why Abduction over Heuristic Grouping?

Replacing the heuristic equivalence-class assignment with an abductive formulation provides three specific advantages:

First, the abductive step is *grammar-aware*: it only merges segments that are jointly coverable by a predicate in $\mathcal{G}$.
Heuristic grouping (e.g., distance-based clustering on profiles) is oblivious to the grammar and may produce classes that are not separable in the predicate language, causing Phase 3 to fail or produce imprecise rules.

Second, the parsimony criterion (minimize $k$) provides a principled, formal basis for choosing the abstraction level.
Heuristic thresholds (e.g., a distance cutoff for clustering) are arbitrary and domain-dependent; the abductive formulation derives the number of regimes from the structure of the data and the expressiveness of the grammar.

Third, over-segmentation in Phase 1 becomes a feature rather than a liability.
With heuristic grouping, spurious change points introduce noise into the clustering step.
With abductive merging, spurious boundaries are absorbed: the abductive step recognizes that the resulting sub-segments are compatible and merges them, making the pipeline robust to the sensitivity settings of the change point detector.

## Why Not a Neural Network?

A predictable objection is that generative neural networks (e.g., variational autoencoders, diffusion models, or transformers) already learn pattern-based models from sensor data that can be sampled to generate new data.
The distinction is not merely one of interpretability.
It is one of analytical semantics.

A Petri net model provides compositional, formal structure that enables the downstream analytical tasks that process mining exists to support:

- **Conformance checking:** A new sensor log can be aligned against $M_s$ with a precise, computable deviation cost. Neural networks offer likelihoods, not alignments.
- **Concurrency:** The model can express that two physical subsystems operate independently. A neural network's latent space does not decompose into concurrent components.
- **Decidable properties:** Soundness, liveness, and boundedness of $M_s$ are decidable and carry physical meaning — a deadlock in the model indicates a sensor regime from which the system cannot progress.
- **Locality:** If a sensor fails or a component is replaced, the effect is traceable to specific rules and specific transitions rather than requiring full retraining.

Neural networks model the distribution of the data.
Petri nets model the mechanism that produces the data.
For process analytics — conformance, root cause analysis, prediction, optimization — mechanism is what is needed.

## Open Challenges

- **Case Notion.** Sensor streams are continuous. Defining what constitutes a single process instance (case) remains an open problem. In many IoT domains, natural cycles (machine cycles, batch runs, shifts) provide implicit case boundaries that can be detected as recurrences of an initial rule or a reset pattern in the rule sequence.
- **Rule Granularity.** The number and specificity of synthesized rules determine the abstraction level of the model. The abductive parsimony criterion provides a principled default — minimize the number of regimes — but domain-specific preferences may require a different trade-off. Too few rules yield an overly coarse model; too many yield an overfitting model that does not generalize. This mirrors the precision-generalization trade-off in classical discovery but is now coupled to the rule synthesis step.
- **Multi-Sensor Composition.** When multiple sensor subsystems interact, rules may span different sensor subsets. The compositional structure of the discovered model should reflect the physical modularity of the sensor network.