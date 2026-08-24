# Project-Muse: Multi-UAV Swarm Exploration using Reinforcement Learning

**MUSE** is a research framework for autonomous exploration of unknown environments using a swarm of unmanned aerial vehicles (UAVs). The project combines classical frontier-based exploration with reinforcement learning (PPO) to achieve efficient, scalable, and adaptive multi-robot coordination.

---

## 1. Motivation

Exploration of unknown, GPS-denied environments (e.g., disaster zones, caves, forests) is a critical task for search-and-rescue, surveillance, and mapping. Single-robot systems are often slow and lack robustness. Swarms of UAVs offer parallelism and redundancy, but effective coordination remains challenging due to limited communication, partial observability, and complex trade-offs between exploration and exploitation.

Classical heuristics such as nearest-frontier and greedy selection are useful but can be myopic and may not adapt their decision criteria as the exploration state changes. Reinforcement learning (RL) provides a data-driven framework for learning policies that maximize cumulative task reward.

This project investigates a **hybrid approach**: classical frontier detection, clustering, utility-based assignment, and path planning are retained for interpretability and structured coordination, while RL is used at a higher level to learn the weights of the utility function.

---

## 2. Problem Formulation

We consider a swarm of $N$ homogeneous UAVs operating in a partially known 2D grid world. The environment is initially unknown except for the information revealed through the UAVs' sensing.

Each UAV has:

- A position on the grid.
- A local/global explored occupancy map maintained by the simulator.
- A circular sensing radius.
- Limited communication range for neighboring-agent information.

The objective is to efficiently explore the environment and reach a predefined coverage target (e.g., 90%) while minimizing unnecessary movement and redundant sensing.

At each decision step, frontier clusters are evaluated using:

<p align="center">
  <b>
    U<sub>i,c</sub> =
    α IG<sub>i,c</sub> −
    β C<sub>i,c</sub> −
    γ R<sub>i,c</sub> +
    δ S<sub>c</sub>
  </b>
</p>

<!--
$$
U_{i,c} = \alpha IG_{i,c} - \beta C_{i,c} - \gamma R_{i,c} + \delta S_c
$$
-->

where:

- **IG<sub>i,c</sub>** = normalized information gain.
- **C<sub>i,c</sub>** = normalized path cost.
- **R<sub>i,c</sub>** = predicted sensing redundancy.
- **S<sub>c</sub>** = normalized cluster size.
- **α, β, γ, δ** = utility weights.

The current classical system uses fixed weights. The RL extension learns these weights dynamically from the exploration state.

---

## 3. Environment

The simulation environment (`environment/`) is built on a custom grid-based engine that supports:

- Random map generation with obstacles.
- A ground-truth occupancy grid.
- A robot occupancy grid containing currently discovered information.
- Circular sensing around each UAV.
- Limited communication radius between UAVs.
- Multiple UAVs exploring the same environment.
- Multiple random seeds for statistical evaluation.

### Default parameters

| Parameter | Value |
|---|---:|
| Grid size | $100 \times 100$ |
| Number of UAVs | 5 |
| Obstacle percentage | 10% |
| Sensor radius | 3 cells |
| Communication radius | 10 cells |
| Coverage target | 90% |
| Map generation | Seeded |

---

## 4. Frontier Detection

Frontiers are boundaries between explored free space and unknown cells. They represent candidate regions for further exploration.

The frontier detection module identifies cells that are adjacent to unexplored regions and produces frontier candidates from the currently known robot map.

These frontier cells are subsequently grouped into clusters to reduce the number of candidate assignments and encourage spatially coherent exploration.

---

## 5. Frontier Clustering

Frontier clustering groups nearby frontier cells into spatial regions.

Each cluster contains:

- A set of frontier cells.
- A centroid.
- A cluster size.
- An estimated information gain.

Instead of having every UAV independently select an individual frontier, the coordination layer evaluates **drone-to-cluster assignments**.

This is useful for multi-UAV exploration because it allows the system to reason about regions rather than isolated frontier cells.

---

## 6. Information Gain

Information gain estimates how much unexplored information is associated with a frontier cluster.

The current implementation evaluates neighboring cells around frontier cells:

$$
IG_c =
\frac{
\text{unexplored neighboring cells}
}{
\text{valid neighboring cells}
}
$$

The resulting value is normalized to a bounded range and used as one component of the utility function.

Cluster size is maintained as a separate utility component rather than being directly added to information gain.

---

## 7. Utility Formulation

The utility function balances exploration benefit against movement and redundancy costs:

<p align="center">
  <b>
    U<sub>i,c</sub> =
    α.IG<sub>i,c</sub> −
    β.C<sub>i,c</sub> −
    γ.R<sub>i,c</sub> +
    δ.S<sub>c</sub>
  </b>
</p>

where:

- <b>α</b> controls the importance of information gain.
- <b>β</b> controls the penalty for path cost.
- <b>γ</b> controls the penalty for predicted sensing redundancy.
- <b>δ</b> controls the preference for larger frontier clusters.

The metrics are normalized before being combined so that one term does not dominate the utility simply because it has a larger numerical scale.

The utility implementation supports dynamic weight updates:

```python
utility.set_weights(
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    delta=delta,
)
```

This interface is used by the RL layer to provide state-dependent weights.

---

## 8. Assignment

### 8.1 Greedy assignment

The initial cluster-based approach evaluates the utility of candidate clusters for each UAV and assigns clusters greedily.

This provides a useful classical baseline but does not optimize the total swarm assignment jointly.

### 8.2 Hungarian assignment

The Hungarian algorithm is used to construct a global drone-to-cluster assignment.

The system builds a utility matrix:

$$
U \in \mathbb{R}^{N \times M}
$$

where each element represents the utility of assigning UAV $i$ to cluster $c$.

Because the Hungarian algorithm solves a minimization problem, utilities are converted into assignment costs by negating them.

The resulting assignment maximizes total swarm utility while ensuring that each UAV receives at most one cluster and each cluster is assigned at most once when sufficient clusters are available.

---

## 9. Path Planning

Two grid-based path planners have been implemented.

### BFS

Breadth-First Search is used as a baseline shortest-path planner on the known free-space map.

### A*

A* uses Manhattan distance as its initial heuristic:

$$
h(n) = |x_n - x_g| + |y_n - y_g|
$$

The planner uses the same four-connected grid motion model as BFS.

A* also records the number of nodes expanded during path planning, allowing computational comparisons with BFS.

Preliminary experiments indicate that A* and BFS produce broadly similar exploration performance in the current environment, while their computational behavior can differ depending on the map and target.

For the initial RL experiments, **BFS is retained as the default planner** to keep the focus on adaptive utility weighting.

---

## 10. Evaluation Metrics

The simulator records:

- **Coverage:** fraction of cells discovered.
- **Exploration time:** steps required to reach 90% coverage.
- **Total travelled distance:** sum across UAVs.
- **Sensing redundancy:** redundant sensing operations as a fraction of total sensing.
- **Visit overlap:** repeated physical visits to cells by multiple UAVs.
- **Movement efficiency:** unique visited cells per unit of total travel distance.
- **Mean pairwise distance:** average Euclidean separation between UAV pairs.
- **Nodes expanded:** search nodes expanded by the path planner.

All final results will be reported over multiple random seeds using mean and standard deviation.

---

# 11. Classical Baselines

The current classical progression is:

```text
Random
   ↓
Nearest Frontier
   ↓
Greedy Frontier
   ↓
Cluster Frontier
   ↓
Cluster + Utility
   ↓
Hungarian + Utility
   ↓
Hungarian + Utility + A*
```

The comparison separates:

**Assignment**
- Random
- Nearest frontier
- Greedy
- Hungarian

**Utility**
- No utility
- Fixed-weight utility
- Normalized weighted utility

**Path planning**
- BFS
- A*

---

# 12. Classical Results

> **Status: TBD — final values will be populated after multi-seed evaluation.**

| Strategy | Time to 90% ↓ | Distance ↓ | Sensing Redundancy ↓ | Visit Overlap ↓ | Movement Efficiency ↑ | Nodes Expanded ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Random | TBD | TBD | TBD | TBD | TBD | TBD |
| Nearest Frontier | TBD | TBD | TBD | TBD | TBD | TBD |
| Greedy Frontier | TBD | TBD | TBD | TBD | TBD | TBD |
| Cluster Frontier | TBD | TBD | TBD | TBD | TBD | TBD |
| Cluster + Utility | TBD | TBD | TBD | TBD | TBD | TBD |
| Hungarian + Utility | TBD | TBD | TBD | TBD | TBD | TBD |
| Hungarian + Utility + A* | TBD | TBD | TBD | TBD | TBD | TBD |

### Coverage vs. Steps

> **Plot: TBD**

`plots/classical/coverage_comparison.png`

### Sensing Redundancy

> **Plot: TBD**

`plots/classical/sensing_redundancy.png`

### Visit Overlap

> **Plot: TBD**

`plots/classical/visit_overlap.png`

### Movement Efficiency

> **Plot: TBD**

`plots/classical/movement_efficiency.png`

### BFS vs. A* Nodes Expanded

> **Plot: TBD**

`plots/classical/bfs_vs_astar_nodes.png`

---

# 13. Reinforcement Learning Formulation

The next stage introduces **Proximal Policy Optimization (PPO)** to learn the utility weights.

Importantly, PPO does **not replace the Hungarian assignment or path planner**.

The intended architecture is:

```text
Exploration State
       ↓
      PPO
       ↓
[α, β, γ, δ]
       ↓
 Utility Function
       ↓
 Utility Matrix
       ↓
   Hungarian
       ↓
      BFS
       ↓
 UAV Exploration
       ↓
 State + Reward
       ↺
```

PPO therefore acts as a high-level adaptive weighting mechanism.

### State

The state will contain compact features describing the current exploration situation, such as:

- Current coverage.
- Number of frontier clusters.
- Information-gain statistics.
- Path-cost statistics.
- Predicted redundancy.
- Cluster-size statistics.
- Sensing redundancy.
- Visit overlap.
- UAV spatial distribution.

The exact state representation is **TBD** and will be finalized during RL experiments.

### Action

The PPO policy outputs four continuous utility weights:

<p align="center">
  <strong>
    a<sub>t</sub> = [α<sub>t</sub>, β<sub>t</sub>, γ<sub>t</sub>, δ<sub>t</sub>]
  </strong>
</p>

The action is passed to:

```python
utility.set_weights(...)
```

### Reward

The reward measures actual exploration performance rather than the numerical value of the handcrafted utility.

An initial candidate formulation is:

<p align="center">
  <strong>
    r<sub>t</sub> =
    w<sub>c</sub> ΔCoverage −
    w<sub>d</sub> ΔDistance −
    w<sub>r</sub> ΔRedundancy
  </strong>
</p>

The reward coefficients are **TBD** and will be documented after initial experiments.

The key distinction is:

> **Utility determines which assignment is selected; reward determines whether the resulting exploration behavior was good.**

---

# 14. PPO Training

The first implementation will use a standard PPO implementation rather than reimplementing PPO from scratch.

The RL environment will expose:

- A continuous four-dimensional action space for $[\alpha,\beta,\gamma,\delta]$.
- A continuous observation space containing exploration-state features.
- A reward based on exploration performance.
- Episode termination when the coverage target is reached or the maximum number of steps is exceeded.

The PPO policy learns:

$$
\pi_\theta(s_t)
\rightarrow
[\alpha_t,\beta_t,\gamma_t,\delta_t]
$$

rather than a fixed global set of weights.

This allows the utility function to adapt its priorities as exploration progresses.

### Training configuration

| Parameter | Value |
|---|---:|
| PPO implementation | TBD |
| Learning rate | TBD |
| Discount factor | TBD |
| GAE λ | TBD |
| Clip range | TBD |
| Rollout length | TBD |
| Batch size | TBD |
| Training steps | TBD |

---

# 15. PPO Results

> **Status: TBD — results will be populated after training.**

PPO will be evaluated on map seeds that were not used during training.

| Strategy | Time to 90% ↓ | Distance ↓ | Sensing Redundancy ↓ | Visit Overlap ↓ | Movement Efficiency ↑ |
|---|---:|---:|---:|---:|---:|
| Fixed-weight Hungarian | TBD | TBD | TBD | TBD | TBD |
| PPO + Hungarian | TBD | TBD | TBD | TBD | TBD |

### PPO Learning Curve

> **Plot: TBD**

`plots/rl/training_reward.png`

### PPO vs. Classical Coverage

> **Plot: TBD**

`plots/rl/coverage_comparison.png`

### Learned Utility Weights

> **Plot: TBD**

`plots/rl/learned_weights.png`

### Generalization to Unseen Seeds

> **Plot: TBD**

`plots/rl/unseen_seed_evaluation.png`

No performance improvement over the classical baseline will be claimed until the corresponding multi-seed evaluation has been completed.

---

# 16. Ablation Studies

Planned ablations include:

### Utility ablation

```text
IG only
IG + Path Cost
IG + Path Cost + Redundancy
IG + Path Cost + Redundancy + Cluster Size
```

### Assignment ablation

```text
Greedy
Hungarian
```

### Planner ablation

```text
BFS
A*
```

### Learning ablation

```text
Fixed weights
Optimized fixed weights (optional)
PPO adaptive weights
```

The goal is to determine whether improvements come from:

1. the utility formulation,
2. global assignment,
3. path planning,
4. or adaptive weight learning.

---

# 17. Limitations

The current simulator has several limitations:

- **Simulation gap:** the current environment is a 2D grid-based simulator and does not yet model full 3D UAV dynamics.
- **Simplified sensing:** UAVs currently use a circular sensing radius rather than a realistic sensor model.
- **Static environment:** obstacles are currently static.
- **Simplified communication:** communication is represented using a fixed spatial radius.
- **Centralised coordination:** the current Hungarian assignment is centrally computed.
- **Reward engineering:** the RL reward is initially hand-designed and may require further investigation.
- **Computational cost:** evaluating many drone-cluster pairs can become expensive as the number of clusters increases.
- **Generalization:** robustness to substantially different maps and obstacle distributions remains to be evaluated.

---

# 18. Future Work

Potential extensions include:

- **Adaptive multi-agent coordination:** move toward decentralized or distributed assignment.
- **Decentralised PPO:** learn policies that rely on local observations and limited communication.
- **3D environment:** extend the simulator to 3D occupancy grids.
- **Realistic UAV dynamics:** incorporate velocity, acceleration, turning constraints and energy consumption.
- **Dynamic obstacles:** incorporate moving obstacles and online replanning.
- **Hierarchical RL:** separate high-level exploration decisions from low-level navigation.
- **Transfer learning:** evaluate sim-to-real transfer using domain randomization.
- **Multi-objective optimization:** incorporate battery, safety and communication constraints.
- **Learned path planning:** investigate whether learned planners can complement or replace classical search.
- **Explainability:** analyze how learned utility weights change according to exploration state.

---

# 19. Project Status

### Completed

- [x] 2D multi-UAV exploration simulator
- [x] Occupancy-grid environment
- [x] Multi-UAV sensing
- [x] Frontier detection
- [x] Frontier clustering
- [x] Information-gain estimation
- [x] Utility-based assignment
- [x] Utility normalization
- [x] Greedy assignment baseline
- [x] Hungarian assignment
- [x] BFS path planning
- [x] A* path planning
- [x] Sensing redundancy metric
- [x] Visit-overlap metric
- [x] Movement-efficiency metric
- [x] Multi-run experiment framework

### In Progress

- [ ] Final classical baseline evaluation
- [ ] PPO environment
- [ ] PPO training
- [ ] Learned utility weights
- [ ] PPO evaluation on unseen maps
- [ ] Ablation studies
- [ ] Final plots and statistical analysis

---

# 20. Citation

If you use this framework in your research, please cite:

```bibtex
@software{ProjectMuse2026,
  author = {Sudhansu, ...},
  title = {Project-Muse: Multi-UAV Swarm Exploration using Reinforcement Learning},
  year = {2026},
  url = {https://github.com/sudhansu3299/Project-Muse}
}
```

---

# License

Distributed under the MIT License. See `LICENSE` for more information.

---

# Contact

Project Link: [https://github.com/sudhansu3299/Project-Muse](https://github.com/sudhansu3299/Project-Muse)
