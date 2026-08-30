# Project-Muse: Multi-UAV Swarm Exploration
### Learning State-Dependent Utility Weights for Multi-Robot Frontier Exploration


MUSE is a modular benchmarking framework for autonomous multi-UAV exploration of unknown environments. It combines frontier-based coordination, path-aware utility assignment, Hungarian task allocation, BFS path planning, and PPO-based learning of state-dependent utility weights.

---

## 1. Motivation

Exploration of unknown, GPS-denied environments is a fundamental problem in
robotics, with applications in search-and-rescue, disaster response,
surveillance, and autonomous mapping. Multi-robot systems can accelerate
exploration by distributing sensing and navigation across multiple agents, but
effective coordination requires balancing competing objectives such as
information gain, travel cost, sensing redundancy, and the spatial structure
of unexplored regions.

Classical frontier-based exploration provides a structured and interpretable
solution, but many coordination strategies rely on manually selected,
fixed preferences between these competing objectives. A weighting that is
effective early in exploration may not remain appropriate as the environment
becomes increasingly explored.

### Research Idea

This project investigates whether **reinforcement learning can learn
state-dependent coordination preferences rather than directly learning robot
motion**.

Instead of replacing the classical exploration pipeline with an end-to-end
RL controller, PPO operates at a higher level and dynamically determines the
relative importance of different frontier-selection objectives:

$$
\boxed{
[\alpha_t,\beta_t,\gamma_t,\delta_t] = \pi_\theta(s_t)
}
$$

The learned weights are used by a path-aware frontier utility function, while
classical task assignment and path planning remain unchanged.

This creates a hybrid architecture:

**Classical baseline**

`State → Fixed Weights → Frontier Utility → Hungarian → BFS`

**Proposed approach**

`State → PPO → Dynamic Weights → Frontier Utility → Hungarian → BFS`

The central question is therefore:

> **Can state-dependent learned utility weighting improve multi-robot
> exploration efficiency and generalization compared with fixed-weight
> frontier coordination?**

The project also provides a modular benchmarking framework in which frontier
detection, clustering, task assignment, utility functions, and path planning
can be independently evaluated and compared.

---

## 2. Problem Formulation

We consider a team of $N$ homogeneous UAVs exploring an initially unknown
2D grid environment. At each coordination step, the system identifies
frontier regions representing boundaries between explored and unexplored
space and groups them into frontier clusters.

For UAV $i$ and frontier cluster $c$, the assignment utility is defined as

$$
U_{i,c} = 
\alpha\*IG_{i,c}
-\beta\*C_{i,c}
-\gamma\*R_{i,c}
+\delta\*S_c
$$

where:

- $IG_{i,c}$ is the normalized information gain associated with assigning
  UAV $i$ to cluster $c$.
- $C_{i,c}$ is the normalized path cost from UAV $i$ to cluster $c$.
- $R_{i,c}$ is the predicted sensing redundancy.
- $S_c$ is the normalized size of frontier cluster $c$.
- $\alpha,\beta,\gamma,\delta$ determine the relative importance of these
  objectives.

The assignment is represented by binary variables

$$
x_{i,c}\in\{0,1\},
$$

where $x_{i,c}=1$ indicates that UAV $i$ is assigned to frontier cluster $c$.

The coordination objective can therefore be expressed as

$$
\max_X
\sum_i\sum_c x_{i,c}U_{i,c},
$$

subject to the task-assignment constraints.

The resulting assignment problem is solved using the Hungarian algorithm.
Once assignments are determined, BFS is used to generate grid-based paths
for the selected UAV-frontier pairs.

### Fixed-weight coordination

The classical baseline uses a constant parameter vector:

$$
\mathbf{w}_{fixed}=
[\alpha,\beta,\gamma,\delta].
$$

The same weights are used throughout an episode.

### State-dependent coordination

The proposed approach replaces the fixed parameter vector with a policy:

$$
\mathbf{w}_t = [\alpha_t,\beta_t,\gamma_t,\delta_t] = \pi_\theta(s_t)
$$


where $s_t$ represents the current exploration state and $\pi_\theta$ is
the PPO policy.

Thus, the underlying assignment and path-planning mechanisms remain the same;
the learned component changes only the **relative priorities used to evaluate
candidate frontier assignments**.

This formulation isolates the research question of whether learning the
utility weights provides an advantage over manually fixed coordination
preferences.

---

## 3. Environment

The simulation environment (`environment/`) is a custom grid-based multi-UAV
simulator designed to evaluate exploration and coordination strategies under
controlled and reproducible conditions.

Each episode consists of a randomly generated 2D occupancy grid containing
obstacles. The simulator maintains both the ground-truth map and the partial
map revealed to the UAV team through sensing. UAVs incrementally discover
unexplored regions as they navigate through the environment.

The simulator supports:

- Randomized obstacle-map generation.
- Ground-truth and partially observed occupancy maps.
- Circular sensing around each UAV.
- Multi-UAV exploration of a shared environment.
- Configurable inter-UAV communication range.
- Reproducible experiments through deterministic random seeds.
- Interchangeable exploration and coordination strategies.

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
## 4. Approach
### 4.1 Modular Exploration Pipeline
We compare a classical frontier-based coordination pipeline (fixed weights) with a learning enhanced pipeline where PPO adapts the utility weights online. 

<img width="1009" height="478" alt="image" src="https://github.com/user-attachments/assets/c0812ab8-0d88-4243-8783-224e3ef578d6" />

Both the pipelines use the same shared modular codebase with same algorithm where the weight selection strategy is different.

### 4.2  Utility weights
The weights control the relative importance of each objective:

| Weight | Objective | Effect when increased |
|---|---|---|
| **α** | Information Gain | Favors frontiers revealing more unknown space |
| **β** | Path Cost | Favors closer / cheaper-to-reach frontiers |
| **γ** | Redundancy | Penalizes regions likely to be redundantly sensed |
| **δ** | Cluster Size | Favors larger frontier regions |


In the classical baseline, these weights remain fixed. In the proposed RL
approach, PPO adapts them dynamically according to the current exploration
state.

---
## 5. Reinforcement Learning

PPO is used to learn the utility weights from the current exploration state.

At timestep $t$:

$$
\mathbf{w}_t = [\alpha_t,\beta_t,\gamma_t,\delta_t] = \pi_\theta(s_t)
$$

The action is passed directly to the utility module, which constructs the
UAV-to-frontier-cluster utility matrix. The existing Hungarian assignment and
BFS/A* planner then execute the resulting assignments.

### Architecture
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

### PPO Components

| Component | Implementation |
|---|---|
| **Policy** | MLP policy trained with PPO |
| **Observation** | Exploration, UAV, and frontier state features |
| **Action** | Continuous $[\alpha,\beta,\gamma,\delta]$ |
| **Utility** | $\alpha IG-\beta C-\gamma R+\delta S$ |
| **Assignment** | Hungarian |
| **Planning** | BFS / A* |
| **Training** | Seeded map environments |
| **Evaluation** | Unseen map seeds |

### Training Objective

The learned policy is evaluated against the fixed-weight baseline using:

- Steps to 90% coverage
- Total travel distance
- Sensing redundancy
- Success rate
- Cumulative reward

The policy is trained using cumulative exploration reward, with checkpoints
saved at different training horizons(25k/50k/75k/100k steps), to study learning and generalization.


### PPO Training
During training, the PPO policy progressively changes the utility weights in
response to the exploration state. The visualization below shows the
evolution of $\alpha$, $\beta$, $\gamma$, and $\delta$ during an episode.

<img width="1000" height="600" alt="ppo_weight_learning" src="https://github.com/user-attachments/assets/f40136e9-e432-4e7c-b6bd-c6e130d3d028" />

*The above gif is a small representation till 10k, but we have trained till 100k.(more about this in experiment section)


### PPO Learning and Adaptive Utility Weights

The PPO policy learns state-dependent utility weights instead of a single fixed
exploration strategy , dynamically changing
the relative importance of information gain, path cost, redundancy, and
frontier size during exploration.

<table>
  <tr>
    <td align="center" width="45%">
      <img src="https://github.com/user-attachments/assets/235ed236-ad39-4d38-a3d8-7b602b1e9d97" height="400"/>
      <br>
      <b>Dynamic Utility Weights</b>
    </td>
    <td align="center" width="55%">
      <img src="https://github.com/user-attachments/assets/4033f151-df87-489b-be5f-1b7889349054" height="400"/>
      <br>
      <b>Weight Evolution During PPO's Inference <br>
        in exploration of unseen maps </b>
    </td>
  </tr>
</table>


---

## 6. Evaluation

Each episode corresponds to one generated map. Seeded environments allow
different exploration strategies to be evaluated on identical maps under
controlled conditions.

PPO models are trained on a designated set of maps and evaluated separately
on **unseen random seeds**. This prevents the evaluation from simply measuring
performance on environments encountered during training.

The primary evaluation objective is to reach the coverage target efficiently,
rather than maximizing coverage alone. Performance is therefore evaluated
using:

- **Steps to 90% coverage** — lower is better.
- **Total travel distance** — lower is better.
- **Sensing redundancy** — lower is better.
- **Success rate** — percentage of episodes reaching 90% coverage.
- **Final coverage** — coverage achieved at termination.
- **Cumulative reward** — used to evaluate the RL objective and learning
  behavior.

The primary efficiency metric is **steps to 90% coverage**, while distance and
redundancy provide complementary measures of movement efficiency and
multi-UAV coordination.

**All final results will be reported over multiple random seeds using mean and standard deviation.**

---

## 7. Experimental Setup

Experiments are organized into two stages:

**Stage 1:** Understand which classical coordination configuration works best. <br>
**Stage 2:** Ask whether PPO's adaptive weighting can outperform that classical reference.


<img width="1536" height="1024" alt="evaluation_muse" src="https://github.com/user-attachments/assets/e222d703-f76c-45da-b63d-3a4798084928" />


### Experimental Stages

The evaluation is organized into four stages:

**Stage 1 — Classical Evaluation**

Classical coordination strategies are evaluated to identify a strong
non-learning reference. The comparison includes random assignment, greedy
frontier assignment, frontier clustering, Hungarian assignment with fixed
utility weights, and BFS/A* path planning.

**Stage 2 — PPO Checkpoint Evaluation**

PPO policies are evaluated at multiple training checkpoints(both PPO-A and PPO-B):

```text
25k → 50k → 75k → 100k
```

**Stage 3 — Training Diversity**

The initial PPO configuration (**PPO-A**) was trained using a smaller set of
maps. As training progressed, later episodes repeatedly encountered previously
seen environments.

To investigate the effect of training-map diversity, a second configuration
(**PPO-B**) was trained using a larger set of environment seeds.

| Configuration | Training seeds | Purpose |
|---|---:|---|
| **PPO-A** | 1–50 | Initial training with limited map diversity |
| **PPO-B** | 1–100 | Training with increased map diversity |

This experiment examines whether exposing the policy to a more diverse set of
environments improves the generalization of the learned utility weights.


**Stage 4 — Unseen-Map Evaluation**

The trained policies and selected classical configurations are evaluated on
map seeds that were not used during training.

All methods are evaluated under identical environment conditions and on the
same set of maps to ensure a controlled comparison.

The evaluation focuses on:

- **Steps to 90% coverage**
- **Total travel distance**
- **Sensing redundancy**
- **Cumulative reward**
- **Success rate**

This evaluation measures whether the learned weighting policy generalizes
beyond the environments encountered during training.

---

### Training Checkpoints

PPO models are saved at intermediate training checkpoints:

```text
25k → 50k → 75k → 100k
```

---

## 8. Results

### 8.1 Classical Evaluation

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
  title = {Project-Muse: Multi-UAV Swarm Exploration},
  year = {2026},
  url = {https://github.com/sudhansu3299/Project-Muse}
}
```

---

# License

Distributed under the MIT License. See `LICENSE` for more information.

---

# Contact
E-Mail: sudhansu3299@gmail.com
Project Link: [https://github.com/sudhansu3299/Project-Muse](https://github.com/sudhansu3299/Project-Muse)
