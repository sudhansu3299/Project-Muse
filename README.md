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

### Learned Utility Weights across timesteps

<img width="2400" height="1500" alt="06_learned_weights" src="https://github.com/user-attachments/assets/f1eeb01e-05f7-435c-86b2-e4fc9c938cb6" />


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

### PPO Training Configuration

| Parameter | Value |
|---|---:|
| PPO implementation | Stable-Baselines3 |
| Policy | MLP |
| Learning rate | $3 \times 10^{-4}$ |
| Discount factor $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Clip range | 0.2 |
| Rollout length | 2048 |
| Batch size | 64 |
| Training steps | 100,000 |

---

## 8. Results

### 8.1 Classical Evaluation

| Strategy | Time to 90% ↓ | Distance ↓ | Sensing Redundancy ↓ | Visit Overlap ↓ | Movement Efficiency ↑ |
|---|---:|---:|---:|---:|---:|
greedy frontier| 580 ± 36 | 2898 ± 179 | 89.0% ± 0.7% | 8.5% ± 2.5% |0.876 ± 0.027
cluster frontier | 684 ± 41 |3420 ± 206 | 90.7% ± 0.6% | 8.4% ± 2.3% | 0.834 ± 0.022
cluster + utility | 669 ± 42 | 3344 ± 210 | 90.5% ± 0.6% | 6.9% ± 2.6% | 0.847 ± 0.025
hungarian + utility + bfs | 652 ± 40 | 3260 ± 202 | 90.3% ± 0.6% | 11.2% ± 2.0% | 0.794 ± 0.017
hungarian + utility + A* | 674 ± 64 | 3370 ± 320 | 90.5% ± 0.8% | 11.9% ± 3.2% | 0.783 ± 0.029


### Coverage vs. Steps

<img width="4164" height="2063" alt="t90_per_seed_classical" src="https://github.com/user-attachments/assets/933dc8d7-a904-4070-b3c1-5dd3cdda0160" />

<img width="3564" height="1762" alt="t90_boxplot_classical" src="https://github.com/user-attachments/assets/4880a8d5-3554-4842-ae0b-9425d67f5799" />


### Sensing Redundancy

<img width="4164" height="2063" alt="sensing_redundancy_per_seed_classical" src="https://github.com/user-attachments/assets/5b02e970-09fb-41a1-8b84-c927f3aac629" />


### Movement Efficiency

<img width="4164" height="2063" alt="movement_efficiency_per_seed_classical" src="https://github.com/user-attachments/assets/3da2d15a-4966-45b4-a352-68f825c34310" />

### 8.2 PPO Evaluation

**PPO-A:** policy trained with limited/repeated environment exposure (51 maps shown twice)
**PPO-B:** policy trained with increased environment diversity (101 unseen maps)

### T90 vs Training Timesteps
<img width="2400" height="1500" alt="01_steps_to_90" src="https://github.com/user-attachments/assets/53839c5f-bdd5-4657-ba91-d4fed7ad4b98" />

So, we chose 2 of the best checkpoints: 50k_a and 100k_b (where 50k and 100k are the timesteps during training)

### Coverage vs. Steps

<img width="3564" height="1764" alt="ppo_t90_per_seed" src="https://github.com/user-attachments/assets/ac4f0feb-0daa-4844-824d-92f65987092c" />

### Sensing Redundancy
<img width="3564" height="1764" alt="ppo_sensing_redundancy_per_seed" src="https://github.com/user-attachments/assets/e4a3f4bd-599f-4e8d-8cdb-ce93b1074b02" />


### Movement Efficiency
<img width="3564" height="1764" alt="ppo_movement_efficiency_per_seed" src="https://github.com/user-attachments/assets/8b99043e-5c6a-466c-beba-85422d163524" />

### 8.3 PPO vs classical baselines

#### TBD


---

# 9. Ablation Studies

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

# 10. Limitations

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

# 11. Future Work

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

# 12. Project Status

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

# 13. Citation

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
