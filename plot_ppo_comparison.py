import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# DATA LOADING
# ============================================================

PPO_DATA_PATH = "results/ppo/evaluation/ppo_all_checkpoint_trajectories.csv"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(PPO_DATA_PATH)

# Filter for PPO-A 50k and PPO-B 100k
df_filtered = df[df['checkpoint'].isin(['ppo_a_50k', 'ppo_b_100k'])].copy()

# Rename checkpoints for better labels
df_filtered['model'] = df_filtered['checkpoint'].replace({
    'ppo_a_50k': 'PPO-A (50k)',
    'ppo_b_100k': 'PPO-B (100k)'
})

print(f"Data loaded: {len(df_filtered)} rows")
print(f"Models: {df_filtered['model'].unique()}")
print(f"Seeds: {df_filtered['seed'].unique()}")


# ============================================================
# T90 PLOT - BAR CHART WITH ERROR BARS
# ============================================================

plt.figure(figsize=(10, 6))

# Calculate mean and std for each model
t90_stats = df_filtered.groupby('model')['steps'].agg(['mean', 'std']).reset_index()
t90_stats = t90_stats.sort_values('mean')

# Create bar positions
x_pos = np.arange(len(t90_stats))
bar_width = 0.6

# Plot bars with error bars
bars = plt.bar(
    x_pos,
    t90_stats['mean'],
    yerr=t90_stats['std'],
    capsize=5,
    alpha=0.7,
    color=['#3498db', '#e74c3c'],
    edgecolor='black',
    linewidth=1.2,
    width=bar_width
)

# Add value labels on top of bars
for i, (idx, row) in enumerate(t90_stats.iterrows()):
    plt.text(
        x_pos[i],
        row['mean'] + row['std'] + 10,
        f"{row['mean']:.0f}",
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold'
    )

plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Time to 90% Coverage (steps)', fontsize=12, fontweight='bold')
plt.title('T90 Comparison: PPO-A (50k) vs PPO-B (100k)', 
          fontsize=14, fontweight='bold', pad=20)

plt.xticks(x_pos, t90_stats['model'])
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/ppo_t90_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/ppo_t90_comparison.png")


# ============================================================
# T90 PLOT - PER SEED
# ============================================================

plt.figure(figsize=(12, 6))

# Plot each model's T90 across seeds
models = df_filtered['model'].unique()
colors = ['#3498db', '#e74c3c']
markers = ['o', 's']

for i, model in enumerate(models):
    model_data = df_filtered[df_filtered['model'] == model].sort_values('seed')
    plt.plot(
        model_data['seed'],
        model_data['steps'],
        marker=markers[i % len(markers)],
        linestyle='-',
        linewidth=2,
        markersize=8,
        color=colors[i % len(colors)],
        label=model,
        alpha=0.8
    )

plt.xlabel('Seed', fontsize=12, fontweight='bold')
plt.ylabel('Time to 90% Coverage (steps)', fontsize=12, fontweight='bold')
plt.title('T90 Per Seed: PPO-A (50k) vs PPO-B (100k)', 
          fontsize=14, fontweight='bold', pad=20)

plt.legend(loc='best', fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/ppo_t90_per_seed.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/ppo_t90_per_seed.png")


# ============================================================
# SENSING REDUNDANCY PLOT - BAR CHART WITH ERROR BARS
# ============================================================

plt.figure(figsize=(10, 6))

# Calculate mean and std for each model
redundancy_stats = df_filtered.groupby('model')['final_redundancy'].agg(['mean', 'std']).reset_index()
redundancy_stats = redundancy_stats.sort_values('mean')

# Create bar positions
x_pos = np.arange(len(redundancy_stats))
bar_width = 0.6

# Plot bars with error bars
bars = plt.bar(
    x_pos,
    redundancy_stats['mean'],
    yerr=redundancy_stats['std'],
    capsize=5,
    alpha=0.7,
    color=['#3498db', '#e74c3c'],
    edgecolor='black',
    linewidth=1.2,
    width=bar_width
)

# Add value labels on top of bars
for i, (idx, row) in enumerate(redundancy_stats.iterrows()):
    plt.text(
        x_pos[i],
        row['mean'] + row['std'] + 0.2,
        f"{row['mean']:.1f}%",
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold'
    )

plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Sensing Redundancy (%)', fontsize=12, fontweight='bold')
plt.title('Sensing Redundancy Comparison: PPO-A (50k) vs PPO-B (100k)', 
          fontsize=14, fontweight='bold', pad=20)

plt.xticks(x_pos, redundancy_stats['model'])
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/ppo_sensing_redundancy_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/ppo_sensing_redundancy_comparison.png")


# ============================================================
# SENSING REDUNDANCY PLOT - PER SEED
# ============================================================

plt.figure(figsize=(12, 6))

# Plot each model's sensing redundancy across seeds
for i, model in enumerate(models):
    model_data = df_filtered[df_filtered['model'] == model].sort_values('seed')
    plt.plot(
        model_data['seed'],
        model_data['final_redundancy'],
        marker=markers[i % len(markers)],
        linestyle='-',
        linewidth=2,
        markersize=8,
        color=colors[i % len(colors)],
        label=model,
        alpha=0.8
    )

plt.xlabel('Seed', fontsize=12, fontweight='bold')
plt.ylabel('Sensing Redundancy (%)', fontsize=12, fontweight='bold')
plt.title('Sensing Redundancy Per Seed: PPO-A (50k) vs PPO-B (100k)', 
          fontsize=14, fontweight='bold', pad=20)

plt.legend(loc='best', fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/ppo_sensing_redundancy_per_seed.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/ppo_sensing_redundancy_per_seed.png")


# ============================================================
# MOVEMENT EFFICIENCY CALCULATION AND PLOT
# ============================================================

# Calculate movement efficiency: coverage / distance
# At 90% coverage, we can use: 90 / total_distance as a proxy
df_filtered['movement_efficiency'] = 90.0 / df_filtered['total_distance']

plt.figure(figsize=(10, 6))

# Calculate mean and std for each model
efficiency_stats = df_filtered.groupby('model')['movement_efficiency'].agg(['mean', 'std']).reset_index()
efficiency_stats = efficiency_stats.sort_values('mean', ascending=False)

# Create bar positions
x_pos = np.arange(len(efficiency_stats))
bar_width = 0.6

# Plot bars with error bars
bars = plt.bar(
    x_pos,
    efficiency_stats['mean'],
    yerr=efficiency_stats['std'],
    capsize=5,
    alpha=0.7,
    color=['#3498db', '#e74c3c'],
    edgecolor='black',
    linewidth=1.2,
    width=bar_width
)

# Add value labels on top of bars
for i, (idx, row) in enumerate(efficiency_stats.iterrows()):
    plt.text(
        x_pos[i],
        row['mean'] + row['std'] + 0.001,
        f"{row['mean']:.4f}",
        ha='center',
        va='bottom',
        fontsize=12,
        fontweight='bold'
    )

plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Movement Efficiency (coverage/distance)', fontsize=12, fontweight='bold')
plt.title('Movement Efficiency Comparison: PPO-A (50k) vs PPO-B (100k)', 
          fontsize=14, fontweight='bold', pad=20)

plt.xticks(x_pos, efficiency_stats['model'])
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/ppo_movement_efficiency_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/ppo_movement_efficiency_comparison.png")


# ============================================================
# MOVEMENT EFFICIENCY PLOT - PER SEED
# ============================================================

plt.figure(figsize=(12, 6))

# Plot each model's movement efficiency across seeds
for i, model in enumerate(models):
    model_data = df_filtered[df_filtered['model'] == model].sort_values('seed')
    plt.plot(
        model_data['seed'],
        model_data['movement_efficiency'],
        marker=markers[i % len(markers)],
        linestyle='-',
        linewidth=2,
        markersize=8,
        color=colors[i % len(colors)],
        label=model,
        alpha=0.8
    )

plt.xlabel('Seed', fontsize=12, fontweight='bold')
plt.ylabel('Movement Efficiency (coverage/distance)', fontsize=12, fontweight='bold')
plt.title('Movement Efficiency Per Seed: PPO-A (50k) vs PPO-B (100k)', 
          fontsize=14, fontweight='bold', pad=20)

plt.legend(loc='best', fontsize=11)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/ppo_movement_efficiency_per_seed.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/ppo_movement_efficiency_per_seed.png")


# ============================================================
# COMBINED PLOT - ALL METRICS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# T90
for i, model in enumerate(models):
    model_data = df_filtered[df_filtered['model'] == model].sort_values('seed')
    axes[0].plot(
        model_data['seed'],
        model_data['steps'],
        marker=markers[i % len(markers)],
        linestyle='-',
        linewidth=2,
        markersize=8,
        color=colors[i % len(colors)],
        label=model,
        alpha=0.8
    )
axes[0].set_xlabel('Seed', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Time to 90% (steps)', fontsize=11, fontweight='bold')
axes[0].set_title('T90', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3, linestyle='--')

# Sensing Redundancy
for i, model in enumerate(models):
    model_data = df_filtered[df_filtered['model'] == model].sort_values('seed')
    axes[1].plot(
        model_data['seed'],
        model_data['final_redundancy'],
        marker=markers[i % len(markers)],
        linestyle='-',
        linewidth=2,
        markersize=8,
        color=colors[i % len(colors)],
        label=model,
        alpha=0.8
    )
axes[1].set_xlabel('Seed', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Sensing Redundancy (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Sensing Redundancy', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3, linestyle='--')

# Movement Efficiency
for i, model in enumerate(models):
    model_data = df_filtered[df_filtered['model'] == model].sort_values('seed')
    axes[2].plot(
        model_data['seed'],
        model_data['movement_efficiency'],
        marker=markers[i % len(markers)],
        linestyle='-',
        linewidth=2,
        markersize=8,
        color=colors[i % len(colors)],
        label=model,
        alpha=0.8
    )
axes[2].set_xlabel('Seed', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Movement Efficiency', fontsize=11, fontweight='bold')
axes[2].set_title('Movement Efficiency', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(alpha=0.3, linestyle='--')

plt.suptitle('PPO-A (50k) vs PPO-B (100k) - Full Comparison', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/ppo_combined_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/ppo_combined_comparison.png")


print("\n" + "="*60)
print("All PPO comparison plots generated successfully!")
print("="*60)
