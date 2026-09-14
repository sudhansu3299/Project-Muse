import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# DATA LOADING
# ============================================================

RAW_RESULTS_PATH = "results/classical_benchmark/raw_results.csv"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(RAW_RESULTS_PATH)

# Filter out random strategy (no data) and keep only classical strategies
df = df[df['strategy'] != 'random']
df = df.dropna(subset=['time_to_90'])

# Get unique strategies and seeds
strategies = df['strategy'].unique()
seeds = df['seed'].unique()

print(f"Strategies: {strategies}")
print(f"Seeds: {seeds}")


# ============================================================
# T90 PLOT - BAR CHART WITH ERROR BARS
# ============================================================

plt.figure(figsize=(12, 6))

# Calculate mean and std for each strategy
strategy_stats = df.groupby('strategy')['time_to_90'].agg(['mean', 'std']).reset_index()
strategy_stats = strategy_stats.sort_values('mean')

# Create bar positions
x_pos = np.arange(len(strategy_stats))
bar_width = 0.6

# Plot bars with error bars
bars = plt.bar(
    x_pos,
    strategy_stats['mean'],
    yerr=strategy_stats['std'],
    capsize=5,
    alpha=0.7,
    color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'],
    edgecolor='black',
    linewidth=1.2,
    width=bar_width
)

# Add value labels on top of bars
for i, (idx, row) in enumerate(strategy_stats.iterrows()):
    plt.text(
        x_pos[i],
        row['mean'] + row['std'] + 10,
        f"{row['mean']:.0f}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.xlabel('Strategy', fontsize=12, fontweight='bold')
plt.ylabel('Time to 90% Coverage (steps)', fontsize=12, fontweight='bold')
plt.title('T90 Across All Unseen Seeds - Classical Strategies Comparison', 
          fontsize=14, fontweight='bold', pad=20)

plt.xticks(x_pos, strategy_stats['strategy'], rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/t90_classical_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"\nGenerated: {OUTPUT_DIR}/t90_classical_comparison.png")


# ============================================================
# T90 PLOT - INDIVIDUAL SEEDS (SCATTER/PLOT)
# ============================================================

plt.figure(figsize=(14, 7))

# Plot each strategy's T90 across seeds
colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
markers = ['o', 's', '^', 'D', 'v']

for i, strategy in enumerate(strategies):
    strategy_data = df[df['strategy'] == strategy]
    plt.plot(
        strategy_data['seed'],
        strategy_data['time_to_90'],
        marker=markers[i % len(markers)],
        linestyle='-',
        linewidth=2,
        markersize=8,
        color=colors[i % len(colors)],
        label=strategy,
        alpha=0.8
    )

plt.xlabel('Seed', fontsize=12, fontweight='bold')
plt.ylabel('Time to 90% Coverage (steps)', fontsize=12, fontweight='bold')
plt.title('T90 Per Seed - Classical Strategies Comparison', 
          fontsize=14, fontweight='bold', pad=20)

plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/t90_per_seed_classical.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/t90_per_seed_classical.png")


# ============================================================
# T90 PLOT - BOX PLOT
# ============================================================

plt.figure(figsize=(12, 6))

# Prepare data for box plot
data_to_plot = [df[df['strategy'] == strategy]['time_to_90'].values for strategy in strategies]

# Create box plot
bp = plt.boxplot(
    data_to_plot,
    labels=strategies,
    patch_artist=True,
    notch=True,
    showmeans=True
)

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Customize median line
for median in bp['medians']:
    median.set(color='black', linewidth=2)

# Customize mean marker
for mean in bp['means']:
    mean.set(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=8)

plt.xlabel('Strategy', fontsize=12, fontweight='bold')
plt.ylabel('Time to 90% Coverage (steps)', fontsize=12, fontweight='bold')
plt.title('T90 Distribution Across Seeds - Classical Strategies', 
          fontsize=14, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/t90_boxplot_classical.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/t90_boxplot_classical.png")


# ============================================================
# SENSING REDUNDANCY PLOT - BAR CHART WITH ERROR BARS
# ============================================================

plt.figure(figsize=(12, 6))

# Calculate mean and std for each strategy
redundancy_stats = df.groupby('strategy')['sensing_redundancy'].agg(['mean', 'std']).reset_index()
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
    color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'],
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
        fontsize=10,
        fontweight='bold'
    )

plt.xlabel('Strategy', fontsize=12, fontweight='bold')
plt.ylabel('Sensing Redundancy (%)', fontsize=12, fontweight='bold')
plt.title('Sensing Redundancy Across All Unseen Seeds - Classical Strategies Comparison', 
          fontsize=14, fontweight='bold', pad=20)

plt.xticks(x_pos, redundancy_stats['strategy'], rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/sensing_redundancy_classical.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/sensing_redundancy_classical.png")


# ============================================================
# MOVEMENT EFFICIENCY PLOT - BAR CHART WITH ERROR BARS
# ============================================================

plt.figure(figsize=(12, 6))

# Calculate mean and std for each strategy
efficiency_stats = df.groupby('strategy')['movement_efficiency'].agg(['mean', 'std']).reset_index()
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
    color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12'],
    edgecolor='black',
    linewidth=1.2,
    width=bar_width
)

# Add value labels on top of bars
for i, (idx, row) in enumerate(efficiency_stats.iterrows()):
    plt.text(
        x_pos[i],
        row['mean'] + row['std'] + 0.01,
        f"{row['mean']:.3f}",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

plt.xlabel('Strategy', fontsize=12, fontweight='bold')
plt.ylabel('Movement Efficiency', fontsize=12, fontweight='bold')
plt.title('Movement Efficiency Across All Unseen Seeds - Classical Strategies Comparison', 
          fontsize=14, fontweight='bold', pad=20)

plt.xticks(x_pos, efficiency_stats['strategy'], rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/movement_efficiency_classical.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/movement_efficiency_classical.png")


# ============================================================
# SENSING REDUNDANCY PLOT - PER SEED
# ============================================================

plt.figure(figsize=(14, 7))

# Plot each strategy's sensing redundancy across seeds
for i, strategy in enumerate(strategies):
    strategy_data = df[df['strategy'] == strategy]
    plt.plot(
        strategy_data['seed'],
        strategy_data['sensing_redundancy'],
        marker=markers[i % len(markers)],
        linestyle='-',
        linewidth=2,
        markersize=8,
        color=colors[i % len(colors)],
        label=strategy,
        alpha=0.8
    )

plt.xlabel('Seed', fontsize=12, fontweight='bold')
plt.ylabel('Sensing Redundancy (%)', fontsize=12, fontweight='bold')
plt.title('Sensing Redundancy Per Seed - Classical Strategies Comparison', 
          fontsize=14, fontweight='bold', pad=20)

plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/sensing_redundancy_per_seed_classical.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/sensing_redundancy_per_seed_classical.png")


# ============================================================
# MOVEMENT EFFICIENCY PLOT - PER SEED
# ============================================================

plt.figure(figsize=(14, 7))

# Plot each strategy's movement efficiency across seeds
for i, strategy in enumerate(strategies):
    strategy_data = df[df['strategy'] == strategy]
    plt.plot(
        strategy_data['seed'],
        strategy_data['movement_efficiency'],
        marker=markers[i % len(markers)],
        linestyle='-',
        linewidth=2,
        markersize=8,
        color=colors[i % len(colors)],
        label=strategy,
        alpha=0.8
    )

plt.xlabel('Seed', fontsize=12, fontweight='bold')
plt.ylabel('Movement Efficiency', fontsize=12, fontweight='bold')
plt.title('Movement Efficiency Per Seed - Classical Strategies Comparison', 
          fontsize=14, fontweight='bold', pad=20)

plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/movement_efficiency_per_seed_classical.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/movement_efficiency_per_seed_classical.png")


# ============================================================
# SENSING REDUNDANCY PLOT - BOX PLOT
# ============================================================

plt.figure(figsize=(12, 6))

# Prepare data for box plot
redundancy_data_to_plot = [df[df['strategy'] == strategy]['sensing_redundancy'].values for strategy in strategies]

# Create box plot
bp = plt.boxplot(
    redundancy_data_to_plot,
    labels=strategies,
    patch_artist=True,
    notch=True,
    showmeans=True
)

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Customize median line
for median in bp['medians']:
    median.set(color='black', linewidth=2)

# Customize mean marker
for mean in bp['means']:
    mean.set(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=8)

plt.xlabel('Strategy', fontsize=12, fontweight='bold')
plt.ylabel('Sensing Redundancy (%)', fontsize=12, fontweight='bold')
plt.title('Sensing Redundancy Distribution Across Seeds - Classical Strategies', 
          fontsize=14, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/sensing_redundancy_boxplot_classical.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/sensing_redundancy_boxplot_classical.png")


# ============================================================
# MOVEMENT EFFICIENCY PLOT - BOX PLOT
# ============================================================

plt.figure(figsize=(12, 6))

# Prepare data for box plot
efficiency_data_to_plot = [df[df['strategy'] == strategy]['movement_efficiency'].values for strategy in strategies]

# Create box plot
bp = plt.boxplot(
    efficiency_data_to_plot,
    labels=strategies,
    patch_artist=True,
    notch=True,
    showmeans=True
)

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

# Customize median line
for median in bp['medians']:
    median.set(color='black', linewidth=2)

# Customize mean marker
for mean in bp['means']:
    mean.set(marker='D', markeredgecolor='black', markerfacecolor='white', markersize=8)

plt.xlabel('Strategy', fontsize=12, fontweight='bold')
plt.ylabel('Movement Efficiency', fontsize=12, fontweight='bold')
plt.title('Movement Efficiency Distribution Across Seeds - Classical Strategies', 
          fontsize=14, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/movement_efficiency_boxplot_classical.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Generated: {OUTPUT_DIR}/movement_efficiency_boxplot_classical.png")


print("\n" + "="*60)
print("All plots generated successfully!")
print("="*60)
