import re
import matplotlib.pyplot as plt
import numpy as np

def parse_file_txt(filepath):
    """Parse the file.txt to extract timestep data for each variant and seed."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by variant sections
    variants = {}
    current_variant = None
    
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        
        # Check for variant header (handles both "Variant X:" and "variant X:")
        variant_match = re.match(r'[Vv]ariant ([A-D]):', line)
        if variant_match:
            current_variant = variant_match.group(1).upper()
            variants[current_variant] = {}
            continue
        
        # Check for CSV format (Variant A style)
        csv_match = re.match(r'[^,]+,(\d+),(\d+)', line)
        if csv_match and current_variant:
            seed = int(csv_match.group(1))
            timestep = int(csv_match.group(2))
            variants[current_variant][seed] = timestep
            continue
        
        # Check for seed header (Variants B, C, D style)
        seed_match = re.match(r'===== SEED (\d+) =====', line)
        if seed_match and current_variant:
            current_seed = int(seed_match.group(1))
            continue
        
        # Check for timestep data (Variants B, C, D style)
        step_match = re.search(r'Reached 90% at step (\d+)', line)
        if step_match and current_variant:
            timestep = int(step_match.group(1))
            variants[current_variant][current_seed] = timestep
    
    return variants

def plot_variant_comparison(variants):
    """Create a plot comparing variants across seeds."""
    # Get all unique seeds
    all_seeds = set()
    for variant_data in variants.values():
        all_seeds.update(variant_data.keys())
    all_seeds = sorted(all_seeds)
    
    # Prepare data for plotting
    variant_names = sorted(variants.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    plt.figure(figsize=(12, 7))
    
    for i, variant in enumerate(variant_names):
        if variant not in variants:
            continue
        
        timesteps = []
        for seed in all_seeds:
            if seed in variants[variant]:
                timesteps.append(variants[variant][seed])
            else:
                timesteps.append(None)
        
        # Filter out None values for plotting
        valid_seeds = [seed for seed, t in zip(all_seeds, timesteps) if t is not None]
        valid_timesteps = [t for t in timesteps if t is not None]
        
        if valid_timesteps:
            plt.plot(valid_seeds, valid_timesteps, 
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    linewidth=2,
                    markersize=8,
                    label=f'Variant {variant}')
    
    plt.xlabel('Seed', fontsize=14, fontweight='bold')
    plt.ylabel('Timestep to reach 90% coverage', fontsize=14, fontweight='bold')
    plt.title('Comparison of Variants Across Seeds', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_path = '/Users/sudhansu/personal/Project-Muse/plots/variant_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()

def calculate_statistics(variants):
    """Calculate mean and std for each variant."""
    stats = {}
    for variant, data in variants.items():
        timesteps = list(data.values())
        if timesteps:
            mean = np.mean(timesteps)
            std = np.std(timesteps)
            stats[variant] = {
                'mean': mean,
                'std': std,
                'count': len(timesteps),
                'min': min(timesteps),
                'max': max(timesteps)
            }
    return stats

def print_statistics(stats):
    """Print statistics in a formatted table."""
    print("\n" + "="*70)
    print("STATISTICS: Mean ± Standard Deviation for Each Variant")
    print("="*70)
    print(f"{'Variant':<12} {'Mean':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10} {'Count':<8}")
    print("-"*70)
    
    for variant in sorted(stats.keys()):
        s = stats[variant]
        print(f"Variant {variant:<6} {s['mean']:<12.2f} {s['std']:<12.2f} {s['min']:<10} {s['max']:<10} {s['count']:<8}")
    
    print("="*70)

def calculate_regret_count(variants):
    """Calculate regret count: how many seeds each variant lost against the best-performing method."""
    # Get all unique seeds
    all_seeds = set()
    for variant_data in variants.values():
        all_seeds.update(variant_data.keys())
    all_seeds = sorted(all_seeds)
    
    # For each seed, find the best performer (lowest timestep)
    regret_counts = {variant: 0 for variant in variants.keys()}
    best_per_seed = {}
    
    for seed in all_seeds:
        best_variant = None
        best_timestep = float('inf')
        
        for variant in variants:
            if seed in variants[variant]:
                timestep = variants[variant][seed]
                if timestep < best_timestep:
                    best_timestep = timestep
                    best_variant = variant
        
        best_per_seed[seed] = (best_variant, best_timestep)
        
        # Count regret for all variants that are not the best
        for variant in variants:
            if seed in variants[variant] and variant != best_variant:
                regret_counts[variant] += 1
    
    return regret_counts, best_per_seed

def print_regret_analysis(regret_counts, best_per_seed, variants):
    """Print regret analysis."""
    print("\n" + "="*70)
    print("REGRET ANALYSIS: Seeds lost against best-performing method")
    print("="*70)
    print(f"{'Variant':<12} {'Regret Count':<15} {'Total Seeds':<12} {'Regret %':<10}")
    print("-"*70)
    
    total_seeds = len(best_per_seed)
    for variant in sorted(regret_counts.keys()):
        regret = regret_counts[variant]
        variant_seeds = len(variants[variant])
        regret_pct = (regret / total_seeds) * 100 if total_seeds > 0 else 0
        print(f"Variant {variant:<6} {regret:<15} {variant_seeds:<12} {regret_pct:<10.1f}%")
    
    print("="*70)
    
    print("\nBest performer per seed:")
    print("-"*70)
    for seed in sorted(best_per_seed.keys()):
        best_variant, best_timestep = best_per_seed[seed]
        print(f"Seed {seed}: Variant {best_variant} (timestep: {best_timestep})")
    print("="*70)
    
    # Specific C vs D comparison
    print("\n" + "="*70)
    print("C vs D HEAD-TO-HEAD COMPARISON")
    print("="*70)
    c_wins = 0
    d_wins = 0
    ties = 0
    
    for seed in variants['C']:
        if seed in variants['D']:
            c_time = variants['C'][seed]
            d_time = variants['D'][seed]
            if c_time < d_time:
                c_wins += 1
            elif d_time < c_time:
                d_wins += 1
            else:
                ties += 1
    
    print(f"Variant C wins: {c_wins}")
    print(f"Variant D wins: {d_wins}")
    print(f"Ties: {ties}")
    print(f"Total compared: {c_wins + d_wins + ties}")
    print("="*70)

if __name__ == '__main__':
    filepath = '/Users/sudhansu/personal/Project-Muse/results/classical_benchmark/file.txt'
    variants = parse_file_txt(filepath)
    
    print("Parsed variants:")
    for variant, data in variants.items():
        print(f"  Variant {variant}: {len(data)} seeds")
        print(f"    Seeds: {sorted(data.keys())}")
        print(f"    Timesteps: {data}")
    
    # Calculate and print statistics
    stats = calculate_statistics(variants)
    print_statistics(stats)
    
    # Calculate and print regret analysis
    regret_counts, best_per_seed = calculate_regret_count(variants)
    print_regret_analysis(regret_counts, best_per_seed, variants)
    
    plot_variant_comparison(variants)
