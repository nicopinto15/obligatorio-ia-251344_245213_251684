import numpy as np
import matplotlib.pyplot as plt
from tactix_env import TacTixEnv
from random_agent import RandomTacTixAgent
from trainer_agent import TrainerAgent
from minimax_agent import MinimaxAgent
from expectimax_agent import ExpectimaxAgent
from play import run_multiple_games, plot_results
import time

def run_tournament():
    """Run a comprehensive tournament between all agents"""
    print("Starting TacTix Tournament...")
    
    # Initialize environment (normal rules)
    env_normal = TacTixEnv(board_size=6)
    
    # Initialize agents
    agents = {
        'Random': RandomTacTixAgent(env_normal),
        'Trainer (Easy)': TrainerAgent(env_normal, difficulty=0.1),
        'Trainer (Medium)': TrainerAgent(env_normal, difficulty=0.5),
        'Trainer (Hard)': TrainerAgent(env_normal, difficulty=0.9),
        'Minimax (Depth 3)': MinimaxAgent(env_normal, depth=3),
        'Minimax (Depth 4)': MinimaxAgent(env_normal, depth=4),
        'Expectimax (Depth 3)': ExpectimaxAgent(env_normal, depth=3),
        'Expectimax (Depth 4)': ExpectimaxAgent(env_normal, depth=4),
    }
    
    # Tournament results
    results = {}
    
    # Normal rules tournament
    print("\n=== NORMAL RULES TOURNAMENT ===")
    results['normal'] = run_rule_tournament(env_normal, agents, "Normal")
    
    # Performance analysis
    analyze_performance(results)
    
    return results

def run_rule_tournament(env, agents, rule_name):
    """Run tournament for specific rules"""
    agent_names = list(agents.keys())
    num_agents = len(agent_names)
    games_per_match = 20
    
    # Results matrix
    win_matrix = np.zeros((num_agents, num_agents))
    time_matrix = np.zeros((num_agents, num_agents))
    
    print(f"\nRunning {rule_name} rules tournament...")
    
    for i, agent1_name in enumerate(agent_names):
        for j, agent2_name in enumerate(agent_names):
            if i != j:
                print(f"  {agent1_name} vs {agent2_name}...")
                
                agent1 = agents[agent1_name]
                agent2 = agents[agent2_name]
                
                start_time = time.time()
                match_results = run_multiple_games(env, agent1, agent2, games_per_match)
                end_time = time.time()
                
                win_matrix[i, j] = match_results['agent1_wins']
                time_matrix[i, j] = end_time - start_time
                
                print(f"    {agent1_name}: {match_results['agent1_wins']}, {agent2_name}: {match_results['agent2_wins']}")
    
    # Calculate win rates
    win_rates = np.sum(win_matrix, axis=1) / (games_per_match * (num_agents - 1))
    
    # Print results
    print(f"\n{rule_name} Rules - Final Win Rates:")
    for i, agent_name in enumerate(agent_names):
        print(f"  {agent_name}: {win_rates[i]:.3f}")
    
    return {
        'agent_names': agent_names,
        'win_matrix': win_matrix,
        'win_rates': win_rates,
        'time_matrix': time_matrix
    }

def analyze_performance(results):
    """Analyze and visualize tournament results"""
    print("\n=== PERFORMANCE ANALYSIS ===")
    
    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Normal rules win rates
    normal_results = results['normal']
    axes[0].bar(range(len(normal_results['agent_names'])), normal_results['win_rates'])
    axes[0].set_title('Normal Rules - Win Rates')
    axes[0].set_ylabel('Win Rate')
    axes[0].set_xticks(range(len(normal_results['agent_names'])))
    axes[0].set_xticklabels(normal_results['agent_names'], rotation=45, ha='right')
    
    # Average game time
    avg_times_normal = np.mean(normal_results['time_matrix'], axis=1)
    axes[1].bar(range(len(normal_results['agent_names'])), avg_times_normal)
    axes[1].set_title('Average Game Time')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_xticks(range(len(normal_results['agent_names'])))
    axes[1].set_xticklabels(normal_results['agent_names'], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('tournament_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis
    print("\nStatistical Analysis:")
    print("Normal Rules:")
    best_normal = np.argmax(normal_results['win_rates'])
    print(f"  Best agent: {normal_results['agent_names'][best_normal]} ({normal_results['win_rates'][best_normal]:.3f})")
    
    # Algorithm comparison
    minimax_agents = [i for i, name in enumerate(normal_results['agent_names']) if 'Minimax' in name]
    expectimax_agents = [i for i, name in enumerate(normal_results['agent_names']) if 'Expectimax' in name]
    
    if minimax_agents and expectimax_agents:
        minimax_avg = np.mean([normal_results['win_rates'][i] for i in minimax_agents])
        expectimax_avg = np.mean([normal_results['win_rates'][i] for i in expectimax_agents])
        
        print(f"\nAlgorithm Comparison (Normal Rules):")
        print(f"  Minimax average: {minimax_avg:.3f}")
        print(f"  Expectimax average: {expectimax_avg:.3f}")
        
        if minimax_avg > expectimax_avg:
            print("  Minimax performs better on average")
        else:
            print("  Expectimax performs better on average")

def depth_analysis():
    """Analyze the effect of search depth on performance"""
    print("\n=== DEPTH ANALYSIS ===")
    
    env = TacTixEnv(board_size=6)
    depths = [2, 3, 4, 5]
    games_per_depth = 20
    
    minimax_results = []
    expectimax_results = []
    
    for depth in depths:
        print(f"Testing depth {depth}...")
        
        # Minimax vs Random
        minimax_agent = MinimaxAgent(env, depth=depth)
        random_agent = RandomTacTixAgent(env)
        
        results = run_multiple_games(env, minimax_agent, random_agent, games_per_depth)
        minimax_win_rate = results['agent1_wins'] / games_per_depth
        minimax_results.append(minimax_win_rate)
        
        # Expectimax vs Random
        expectimax_agent = ExpectimaxAgent(env, depth=depth)
        results = run_multiple_games(env, expectimax_agent, random_agent, games_per_depth)
        expectimax_win_rate = results['agent1_wins'] / games_per_depth
        expectimax_results.append(expectimax_win_rate)
        
        print(f"  Minimax (depth {depth}): {minimax_win_rate:.3f}")
        print(f"  Expectimax (depth {depth}): {expectimax_win_rate:.3f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(depths, minimax_results, 'o-', label='Minimax', linewidth=2, markersize=8)
    plt.plot(depths, expectimax_results, 's-', label='Expectimax', linewidth=2, markersize=8)
    plt.xlabel('Search Depth')
    plt.ylabel('Win Rate vs Random')
    plt.title('Performance vs Search Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('depth_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Run full tournament
    tournament_results = run_tournament()
    
    # Run depth analysis
    depth_analysis()
    
    print("\nTournament completed! Results saved to tournament_results.png and depth_analysis.png")
