#!/usr/bin/env python3
"""
Entrenamiento Progresivo Stochastic Q-Learning
18 iteraciones con ajuste gradual de hiperparámetros
5,000 episodios por iteración
Total: 90,000 episodios (~4-5 horas)
7 acciones para mayor granularidad de control
Basado en Fourati et al. (2024)
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from datetime import datetime
import descent_env
from stochastic_q_learning_agent import StochasticQLearningAgent

def create_progressive_config():
    """Crea configuración progresiva de 18 iteraciones"""
    configs = [
        {'episodes': 5000, 'alpha': 0.9, 'gamma': 0.9, 'epsilon': 1.0},
        {'episodes': 5000, 'alpha': 0.9, 'gamma': 0.9, 'epsilon': 0.9},
        {'episodes': 5000, 'alpha': 0.9, 'gamma': 0.95, 'epsilon': 0.9},
        {'episodes': 5000, 'alpha': 0.8, 'gamma': 0.95, 'epsilon': 0.9},
        {'episodes': 5000, 'alpha': 0.6, 'gamma': 0.95, 'epsilon': 0.9},
        {'episodes': 5000, 'alpha': 0.6, 'gamma': 0.95, 'epsilon': 0.7},
        {'episodes': 5000, 'alpha': 0.6, 'gamma': 0.98, 'epsilon': 0.7},
        {'episodes': 5000, 'alpha': 0.5, 'gamma': 0.98, 'epsilon': 0.7},
        {'episodes': 5000, 'alpha': 0.5, 'gamma': 0.98, 'epsilon': 0.5},
        {'episodes': 5000, 'alpha': 0.5, 'gamma': 0.98, 'epsilon': 0.3},
        {'episodes': 5000, 'alpha': 0.3, 'gamma': 0.98, 'epsilon': 0.3},
        {'episodes': 5000, 'alpha': 0.3, 'gamma': 0.98, 'epsilon': 0.1},
        {'episodes': 5000, 'alpha': 0.3, 'gamma': 0.99, 'epsilon': 0.1},
        {'episodes': 5000, 'alpha': 0.2, 'gamma': 0.99, 'epsilon': 0.1},
        {'episodes': 5000, 'alpha': 0.1, 'gamma': 0.99, 'epsilon': 0.01},
        {'episodes': 5000, 'alpha': 0.1, 'gamma': 0.995, 'epsilon': 0.01},
        {'episodes': 5000, 'alpha': 0.05, 'gamma': 0.995, 'epsilon': 0.01},
        {'episodes': 5000, 'alpha': 0.05, 'gamma': 0.999, 'epsilon': 0.001},
    ]
    total_episodes = sum(config['episodes'] for config in configs)
    print(f"📊 Total de episodios configurados: {total_episodes:,}")
    return configs

def train_progressive_stochastic_qlearning():
    print("🎯 ENTRENAMIENTO PROGRESIVO STOCHASTIC Q-LEARNING")
    print("="*60)
    print("📋 Configuración:")
    print("   • 18 iteraciones con hiperparámetros ajustables")
    print("   • 5,000 episodios por iteración")
    print("   • Total: 90,000 episodios")
    print("   • Estrategia: Exploración → Refinamiento → Explotación")
    print("")
    configs = create_progressive_config()
    env = descent_env.DescentEnv()
    n_actions = 7
    state_bins = [6, 6, 8, 8]
    agent = StochasticQLearningAgent(
        n_actions=n_actions,
        state_bins=state_bins,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0
    )
    all_episode_rewards = []
    iteration_summaries = []
    q_table_evolution = []
    total_episodes_trained = 0
    start_time = datetime.now()
    print("🚀 Iniciando entrenamiento progresivo...")
    print("")
    for iteration, config in enumerate(configs, 1):
        print(f"🔄 ITERACIÓN {iteration}/18")
        print(f"   Episodes: {config['episodes']:,}")
        print(f"   Alpha: {config['alpha']}")
        print(f"   Gamma: {config['gamma']}")
        print(f"   Epsilon: {config['epsilon']}")
        agent.alpha = config['alpha']
        agent.gamma = config['gamma']
        agent.epsilon = config['epsilon']
        iteration_rewards = []
        for episode in range(config['episodes']):
            obs, info = env.reset()
            state = agent.discretize_state(obs)
            total_reward = 0
            done = False
            while not done:
                discrete_action = agent.choose_action(state)
                continuous_action = agent.discrete_to_continuous_action(discrete_action)
                next_obs, reward, terminated, truncated, info = env.step(np.array([continuous_action]))
                done = terminated or truncated
                next_state = agent.discretize_state(next_obs)
                total_reward += reward
                agent.update(state, discrete_action, reward, next_state, done)
                state = next_state
            iteration_rewards.append(total_reward)
            total_episodes_trained += 1
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(iteration_rewards[-1000:])
                print(f"     Episodio {episode+1:,}/{config['episodes']:,} | "
                      f"Reward promedio: {avg_reward:.3f}")
        iteration_avg = np.mean(iteration_rewards)
        iteration_std = np.std(iteration_rewards)
        iteration_min = np.min(iteration_rewards)
        iteration_max = np.max(iteration_rewards)
        iteration_summary = {
            'iteration': iteration,
            'episodes': config['episodes'],
            'alpha': config['alpha'],
            'gamma': config['gamma'],
            'epsilon': config['epsilon'],
            'avg_reward': iteration_avg,
            'std_reward': iteration_std,
            'min_reward': iteration_min,
            'max_reward': iteration_max,
            'total_episodes': total_episodes_trained
        }
        iteration_summaries.append(iteration_summary)
        all_episode_rewards.extend(iteration_rewards)
        if iteration % 3 == 0:
            q_table_snapshot = {
                'iteration': iteration,
                'q_table': dict(agent.q_table),
                'states_explored': len(agent.q_table)
            }
            q_table_evolution.append(q_table_snapshot)
        print(f"   ✅ Completada | Reward: {iteration_avg:.3f}±{iteration_std:.3f} | "
              f"Estados explorados: {len(agent.q_table):,}")
        print("")
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()
    print(f"🎉 ENTRENAMIENTO PROGRESIVO COMPLETADO!")
    print(f"="*60)
    print(f"⏱️  Tiempo total: {total_duration/3600:.2f} horas")
    print(f"📊 Episodios entrenados: {total_episodes_trained:,}")
    print(f"🗺️  Estados explorados: {len(agent.q_table):,}")
    print(f"📈 Recompensa final: {np.mean(all_episode_rewards[-1000:]):.3f}")
    save_results(agent, iteration_summaries, all_episode_rewards, 
                 q_table_evolution, total_duration, start_time)
    create_visualizations(iteration_summaries, all_episode_rewards, q_table_evolution)
    env.close()
    return agent, iteration_summaries

def save_results(agent, iteration_summaries, all_episode_rewards, 
                 q_table_evolution, total_duration, start_time):
    os.makedirs("models", exist_ok=True)
    model_path = "models/progressive_stochastic_qlearning.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'algorithm': 'Progressive Stochastic Q-Learning',
            'description': 'Entrenamiento progresivo con 18 iteraciones',
            'q_table': dict(agent.q_table),
            'state_bins': agent.state_bins,
            'n_actions': agent.n_actions,
            'stoch_k': agent.stoch_k,
            'iteration_summaries': iteration_summaries,
            'all_episode_rewards': all_episode_rewards,
            'q_table_evolution': q_table_evolution,
            'total_episodes': len(all_episode_rewards),
            'total_duration_seconds': total_duration,
            'states_explored': len(agent.q_table),
            'final_hyperparams': {
                'alpha': agent.alpha,
                'gamma': agent.gamma,
                'epsilon': agent.epsilon
            },
            'timestamp': start_time.strftime("%Y-%m-%d %H:%M:%S")
        }, f)
    print(f"💾 Modelo guardado: {model_path}")

def create_visualizations(iteration_summaries, all_episode_rewards, q_table_evolution):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    iterations = [s['iteration'] for s in iteration_summaries]
    avg_rewards = [s['avg_reward'] for s in iteration_summaries]
    std_rewards = [s['std_reward'] for s in iteration_summaries]
    ax1.errorbar(iterations, avg_rewards, yerr=std_rewards, 
                capsize=5, marker='o', linewidth=2, markersize=6)
    ax1.set_title('Evolución de Recompensas por Iteración', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Recompensa Promedio')
    ax1.grid(True, alpha=0.3)
    alphas = [s['alpha'] for s in iteration_summaries]
    gammas = [s['gamma'] for s in iteration_summaries]
    epsilons = [s['epsilon'] for s in iteration_summaries]
    ax2.plot(iterations, alphas, 'o-', label='Alpha', linewidth=2, markersize=4)
    ax2.plot(iterations, gammas, 's-', label='Gamma', linewidth=2, markersize=4)
    ax2.plot(iterations, epsilons, '^-', label='Epsilon', linewidth=2, markersize=4)
    ax2.set_title('Evolución de Hiperparámetros', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Valor')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    window_size = 500
    if len(all_episode_rewards) >= window_size:
        moving_avg = []
        for i in range(window_size, len(all_episode_rewards) + 1):
            moving_avg.append(np.mean(all_episode_rewards[i-window_size:i]))
        ax3.plot(range(window_size, len(all_episode_rewards) + 1), moving_avg, 
                linewidth=2, color='green')
        ax3.set_title(f'Curva de Aprendizaje (Ventana móvil {window_size})', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Episodio')
        ax3.set_ylabel('Recompensa Promedio')
        ax3.grid(True, alpha=0.3)
    if q_table_evolution:
        exploration_iterations = [snapshot['iteration'] for snapshot in q_table_evolution]
        states_explored = [snapshot['states_explored'] for snapshot in q_table_evolution]
        ax4.plot(exploration_iterations, states_explored, 'o-', 
                linewidth=2, markersize=8, color='purple')
        ax4.set_title('Exploración de Estados', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Iteración')
        ax4.set_ylabel('Estados Explorados')
        ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('progressive_stochastic_qlearning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📈 Gráficos guardados: progressive_stochastic_qlearning_analysis.png")
    create_summary_table(iteration_summaries)

def create_summary_table(iteration_summaries):
    print(f"\n📋 TABLA RESUMEN - ENTRENAMIENTO PROGRESIVO STOCHASTIC Q-LEARNING")
    print("="*100)
    print(f"{'Iter':<4} | {'Episodes':<8} | {'Alpha':<5} | {'Gamma':<5} | {'Epsilon':<7} | {'Recompensa':<10} | {'Estados':<8}")
    print("-"*100)
    for s in iteration_summaries:
        print(f"{s['iteration']:<4} | {s['episodes']:<8} | {s['alpha']:<5} | "
              f"{s['gamma']:<5} | {s['epsilon']:<7} | {s['avg_reward']:<10.0f} | "
              f"{len(iteration_summaries):<8}")
    print("-"*100)
    print(f"Total: {sum(s['episodes'] for s in iteration_summaries):,} episodios (90,000 episodios esperados)")

if __name__ == "__main__":
    agent, summaries = train_progressive_stochastic_qlearning() 