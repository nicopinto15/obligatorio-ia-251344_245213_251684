#!/usr/bin/env python3
"""
Stochastic Q-Learning Agent para DescentEnv
Implementación basada en el artículo "Stochastic Q-learning for Large Discrete Action Spaces"
Fourati et al. (2024) - https://arxiv.org/abs/2405.10310

En lugar de maximizar sobre todas las acciones, selecciona un subconjunto estocástico 
de tamaño O(log(n)) para reducir complejidad computacional.
"""

import numpy as np
import pickle
import os
import argparse
from collections import defaultdict, deque
import random
from datetime import datetime
import math
import descent_env


class StochasticQLearningAgent:
    def __init__(self, n_actions, state_bins, alpha=0.1, gamma=0.99, epsilon=0.1, stoch_k=None):
        """
        Stochastic Q-Learning Agent
        
        Args:
            n_actions: Número total de acciones discretas
            state_bins: Lista con número de bins para cada dimensión del estado
            alpha: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon: Probabilidad de exploración
            stoch_k: Tamaño del subconjunto estocástico (default: ceil(log2(n_actions)))
        """
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.n_actions = n_actions
        self.state_bins = state_bins
        
      
        self.stoch_k = stoch_k if stoch_k is not None else max(1, math.ceil(math.log2(n_actions)))
        
        print(f"🎲 Stochastic Q-learning configurado:")
        print(f"   Acciones totales: {n_actions}")
        print(f"   Subconjunto estocástico k: {self.stoch_k}")
        print(f"   Reducción computacional: {n_actions/self.stoch_k:.1f}x")
        
    def discretize_state(self, obs_dict):
        """Discretización del estado (igual que Q-learning tradicional)"""
        altitude = obs_dict["altitude"][0]
        vz = obs_dict["vz"][0] 
        target_altitude = obs_dict["target_altitude"][0]
        runway_distance = obs_dict["runway_distance"][0]
        
        state_ranges = [(-3, 3), (-3, 3), (-3, 3), (-3, 3)]
        
        discretized = []
        for i, (value, n_bins, (min_val, max_val)) in enumerate(zip(
            [altitude, vz, target_altitude, runway_distance], 
            self.state_bins, 
            state_ranges
        )):
            clipped_value = np.clip(value, min_val, max_val)
            bin_size = (max_val - min_val) / n_bins
            bin_index = int((clipped_value - min_val) / bin_size)
            bin_index = min(bin_index, n_bins - 1)
            discretized.append(bin_index)
            
        return tuple(discretized)

    def stochastic_argmax(self, q_values):
        """
        Implementación del argmax estocástico del paper
        En lugar de evaluar todas las acciones, selecciona un subconjunto aleatorio
        """
        # Seleccionar k acciones aleatorias
        action_indices = np.random.choice(self.n_actions, size=self.stoch_k, replace=False)
        
        # Obtener Q-values solo para el subconjunto seleccionado
        subset_q_values = q_values[action_indices]
        
        # Encontrar el índice del máximo dentro del subconjunto
        best_in_subset = np.argmax(subset_q_values)
        
        # Devolver la acción original correspondiente
        return action_indices[best_in_subset]

    def choose_action(self, state):
        """Selección de acción usando epsilon-greedy con argmax estocástico"""
        if random.random() < self.epsilon:
            # Exploración: acción completamente aleatoria
            return random.randint(0, self.n_actions - 1)
        else:
            # Explotación: argmax estocástico
            q_values = self.q_table[state]
            return self.stochastic_argmax(q_values)
    
    def discrete_to_continuous_action(self, discrete_action):
        """Convierte acción discreta a continua para el entorno"""
        return (discrete_action / (self.n_actions - 1)) * 2 - 1

    def update(self, state, action, reward, next_state, done):
        """
        Actualización Q-table usando Stochastic Q-learning
        La diferencia clave: usar argmax estocástico para el siguiente estado
        """
        if done:
            td_target = reward
        else:
            # Aquí está la innovación del paper: usar argmax estocástico
            next_q_values = self.q_table[next_state]
            best_next_action = self.stochastic_argmax(next_q_values)
            best_next = next_q_values[best_next_action]
            td_target = reward + self.gamma * best_next
        
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def decay_epsilon(self, decay_rate, min_epsilon):
        """Decae epsilon para reducir exploración"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


class EarlyStoppingMonitor:
    def __init__(self, patience=200, min_improvement=0.5, window_size=100):
        self.patience = patience
        self.min_improvement = min_improvement
        self.window_size = window_size
        
        self.rewards_history = deque(maxlen=window_size)
        self.best_avg_reward = -float('inf')
        self.episodes_without_improvement = 0
        
    def should_stop(self, current_reward):
        self.rewards_history.append(current_reward)
        
        if len(self.rewards_history) < self.window_size:
            return False, ""
        
        current_avg = np.mean(self.rewards_history)
        improvement = current_avg - self.best_avg_reward
        
        if improvement > self.min_improvement:
            self.best_avg_reward = current_avg
            self.episodes_without_improvement = 0
            return False, f"Mejora detectada: {improvement:.3f}"
        else:
            self.episodes_without_improvement += 1
            
        if self.episodes_without_improvement >= self.patience:
            return True, f"Sin mejora por {self.patience} episodios (mejor promedio: {self.best_avg_reward:.3f})"
        
        return False, f"Sin mejora por {self.episodes_without_improvement}/{self.patience} episodios"


def train_stochastic_qlearning(stoch_k=None):
    """Entrena el agente Stochastic Q-learning"""
    
    print("🎲 Iniciando entrenamiento Stochastic Q-Learning")
    print("📄 Basado en: Fourati et al. (2024) - Stochastic Q-learning for Large Discrete Action Spaces")
    print("🔗 https://arxiv.org/abs/2405.10310")
    print("")
    print("💡 Innovación clave: Usar subconjunto estocástico O(log(n)) en lugar de todas las acciones")
    print("📊 Criterios de parada:")
    print("  • Paciencia: 200 episodios sin mejora")
    print("  • Mejora mínima: 0.5 en recompensa promedio")
    print("  • Ventana de evaluación: 100 episodios")
    print("  • Epsilon mínimo: 0.01")
    print("")
    
    # Configuración
    env = descent_env.DescentEnv()
    n_actions = 5
    state_bins = [6, 6, 12, 12]
    
    # Crear agente Stochastic Q-learning
    agent = StochasticQLearningAgent(
        n_actions=n_actions,
        state_bins=state_bins,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        stoch_k=stoch_k
    )
    
    # Monitor de parada temprana
    early_stopping = EarlyStoppingMonitor(patience=200, min_improvement=0.5, window_size=100)
    
    # Métricas
    episode_rewards = []
    max_episodes = 10000
    eval_frequency = 50
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    start_time = datetime.now()
    
    for episode in range(max_episodes):
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
            
            # Actualización usando Stochastic Q-learning
            agent.update(state, discrete_action, reward, next_state, done)
            state = next_state
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon(epsilon_decay, epsilon_min)
        
        # Evaluar progreso
        if (episode + 1) % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            
            # Verificar criterios de parada
            should_stop, reason = early_stopping.should_stop(total_reward)
            epsilon_converged = agent.epsilon <= epsilon_min
            
            print(f"Episodio {episode + 1:4d} | "
                  f"Recompensa promedio: {avg_reward:7.3f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Estado: {reason}")
            
            # Condiciones de parada
            if should_stop and epsilon_converged:
                print(f"\n🎯 ENTRENAMIENTO STOCHASTIC Q-LEARNING COMPLETADO!")
                print(f"   Razón: {reason}")
                print(f"   Episodios totales: {episode + 1}")
                print(f"   Recompensa final: {avg_reward:.3f}")
                break
            elif should_stop:
                print(f"   ⏳ Convergencia detectada, esperando epsilon mínimo...")
            elif epsilon_converged:
                print(f"   ⏳ Epsilon mínimo alcanzado, esperando convergencia...")
    
    else:
        print(f"\n⚠️  Entrenamiento terminado por límite máximo ({max_episodes} episodios)")
        reason = f"Límite máximo ({max_episodes} episodios)"
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    model_path = "models/stochastic_q_table.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'algorithm': 'Stochastic Q-learning',
            'paper_reference': 'Fourati et al. (2024) - https://arxiv.org/abs/2405.10310',
            'q_table': dict(agent.q_table),
            'state_bins': agent.state_bins,
            'n_actions': agent.n_actions,
            'stoch_k': agent.stoch_k,
            'episode_rewards': episode_rewards,
            'final_episode': episode + 1,
            'convergence_reason': reason,
            'training_config': {
                'alpha': agent.alpha,
                'gamma': agent.gamma,
                'epsilon_start': agent.epsilon_start,
                'epsilon_min': epsilon_min,
                'epsilon_decay': epsilon_decay,
                'max_episodes': max_episodes
            },
            'training_duration_seconds': duration,
            'timestamp': start_time.strftime("%Y-%m-%d %H:%M:%S")
        }, f)
    
    print(f"\n✅ STOCHASTIC Q-LEARNING GUARDADO:")
    print(f"   Archivo: {model_path}")
    print(f"   Duración: {duration/60:.1f} minutos")
    
    # Estadísticas finales y comparación
    final_100_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print(f"\n📈 Estadísticas finales:")
    print(f"   Episodios entrenados: {len(episode_rewards):,}")
    print(f"   Recompensa promedio (últimos 100): {final_100_avg:.3f}")
    print(f"   Mejor recompensa: {max(episode_rewards):.3f}")
    print(f"   Estados explorados: {len(agent.q_table):,}")
    print(f"   Epsilon final: {agent.epsilon:.3f}")
    
    print(f"\n🔬 VENTAJAS DEL STOCHASTIC Q-LEARNING:")
    print(f"   Reducción computacional: {n_actions/agent.stoch_k:.1f}x por paso")
    print(f"   Complejidad por acción: O(log(n)) vs O(n) tradicional")
    print(f"   Escalabilidad: Mejor para espacios de acción grandes")
    
    env.close()
    return model_path


def compare_with_traditional():
    """Función para comparar Stochastic vs Traditional Q-learning"""
    print("\n" + "="*60)
    print("🆚 COMPARACIÓN: STOCHASTIC vs TRADITIONAL Q-LEARNING")
    print("="*60)
    
    # Simular diferentes tamaños de espacio de acciones
    action_spaces = [5, 10, 50, 100, 500, 1000]
    
    print(f"{'Acciones':<10} | {'Traditional':<12} | {'Stochastic':<12} | {'Speedup':<8}")
    print("-" * 50)
    
    for n_actions in action_spaces:
        traditional_complexity = n_actions  # O(n)
        stoch_k = max(1, math.ceil(math.log2(n_actions)))
        stochastic_complexity = stoch_k  # O(log(n))
        speedup = traditional_complexity / stochastic_complexity
        
        print(f"{n_actions:<10} | {traditional_complexity:<12} | {stochastic_complexity:<12} | {speedup:<8.1f}x")


def main():
    parser = argparse.ArgumentParser(description='Entrenar Stochastic Q-Learning para DescentEnv')
    
    parser.add_argument('--stoch_k', type=int, help='Tamaño del subconjunto estocástico (default: ceil(log2(n_actions)))')
    parser.add_argument('--compare', action='store_true', help='Mostrar comparación de complejidad computacional')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_traditional()
        return
    
    # Entrenar Stochastic Q-learning
    model_path = train_stochastic_qlearning(args.stoch_k)
    
    print(f"\n✔ Stochastic Q-learning entrenado y modelo guardado en {model_path}")
    print(f"📚 Implementación basada en: https://arxiv.org/abs/2405.10310")


if __name__ == "__main__":
    main() 