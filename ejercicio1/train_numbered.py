#!/usr/bin/env python3
"""
Entrenamiento Q-Learning numerado para DescentEnv
Permite hacer múltiples entrenamientos guardando cada uno por separado
"""

import numpy as np
import pickle
import os
import argparse
from collections import defaultdict, deque
import random
from datetime import datetime
import descent_env


class QLearningAgentNumbered:
    def __init__(self, n_actions, state_bins, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.n_actions = n_actions
        self.state_bins = state_bins
        
    def discretize_state(self, obs_dict):
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

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))
    
    def discrete_to_continuous_action(self, discrete_action):
        return (discrete_action / (self.n_actions - 1)) * 2 - 1

    def update(self, state, action, reward, next_state, done):
        if done:
            td_target = reward
        else:
            best_next = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * best_next
        
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def decay_epsilon(self, decay_rate, min_epsilon):
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


def get_next_training_number():
    """Encuentra el próximo número de entrenamiento disponible"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        return 1
    
    existing_numbers = []
    for filename in os.listdir(models_dir):
        if filename.startswith("entrenamiento_") and filename.endswith(".pkl"):
            try:
                number = int(filename.split("_")[1].split(".")[0])
                existing_numbers.append(number)
            except ValueError:
                continue
    
    return max(existing_numbers) + 1 if existing_numbers else 1


def train_numbered_agent(training_number=None):
    """Entrena un agente con número específico"""
    
    if training_number is None:
        training_number = get_next_training_number()
    
    print(f"🚀 ENTRENAMIENTO {training_number} - Q-Learning DescentEnv")
    print("📊 Criterios de parada:")
    print("  • Máximo: 10,000 episodios")
    print("  • Paciencia: 200 episodios sin mejora")
    print("  • Mejora mínima: 0.5 en recompensa promedio")
    print("  • Ventana de evaluación: 100 episodios")
    print("  • Epsilon mínimo: 0.01")
    print("")
    
    # Configuración
    env = descent_env.DescentEnv()
    n_actions = 5
    state_bins = [6, 6, 12, 12]
    
    # Crear agente
    agent = QLearningAgentNumbered(
        n_actions=n_actions,
        state_bins=state_bins,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0
    )
    
    # Monitor de parada temprana
    early_stopping = EarlyStoppingMonitor(patience=200, min_improvement=0.5, window_size=100)
    
    # Métricas
    episode_rewards = []
    max_episodes = 70000  # Comparable con otros estudiantes
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
                print(f"\n🎯 ENTRENAMIENTO {training_number} COMPLETADO!")
                print(f"   Razón: {reason}")
                print(f"   Episodios totales: {episode + 1}")
                print(f"   Recompensa final: {avg_reward:.3f}")
                break
            elif should_stop:
                print(f"   ⏳ Convergencia detectada, esperando epsilon mínimo...")
            elif epsilon_converged:
                print(f"   ⏳ Epsilon mínimo alcanzado, esperando convergencia...")
    
    else:
        print(f"\n⚠️  Entrenamiento {training_number} terminado por límite máximo ({max_episodes} episodios)")
        reason = f"Límite máximo ({max_episodes} episodios)"
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    model_path = f"models/entrenamiento_{training_number}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'training_number': training_number,
            'q_table': dict(agent.q_table),
            'state_bins': agent.state_bins,
            'n_actions': agent.n_actions,
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
    
    print(f"\n✅ ENTRENAMIENTO {training_number} GUARDADO:")
    print(f"   Archivo: {model_path}")
    print(f"   Duración: {duration/60:.1f} minutos")
    
    # Estadísticas finales
    final_100_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print(f"📈 Estadísticas finales:")
    print(f"   Episodios entrenados: {len(episode_rewards):,}")
    print(f"   Recompensa promedio (últimos 100): {final_100_avg:.3f}")
    print(f"   Mejor recompensa: {max(episode_rewards):.3f}")
    print(f"   Estados explorados: {len(agent.q_table):,}")
    print(f"   Epsilon final: {agent.epsilon:.3f}")
    
    env.close()
    return model_path


def list_trainings():
    """Lista todos los entrenamientos existentes"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("📁 No existe el directorio models/")
        return
    
    trainings = []
    for filename in os.listdir(models_dir):
        if filename.startswith("entrenamiento_") and filename.endswith(".pkl"):
            try:
                number = int(filename.split("_")[1].split(".")[0])
                trainings.append((number, filename))
            except ValueError:
                continue
    
    if not trainings:
        print("📝 No hay entrenamientos numerados encontrados")
        return
    
    trainings.sort()
    
    print(f"\n📊 ENTRENAMIENTOS EXISTENTES:")
    print(f"{'Núm.':<4} | {'Archivo':<25} | {'Fecha modificación':<20}")
    print("-" * 55)
    
    for number, filename in trainings:
        filepath = os.path.join(models_dir, filename)
        mod_time = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{number:<4} | {filename:<25} | {mod_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenamiento Q-Learning numerado para DescentEnv')
    
    parser.add_argument('--number', type=int, help='Número específico de entrenamiento (auto si no se especifica)')
    parser.add_argument('--list', action='store_true', help='Lista entrenamientos existentes')
    
    args = parser.parse_args()
    
    if args.list:
        list_trainings()
    else:
        train_numbered_agent(args.number) 