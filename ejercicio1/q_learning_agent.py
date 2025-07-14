#!/usr/bin/env python3
"""
Q-Learning Agent para DescentEnv
Implementa discretización del espacio de estados y acciones para Q-Learning tabular
"""

import numpy as np
import pickle
import argparse
import os
from collections import defaultdict
import random

import descent_env


class QLearningAgent:
    def __init__(self, n_actions, state_bins, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Q-Learning Agent con discretización de estados
        
        Args:
            n_actions: Número de acciones discretas
            state_bins: Lista con número de bins para cada dimensión del estado
            alpha: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon: Probabilidad de exploración
        """
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.n_actions = n_actions
        self.state_bins = state_bins
        
    def discretize_state(self, obs_dict):
        """
        Discretiza las observaciones del entorno DescentEnv
        
        Args:
            obs_dict: Diccionario con observaciones {altitude, vz, target_altitude, runway_distance}
        
        Returns:
            tuple: Estado discretizado
        """
        # Extraer valores de las observaciones normalizadas
        altitude = obs_dict["altitude"][0]
        vz = obs_dict["vz"][0] 
        target_altitude = obs_dict["target_altitude"][0]
        runway_distance = obs_dict["runway_distance"][0]
        
        # Discretizar cada dimensión usando bins uniformes
        # Asumiendo que los valores están normalizados aproximadamente en [-3, 3]
        state_ranges = [(-3, 3), (-3, 3), (-3, 3), (-3, 3)]
        
        discretized = []
        for i, (value, n_bins, (min_val, max_val)) in enumerate(zip(
            [altitude, vz, target_altitude, runway_distance], 
            self.state_bins, 
            state_ranges
        )):
            # Clip value to range and discretize
            clipped_value = np.clip(value, min_val, max_val)
            bin_size = (max_val - min_val) / n_bins
            bin_index = int((clipped_value - min_val) / bin_size)
            bin_index = min(bin_index, n_bins - 1)  # Ensure within bounds
            discretized.append(bin_index)
            
        return tuple(discretized)

    def choose_action(self, state):
        """Selecciona acción usando epsilon-greedy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[state]))
    
    def discrete_to_continuous_action(self, discrete_action):
        """Convierte acción discreta a continua para el entorno"""
        # Mapear acciones discretas a continuas en [-1, 1]
        return (discrete_action / (self.n_actions - 1)) * 2 - 1

    def update(self, state, action, reward, next_state, done):
        """Actualiza Q-table usando Q-Learning"""
        if done:
            td_target = reward
        else:
            best_next = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * best_next
        
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def decay_epsilon(self, decay_rate, min_epsilon):
        """Decae epsilon para reducir exploración"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)


def train_agent(args):
    """Función principal de entrenamiento"""
    print(f"🚀 Iniciando entrenamiento Q-Learning con {args.episodes} episodios")
    
    # Crear entorno
    env = descent_env.DescentEnv()
    
    # Determinar número de acciones discretas
    n_actions = 5  # Discretizar acciones continuas [-1, 1] en 5 acciones
    
    # Crear agente
    agent = QLearningAgent(
        n_actions=n_actions,
        state_bins=args.bins,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.eps_start
    )
    
    # Métricas de entrenamiento
    episode_rewards = []
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        state = agent.discretize_state(obs)
        total_reward = 0
        done = False
        
        while not done:
            # Seleccionar y ejecutar acción
            discrete_action = agent.choose_action(state)
            continuous_action = agent.discrete_to_continuous_action(discrete_action)
            
            next_obs, reward, terminated, truncated, info = env.step(np.array([continuous_action]))
            done = terminated or truncated
            
            next_state = agent.discretize_state(next_obs)
            total_reward += reward
            
            # Actualizar Q-table
            agent.update(state, discrete_action, reward, next_state, done)
            
            state = next_state
        
        episode_rewards.append(total_reward)
        
        # Decay epsilon
        agent.decay_epsilon(args.eps_decay, args.eps_min)
        
        # Logging
        if (episode + 1) % args.eval == 0:
            avg_reward = np.mean(episode_rewards[-args.eval:])
            print(f"Episodio {episode + 1}/{args.episodes} | "
                  f"Recompensa media (últimos {args.eval}): {avg_reward:.3f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Crear directorio de salida
    os.makedirs(args.out, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(args.out, "q_table.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'q_table': dict(agent.q_table),
            'state_bins': agent.state_bins,
            'n_actions': agent.n_actions,
            'episode_rewards': episode_rewards
        }, f)
    
    print(f"✅ Modelo guardado en: {model_path}")
    print(f"📊 Recompensa media final (últimos 100 episodios): {np.mean(episode_rewards[-100:]):.3f}")
    
    env.close()
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Entrenar agente Q-Learning para DescentEnv')
    
    # Argumentos del entorno - no usado pero mantengo compatibilidad
    parser.add_argument('--env', type=str, default='DescentEnv-v0', 
                       help='Identificador del entorno (no usado, solo compatibilidad)')
    
    # Configuración de salida
    parser.add_argument('--out', type=str, default='models', 
                       help='Directorio de salida para modelos')
    
    # Discretización
    parser.add_argument('--bins', type=int, nargs=4, default=[6, 6, 12, 12],
                       help='Número de bins para [altitude, vz, target_altitude, runway_distance]')
    
    # Hiperparámetros de Q-Learning
    parser.add_argument('--alpha', type=float, default=0.1, 
                       help='Tasa de aprendizaje')
    parser.add_argument('--gamma', type=float, default=0.99, 
                       help='Factor de descuento')
    
    # Parámetros de exploración
    parser.add_argument('--eps_start', type=float, default=1.0, 
                       help='Epsilon inicial')
    parser.add_argument('--eps_decay', type=float, default=0.995, 
                       help='Factor de decay de epsilon')
    parser.add_argument('--eps_min', type=float, default=0.01, 
                       help='Epsilon mínimo')
    
    # Configuración de entrenamiento
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Número de episodios de entrenamiento')
    parser.add_argument('--eval', type=int, default=100, 
                       help='Frecuencia de evaluación (episodios)')
    
    args = parser.parse_args()
    
    print("🔧 Configuración del entrenamiento:")
    print(f"  Bins de estado: {args.bins}")
    print(f"  Alpha: {args.alpha}, Gamma: {args.gamma}")
    print(f"  Epsilon: {args.eps_start} → {args.eps_min} (decay: {args.eps_decay})")
    print(f"  Episodios: {args.episodes}")
    print("")
    
    # Entrenar agente
    model_path = train_agent(args)
    
    print("\n✔ descent-env listo: Agente Q-Learning entrenado y modelo guardado en", model_path)


if __name__ == "__main__":
    main()
