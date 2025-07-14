import argparse
import pickle
import numpy as np
import time
from descent_env import DescentEnv

def discretize_state(obs, state_bins):
    # Asume 4 variables: altitude, vz, target_altitude, runway_distance
    altitude = obs["altitude"][0]
    vz = obs["vz"][0]
    target_altitude = obs["target_altitude"][0]
    runway_distance = obs["runway_distance"][0]
    state_ranges = [(-3, 3), (-3, 3), (-3, 3), (-3, 3)]
    discretized = []
    for value, n_bins, (min_val, max_val) in zip(
        [altitude, vz, target_altitude, runway_distance],
        state_bins,
        state_ranges
    ):
        clipped_value = np.clip(value, min_val, max_val)
        bin_size = (max_val - min_val) / n_bins
        bin_index = int((clipped_value - min_val) / bin_size)
        bin_index = min(bin_index, n_bins - 1)
        discretized.append(bin_index)
    return tuple(discretized)

def main():
    parser = argparse.ArgumentParser(description="Testea visualmente un agente entrenado (Q-table) en DescentEnv.")
    parser.add_argument('--model', type=str, default='models/entrenamiento_1.pkl', help='Ruta al archivo .pkl del modelo')
    parser.add_argument('--episodes', type=int, default=5, help='Cantidad de episodios de testeo')
    args = parser.parse_args()

    # Cargar modelo
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    q_table = model['q_table']
    state_bins = model['state_bins']
    n_actions = model['n_actions']

    print(f"\n🔍 Probando modelo: {args.model}")
    print(f"   Acciones: {n_actions} | State bins: {state_bins}")
    print(f"   Episodios de test: {args.episodes}\n")

    def discrete_to_continuous_action(discrete_action):
        return (discrete_action / (n_actions - 1)) * 2 - 1

    env = DescentEnv(render_mode='human')
    rewards = []
    steps_list = []
    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        state = discretize_state(obs, state_bins)
        done = False
        total_reward = 0
        steps = 0
        while not done:
            # Política óptima (greedy)
            q_values = q_table[state] if state in q_table else np.zeros(n_actions)
            discrete_action = int(np.argmax(q_values))
            continuous_action = discrete_to_continuous_action(discrete_action)
            obs, reward, terminated, truncated, info = env.step(np.array([continuous_action]))
            done = terminated or truncated
            state = discretize_state(obs, state_bins)
            total_reward += reward
            steps += 1
            env.render()
            time.sleep(0.05)
        rewards.append(total_reward)
        steps_list.append(steps)
        print(f"Episodio {ep}: Reward total = {total_reward:.2f} | Pasos = {steps}")
    env.close()
    print("\nResumen:")
    print(f"Reward promedio: {np.mean(rewards):.2f}")
    print(f"Pasos promedio: {np.mean(steps_list):.1f}")

if __name__ == "__main__":
    main() 