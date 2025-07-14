import sys
from tournament import run_tournament, depth_analysis
from tactix_env import TacTixEnv
from minimax_agent import MinimaxAgent
from expectimax_agent import ExpectimaxAgent
from random_agent import RandomTacTixAgent
from trainer_agent import TrainerAgent
from play import run_multiple_games, plot_results


def main():
    print("\nOpciones disponibles:")
    print("1. Torneo completo entre todos los agentes")
    print("2. Partidas entre dos agentes específicos")
    print("3. Análisis de profundidad (depth analysis)")
    opcion = input("\nSelecciona una opción (1/2/3): ").strip()

    if opcion == "1":
        run_tournament()
    elif opcion == "2":
        print("\nAgentes disponibles:")
        print("1. Random")
        print("2. Trainer (Easy)")
        print("3. Trainer (Medium)")
        print("4. Trainer (Hard)")
        print("5. Minimax (Depth 3)")
        print("6. Minimax (Depth 4)")
        print("7. Expectimax (Depth 3)")
        print("8. Expectimax (Depth 4)")
        a1 = input("Selecciona el número del primer agente: ").strip()
        a2 = input("Selecciona el número del segundo agente: ").strip()
        n = int(input("¿Cuántas partidas quieres correr?: ").strip())
        env = TacTixEnv(board_size=6)
        agentes = [
            RandomTacTixAgent(env),
            TrainerAgent(env, difficulty=0.1),
            TrainerAgent(env, difficulty=0.5),
            TrainerAgent(env, difficulty=0.9),
            MinimaxAgent(env, depth=3),
            MinimaxAgent(env, depth=4),
            ExpectimaxAgent(env, depth=3),
            ExpectimaxAgent(env, depth=4),
        ]
        agent1 = agentes[int(a1)-1]
        agent2 = agentes[int(a2)-1]
        results = run_multiple_games(env, agent1, agent2, num_games=n)
        print(f"\nResultados: {results}")
        plot_results(results)
    elif opcion == "3":
        depth_analysis()
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main() 