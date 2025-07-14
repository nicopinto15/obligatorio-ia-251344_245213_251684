# Proyecto BORED: Agentes Inteligentes para TacTix

Este proyecto forma parte de la iniciativa SkyNotCrash™ y corresponde a la tarea "Board-Oriented Reasoning for Emergent Domination" (BORED). El objetivo principal es desarrollar y evaluar agentes de inteligencia artificial capaces de competir en el juego de tablero **TacTix**.

Para ello, se han implementado dos de las técnicas fundamentales en la teoría de juegos adversariales:

1.  **Minimax con poda Alfa-Beta**: Un algoritmo que busca la jugada óptima asumiendo que el oponente también jugará de manera perfecta.
2.  **Expectimax**: una variante de Minimax que, en lugar de asumir un oponente perfecto, modela al rival como un agente racional pero falible, tomando decisiones basadas en probabilidades.

El propósito de este trabajo es no solo implementar estos agentes, sino también experimentar con distintas **funciones de evaluación (heurísticas)** para dotarlos de una "comprensión" del juego, y finalmente comparar el rendimiento y comportamiento de ambas estrategias.

## El Juego: TacTix

TacTix es un juego de estrategia imparcial para dos jugadores. Un juego es "imparcial" cuando los movimientos disponibles dependen únicamente del estado del juego, no de qué jugador está jugando.

**Reglas y Mecánicas:**

- **Tablero:** El juego se desarrolla en una cuadrícula de N x N (en nuestra implementación, por defecto es de 6x6). Inicialmente, todas las casillas están "disponibles".
- **Movimientos:** En su turno, un jugador debe elegir una fila o una columna y retirar un grupo contiguo de una o más casillas que no hayan sido retiradas previamente.
- **Condición de Victoria:** La partida sigue las reglas de un "juego normal". Esto significa que el **último jugador en realizar un movimiento válido gana**. La partida termina cuando no quedan más piezas en el tablero.

El entorno del juego está definido en `tactix_env.py`, utilizando la librería `gymnasium` para estandarizar la interacción entre los agentes y el juego.

## Estructura del Código

El proyecto está organizado en los siguientes archivos clave dentro de la carpeta `tactix/`:

- `tactix_env.py`: Define la clase `TacTixEnv`, que contiene toda la lógica del juego: el tablero, las reglas, la validación de jugadas y el estado de la partida.
- `agent.py`: Una clase base abstracta que define la interfaz común para todos los agentes. Garantiza que cualquier agente tenga un método `act` para interactuar con el entorno.
- `minimax_agent.py`: Contiene la implementación del `MinimaxAgent`. Este agente utiliza el algoritmo Minimax con la optimización de poda Alfa-Beta para tomar decisiones.
- `expectimax_agent.py`: Contiene la implementación del `ExpectimaxAgent`, que utiliza una versión probabilística del árbol de búsqueda para decidir su jugada.
- `random_agent.py`: Un agente de base que realiza movimientos válidos al azar. Sirve como un punto de referencia para medir la efectividad de los agentes inteligentes.
- `play.py`: Un script que permite jugar una partida de TacTix en la consola, por ejemplo, enfrentando a un humano contra uno de los agentes de IA.
- `tournament.py`: Un script para ejecutar un torneo automático entre diferentes agentes, mostrando los resultados para comparar su rendimiento.
- `tactix.ipynb`: Un Jupyter Notebook utilizado para la experimentación, visualización de resultados y análisis del comportamiento de los agentes.

A continuación, se detalla el diseño de los dos agentes principales.

## Diseño del Agente: `MinimaxAgent`

El `MinimaxAgent` es un adversario clásico, diseñado para juegos de suma cero y de información perfecta como TacTix. Su objetivo es maximizar su propia puntuación asumiendo que su oponente (el minimizador) siempre jugará de forma óptima para minimizar la puntuación del agente.

### Poda Alfa-Beta

Para evitar explorar el árbol de juego completo, que sería computacionalmente inviable, el agente implementa la **poda Alfa-Beta**. Esta optimización "poda" ramas enteras del árbol de búsqueda que se sabe que no pueden influir en el resultado final.

- **Alfa**: Representa el mejor valor encontrado hasta ahora para el jugador maximizador (nuestro agente).
- **Beta**: Representa el mejor valor encontrado hasta ahora para el jugador minimizador (el oponente).
  La poda ocurre si en algún momento `alfa` es mayor o igual que `beta`, lo que significa que el oponente tiene una opción mejor en otra rama del árbol y nunca permitiría llegar a la posición actual.

### Función de Evaluación Heurística

La verdadera inteligencia del agente reside en su función de evaluación (`evaluate_board`), que le permite asignar una puntuación a un estado del tablero sin necesidad de explorar hasta el final de la partida. Nuestra implementación utiliza una **función de evaluación compuesta**, que es una suma ponderada de cuatro heurísticas diferentes, lo que permite un análisis más robusto y matizado del estado del juego.

La fórmula de evaluación es:
`puntuación = 0.4 * nim_sum + 0.3 * movilidad + 0.2 * piezas + 0.1 * estructura`

A continuación se detalla cada componente:

1.  **`nim_sum_evaluation` (Peso: 0.4)**

    - **Qué es:** Esta es la heurística más potente. Se basa en la [teoría de Sprague-Grundy](https://es.wikipedia.org/wiki/Teorema_de_Sprague-Grundy) para juegos imparciales. Calcula el "Nim-sum" (una suma XOR) de las longitudes de todos los segmentos continuos de piezas en el tablero.
    - **Por qué funciona:** En la teoría de juegos combinatorios, una posición con un Nim-sum de cero es una "posición P" (el jugador _anterior_ tiene la estrategia ganadora), mientras que una posición con un Nim-sum distinto de cero es una "posición N" (el jugador _siguiente_ tiene una estrategia ganadora).
    - **Implementación:** La función devuelve -100 si el Nim-sum es cero (posición perdedora) y un valor positivo en caso contrario, guiando al agente a dejar al oponente en posiciones con Nim-sum cero.

2.  **`mobility_evaluation` (Peso: 0.3)**

    - **Qué es:** Mide el número total de movimientos válidos disponibles desde la posición actual.
    - **Por qué funciona:** Intuitivamente, tener más opciones de movimiento (alta movilidad) es ventajoso, ya que restringe menos las acciones futuras y puede forzar al oponente a un estado con menos opciones.

3.  **`piece_count_evaluation` (Peso: 0.2)**

    - **Qué es:** Simplemente cuenta el número total de piezas que quedan en el tablero.
    - **Por qué funciona:** En un juego normal, la partida se acerca a su fin a medida que se quitan piezas. Esta heurística ayuda al agente a tener una noción del progreso del juego.

4.  **`structure_evaluation` (Peso: 0.1)**
    - **Qué es:** Evalúa la configuración estructural de las piezas. Concretamente, penaliza las piezas que están aisladas (sin vecinos adyacentes) y recompensa las que forman estructuras conectadas.
    - **Por qué funciona:** Forzar la creación de piezas aisladas puede ser una estrategia para simplificar el juego a favor de un jugador que pueda calcular mejor las consecuencias. Esta heurística anima al agente a mantener sus piezas agrupadas, lo que puede complicar el cálculo para el oponente.

La ponderación de estas heurísticas fue ajustada experimentalmente para dar prioridad a la estrategia matemática (Nim-sum) y a la flexibilidad táctica (movilidad), resultando en un agente muy competente.

## Diseño del Agente: `ExpectimaxAgent`

El `ExpectimaxAgent` es una alternativa al Minimax, especialmente útil cuando no se puede asumir que el oponente jugará de manera perfecta. En lugar de tener un "minimizador", el algoritmo Expectimax introduce un "nodo de probabilidad" para modelar al oponente.

### La Diferencia Clave: Nodos de Probabilidad (Chance Nodes)

Mientras que Minimax se prepara para el peor escenario posible, Expectimax calcula el **valor esperado** de sus movimientos. Lo hace promediando los resultados de todas las acciones posibles del oponente, ponderadas por la probabilidad de que cada acción ocurra.

- **Nodo MAX (Nuestro Agente):** Funciona igual que en Minimax. Elige el movimiento que lleva al resultado con el valor máximo.
- **Nodo Chance (El Oponente):** En lugar de elegir el movimiento con el valor mínimo, calcula un promedio de los valores de todos los estados sucesores.

Este enfoque hace que el agente juegue de una forma más "optimista" o "realista". Puede decidir tomar un camino que, aunque riesgoso contra un oponente perfecto, tiene una alta probabilidad de éxito contra un oponente que podría cometer un error.

### Modelado del Oponente con Softmax

Para que Expectimax funcione, necesita un modelo de probabilidades del oponente. ¿Cómo estimamos la probabilidad de cada una de las jugadas del rival?

Nuestra implementación utiliza la **función de evaluación** del propio agente para "ponerse en el lugar" del oponente.

1. Para cada jugada posible del oponente, se evalúa el estado resultante del tablero con la misma función heurística (`evaluate_board`).
2. Se obtienen así puntuaciones para cada jugada posible del rival.
3. Estas puntuaciones se convierten en una distribución de probabilidad utilizando la función **Softmax**.

Softmax asigna una mayor probabilidad a las jugadas con una puntuación más alta. Esto modela a un oponente **racional**: es más probable que elija buenas jugadas que malas, pero no está garantizado que elija siempre la mejor. Esto simula de forma más realista a un jugador humano o a un agente no determinista.

### Comparación Directa: Minimax vs. Expectimax

| Característica | Minimax                                                      | Expectimax                                                             |
| :------------- | :----------------------------------------------------------- | :--------------------------------------------------------------------- |
| **Oponente**   | Asume un adversario perfecto y óptimo.                       | Asume un adversario racional pero falible/probabilístico.              |
| **Estrategia** | Pesimista: se prepara para el peor caso.                     | Realista: juega para el mejor resultado esperado (promedio).           |
| **Ideal para** | Juegos deterministas contra IAs perfectas.                   | Juegos con elementos de azar o contra oponentes imperfectos (humanos). |
| **Robustez**   | Puede ser explotado si el oponente no juega de forma óptima. | Más robusto contra oponentes que cometen errores.                      |

En el contexto de TacTix, un juego sin azar, Minimax es teóricamente superior si el oponente es igualmente perfecto. Sin embargo, Expectimax puede lograr mejores resultados contra una gama más amplia de oponentes, incluidos los otros agentes o un jugador humano.

## Guía de Uso

Para ver a los agentes en acción, puedes utilizar los scripts proporcionados.

### Jugar una Partida (Humano vs. IA)

Puedes ejecutar el script `play.py` para jugar una partida en la terminal. Puedes elegir qué agente quieres que controle a cada jugador. Por ejemplo, para jugar como el Jugador 1 contra el agente Minimax:

```bash
python tactix/play.py
```

Sigue las instrucciones en pantalla para seleccionar los agentes y realizar tus movimientos.

### Ejecutar un Torneo

Para comparar el rendimiento de los agentes de forma automática, utiliza el script `tournament.py`. Este script enfrentará a los agentes entre sí en múltiples partidas y mostrará un resumen de los resultados.

```bash
python tactix/tournament.py
```

Esto es fundamental para obtener los datos necesarios para el informe final y evaluar cuantitativamente qué agente y qué heurísticas funcionan mejor.
