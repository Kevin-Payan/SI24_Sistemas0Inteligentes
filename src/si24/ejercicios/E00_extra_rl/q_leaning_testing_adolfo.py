import gymnasium as gym
import numpy as np


class RandomAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_actions = env.action_space.n

        # Tabla estados x acciones
        self.Q = np.zeros((env.observation_space.n,
                           env.action_space.n))
        # Parameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def act(self, observation):
        return self.action_space.sample()

 
    def step(self, state, action, reward, next_state):
        pass


class QLearningAgent(RandomAgent):
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1): #el algoritmo inicia con estos params.
        super().__init__(env, alpha, gamma, epsilon)

    def act(self, observation):
        if np.random.random() < self.epsilon: #si es menor a epsilon, se va a arriesgar
            return env.action_space.sample()  # Exploration
        else:
            return np.argmax(self.Q[observation])  # Exploitation

    # Update Q values using Q-learning
    def step(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        # TODO: Implementa la actualización de Q-learning usando la ecuación vista en clase
        self.Q[state][action] = self.Q[state][action] + self.alpha*(reward + self.gamma*max(self.Q[next_state] -  self.Q[state][action]))


if __name__ == "__main__":
    # TODO:
    # Este ejercicio cuenta como 5 pts extra en el primer examen parcial
    # 1. completa el código para implementar q learning,
    # 2. modifica los hiperparámetros para que el agente aprenda
    # 3. ejecuta el script para ver el comportamiento del agente
    # 4. Implementa una técnica para reducir la exploración conforme el agente aprende
    # https://gymnasium.farama.org/environments/toy_text/cliff_walking/
    env = gym.make("CliffWalking-v0", render_mode="human")

    n_episodes = 400
    episode_length = 200
    agent = QLearningAgent(env, alpha=0.25, gamma=0.9, epsilon=0.9)
    #Si aumento el alpha, es que tanto peso le doy al q-state pasado, en donde 1 es no tomarlo en cuenta y 0 es tomarlo en cuenta
    #Como es un ambiente en donde nada mas hay un reward, es conveniente bajar el epsilon progresivamente para que no vaya a explorar de oquis
    for e in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0
        for i in range(episode_length):
            # take a random action
            action = agent.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            # update agent
            agent.step(obs, action, reward, next_obs)

            if done:
                break
            ep_return += reward
            obs = next_obs
            print(agent.Q) #Este lo comento a veces para hacer los tests rapidos
            env.render()
            
        print(f"Se acabo el episodio numero: ", e)
        agent.alpha = agent.alpha *0.995 #cada vez vamos dejando de tomar en cuenta el q-state pasado, ya que vamos aprendiendo mas y mas y suponiendo que vamos en el camino correcto
        agent.epsilon = agent.epsilon * 0.8 #Segun este video: https://www.youtube.com/watch?v=mo96Nqlo1L8 , dice que conforme mas aprenda del entorno, deberia de ser mas greedy y menos explorativo

        #con esta configuracion, aprende rapido el carnalito

        # TODO: Implementa algun código para reducir la exploración del agente conforme aprende
        # puedes decidir hacerlo por episodio, por paso del tiempo, retorno promedio, etc.


        print(f"Episode {e} return: ", ep_return)
    env.close()