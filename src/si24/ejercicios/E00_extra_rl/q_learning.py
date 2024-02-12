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
        return


class QLearningAgent(RandomAgent):
    def __init__(self, env, q_table_file=None):
        super().__init__(env)
        self.Q = np.load(q_table_file)

    def act(self, observation):
        return np.argmax(self.Q[observation])

    # Update Q values using Q-learning
    def step(self, state, action, reward, next_state):
        pass
        # best_next_action = np.argmax(self.Q[next_state])
        # Actualización de Q-learning
        # self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action])
 
if __name__ == "__main__":

    env = gym.make("CliffWalking-v0", render_mode="human")
    
    q_table_file = 'q_table_episode_1000.npy'
    agent = QLearningAgent(env, q_table_file)

    obs, _ = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        step_result = env.step(action)
        next_obs = step_result[0]
        reward = step_result[1]
        done = step_result[2]
        env.render()
        obs = next_obs

    env.close()


    
    """ n_episodes = 1001
    episode_length = 200
    agent = QLearningAgent(env, alpha=0.5, gamma=0.9, epsilon=1)
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
            print(agent.Q)
            env.render()
            #agent.epsilon -= 0.000005   
            agent.epsilon *= 0.999995 
            print(agent.epsilon) 
        print(f"Episode {e} return: ", ep_return)
        # Se guarda la Q-table cada 100 episodios
        if (e) % 100 == 0:
            np.save(f'q_table_episode_{e}.npy', agent.Q)
    env.close() """
   