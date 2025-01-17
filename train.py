import numpy as np
from Environement.env import Environment
from Agent.agent import Agent

def main():
    env = Environment()
    state_size = (env.grid_size, env.grid_size)
    action_size = 4  # 0=Haut, 1=Bas, 2=Gauche, 3=Droite
    agent = Agent(state_size, action_size)

    episodes = 100000
    rewards = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_idx = state_to_index(state, env.grid_size)
            action = agent.choose_action(state_idx)
            next_state, reward, done = env.step(action)
            next_state_idx = state_to_index(next_state, env.grid_size)
            agent.learn(state_idx, action, reward, next_state_idx, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if episode % 100 == 0:
            average_reward = np.mean(rewards[-100:])
            print(f"Episode {episode}/{episodes}, Total Reward: {total_reward}, Average Reward: {average_reward}")

    # Sauvegarder le modèle (Q-table) dans un fichier NumPy
    with open("q_table.npy", "wb") as f:
        np.save(f, agent.q_table)

    # Afficher une partie du modèle pour inspection (exemple : sous-table [0:2, 0:2, :])
    print("Extrait du modèle (Q-table) :")
    print(agent.q_table[0:2, 0:2, :])

def state_to_index(state, grid_size):
    positions = np.where(state == 5)
    if len(positions[0]) == 0 or len(positions[1]) == 0:
        raise ValueError("Agent position not found in the state.")
    x, y = positions[0][0], positions[1][0]
    return (x, y)

if __name__ == "__main__":
    main()