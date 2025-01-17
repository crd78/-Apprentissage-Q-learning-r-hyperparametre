import numpy as np

class Environment:
    def __init__(self):
        self.grid_size = 4
        self.reset()
        self.steps = 0

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        # Placer deux poisons toujours aux mêmes positions
        self.poison_positions = [(1, 1), (2, 2)]
        for pos in self.poison_positions:
            self.grid[pos] = -10
        # Placer un petit fromage à une position fixe
        self.small_cheese = (1, 2)
        self.grid[self.small_cheese] = 1
        # Placer un gros fromage à une position fixe
        self.big_cheese = (3, 3)
        self.grid[self.big_cheese] = 10
        # Position de l'agent toujours en (0, 0)
        self.agent_pos = (0, 0)
        self.grid[self.agent_pos] = 5
        self.steps = 0
        return self.grid

    def _place_items(self, count, exclude=None):
        if exclude is None:
            exclude = []
        available = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) if (i, j) not in exclude]
        selected_indices = np.random.choice(len(available), count, replace=False)
        return [available[i] for i in selected_indices]

    def step(self, action):
        new_position, valid_move = self._move(action)
        if not valid_move:
            reward = -1  # Pénalité pour action invalide
            done = False
        else:
            self.grid[self.agent_pos] = 0  # Vider l'ancienne position
            self.agent_pos = new_position
            cell_value = self.grid[self.agent_pos]
            
            # Incrémenter le compteur de pas
            self.steps += 1

            if cell_value == 0:
                reward = 0
                done = False
            elif cell_value == 1:
                reward = 1
                done = False  # Ne pas terminer l'épisode pour petit fromage
            elif cell_value == 10:
                reward = 10
                done = True
            elif cell_value == -10:
                reward = -10
                done = True

            # Terminer l'épisode si 10 pas sont effectués
            if self.steps >= 10:
                done = True

            self.grid[self.agent_pos] = 5  # Marquer la nouvelle position de l'agent
        return self.grid, reward, done

    def _move(self, action):
        x, y = self.agent_pos
        intended_x, intended_y = x, y

        if action == 0:  # Haut
            intended_x = x - 1
        elif action == 1:  # Bas
            intended_x = x + 1
        elif action == 2:  # Gauche
            intended_y = y - 1
        elif action == 3:  # Droite
            intended_y = y + 1

        # Vérifier si la nouvelle position est dans les limites
        if 0 <= intended_x < self.grid_size and 0 <= intended_y < self.grid_size:
            return (intended_x, intended_y), True
        else:
            return (x, y), False  # Position inchangée si hors limites

    def render(self):
        print(self.grid)