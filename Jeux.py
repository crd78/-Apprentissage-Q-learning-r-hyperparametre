import pygame
import numpy as np
from Environement.env import Environment
import time

# Initialisation de Pygame
pygame.init()

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)      # Pour les poisons
GREEN = (0, 255, 0)    # Pour le petit fromage
BLUE = (0, 0, 255)     # Pour l'agent
YELLOW = (255, 255, 0) # Pour le gros fromage

# Paramètres de la fenêtre
CELL_SIZE = 100
GRID_SIZE = 4
WINDOW_SIZE = CELL_SIZE * GRID_SIZE

# Création de la fenêtre
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Labyrinthe Q-Learning")

def draw_grid(env):
    screen.fill(WHITE)
    
    # Dessiner la grille
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            
            # Dessiner les éléments
            cell_value = env.grid[x, y]
            if cell_value == -10:  # Poison
                pygame.draw.circle(screen, RED, 
                                 (y * CELL_SIZE + CELL_SIZE//2, 
                                  x * CELL_SIZE + CELL_SIZE//2), 
                                 CELL_SIZE//3)
            elif cell_value == 1:  # Petit fromage
                pygame.draw.circle(screen, GREEN, 
                                 (y * CELL_SIZE + CELL_SIZE//2, 
                                  x * CELL_SIZE + CELL_SIZE//2), 
                                 CELL_SIZE//3)
            elif cell_value == 10:  # Gros fromage
                pygame.draw.circle(screen, YELLOW, 
                                 (y * CELL_SIZE + CELL_SIZE//2, 
                                  x * CELL_SIZE + CELL_SIZE//2), 
                                 CELL_SIZE//3)
            elif cell_value == 5:  # Agent
                pygame.draw.circle(screen, BLUE, 
                                 (y * CELL_SIZE + CELL_SIZE//2, 
                                  x * CELL_SIZE + CELL_SIZE//2), 
                                 CELL_SIZE//3)
    
    pygame.display.flip()

def main():
    try:
        # Charger la Q-table avec vérification
        q_table = np.load('q_table.npy')
        print("Q-table chargée:", q_table.shape)
        
        env = Environment()
        running = True
        env.reset()
        total_reward = 0
        
        # Police pour le texte
        font = pygame.font.Font(None, 36)
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
            # Afficher l'état actuel
            draw_grid(env)
            
            # Obtenir la position actuelle de l'agent
            state_idx = (env.agent_pos[0], env.agent_pos[1])
            print("Position actuelle:", state_idx)
            
            # Vérifier l'action choisie
            action = np.argmax(q_table[state_idx])
            print("Action choisie:", action)
            
            # Effectuer l'action
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            # Petit délai pour voir le mouvement
            time.sleep(0.5)
            
            if done and reward == 10:
                print("Gros fromage atteint!")
                draw_grid(env)
                time.sleep(2)  # Augmenter le délai
                
                # Afficher le score final
                screen.fill(WHITE)
                text = font.render(f"Score Final: {total_reward}", True, BLACK)
                text_rect = text.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2 - 50))
                screen.blit(text, text_rect)
                
                # Créer un bouton "Fermer"
                button = pygame.Rect(WINDOW_SIZE/2 - 50, WINDOW_SIZE/2 + 50, 100, 40)
                pygame.draw.rect(screen, BLACK, button)
                button_text = font.render("Fermer", True, WHITE)
                button_text_rect = button_text.get_rect(center=button.center)
                screen.blit(button_text, button_text_rect)
                
                pygame.display.flip()
                
                # Attendre que l'utilisateur clique sur le bouton
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            waiting = False
                            running = False
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            mouse_pos = event.pos
                            if button.collidepoint(mouse_pos):
                                waiting = False
                                running = False
            elif done:
                env.reset()
                total_reward = 0
                
    except Exception as e:
        print("Erreur:", e)
        pygame.quit()

if __name__ == "__main__":
    main()
    pygame.quit()