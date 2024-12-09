# visualize_state_pygame.py

import sys
import os
import numpy as np
import pygame
import rlcard

# Set seed for reproducibility
seed_value = 42
np.random.seed(seed_value)

# Initialize the environment
env = rlcard.make('leduc-holdem')
env.seed(seed_value)

# Reset the environment and get the initial state
state, player_id = env.reset()

# Ensure we're working with player 0's state
while player_id != 0:
    action = np.random.choice(env.num_actions)
    state, player_id = env.step(action)

# Now, 'state' is the observation for player 0
target_state = state

# Extract game state information
# The 'hand' and 'public_card' are strings representing the cards
player_hand = target_state['raw_obs']['hand']      # Player's private card(s)
public_card = target_state['raw_obs'].get('public_card', None)  # Public card if any

# Ensure player_hand is a list
if isinstance(player_hand, str):
    player_hand = [player_hand]

# Ensure public_card is a list if it's not None
if public_card is not None and isinstance(public_card, str):
    public_card = [public_card]

# Print statements for debugging (optional)
print("Player's Hand:", player_hand)
print("Public Card:", public_card)

# Initialize Pygame
pygame.init()

# Set up the display
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Leduc Hold\'em State Visualization')

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Function to draw a card with image
def draw_card(surface, card_name, position):
    image_path = os.path.join('card_images', f'{card_name}.png')
    if os.path.exists(image_path):
        card_image = pygame.image.load(image_path)
        card_image = pygame.transform.scale(card_image, (100, 150))  # Scale image to desired size
        surface.blit(card_image, position)
    else:
        # If image not found, draw a rectangle with the card name
        pygame.draw.rect(surface, WHITE, (*position, 100, 150))  # Draw card rectangle
        font = pygame.font.SysFont(None, 36)
        text = font.render(card_name, True, BLACK)
        text_rect = text.get_rect(center=(position[0]+50, position[1]+75))
        surface.blit(text, text_rect)

# Main loop to render the state
def render_state():
    running = True
    while running:
        window.fill((34, 139, 34))  # Green background for poker table

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw player's hand
        for idx, card in enumerate(player_hand):
            position = (150 + idx*120, 400)  # Adjust position as needed
            draw_card(window, card, position)
        # Label for player's hand
        font = pygame.font.SysFont(None, 24)
        text = font.render("Player's Hand", True, WHITE)
        window.blit(text, (150, 370))

        # Draw public card if available
        if public_card:
            for idx, card in enumerate(public_card):
                position = (350 + idx*120, 200)  # Adjust position as needed
                draw_card(window, card, position)
            # Label for public card
            text = font.render("Public Card", True, WHITE)
            window.blit(text, (350, 170))
        else:
            # Indicate that there is no public card yet
            font = pygame.font.SysFont(None, 24)
            text = font.render("No Public Card Yet", True, WHITE)
            window.blit(text, (350, 200))

        # Update the display
        pygame.display.flip()

        # Limit the loop to one iteration for the script
        # Remove or adjust the following lines if you want the window to stay open
        pygame.time.wait(5000)  # Wait for 5 seconds
        running = False

    pygame.quit()
    print("State visualization displayed using Pygame")

# Run the visualization
render_state()
