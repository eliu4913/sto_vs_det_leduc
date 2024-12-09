# visualize_agents.py

import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rlcard
import pygame  # Import Pygame for visualization

# Set seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

# Import your custom agents
from agents.dqn_agent import DQNAgentBase
from agents.ppo_agent import PPOAgentBase

# Suppress TensorFlow warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

# Initialize the environment with seed
env = rlcard.make('leduc-holdem')
env.seed(seed_value)

# Get the number of actions and state shape
action_num = env.num_actions
state_shape = env.state_shape[0]

# Initialize the DQN agent
dqn_agent = DQNAgentBase(
    scope='dqn',
    action_num=action_num,
    state_shape=state_shape,
    mlp_layers=[64, 64],
    replay_memory_size=20000,
    replay_memory_init_size=1000,
    update_target_estimator_every=1000,
    discount_factor=0.99,
    epsilon_start=0.0,  # Set epsilon to 0 to make it fully deterministic
    epsilon_end=0.0,
    epsilon_decay_steps=0,
    batch_size=32,
    learning_rate=0.0005
)

# Load the DQN model
dqn_model_path = os.path.join(project_root, 'models', 'dqn_model_leduc.keras')
dqn_agent.load(dqn_model_path)
print(f"DQN model loaded from {dqn_model_path}")

# Initialize the PPO agent
ppo_agent = PPOAgentBase(
    scope='ppo',
    action_num=action_num,
    state_shape=state_shape,
    mlp_layers=[64, 64],
    clip_ratio=0.2,
    learning_rate=0.0003,
    value_coef=0.5,
    entropy_coef=0.01,
    update_target_every=512,
    gamma=0.99,
    lam=0.95,
    epochs=10,
    minibatch_size=64
)

# Load the PPO model weights
policy_weights_path = os.path.join(project_root, 'models', 'ppo_leduc_model_policy.weights.h5')
value_weights_path = os.path.join(project_root, 'models', 'ppo_leduc_model_value.weights.h5')

ppo_agent.policy_network.load_weights(policy_weights_path)
ppo_agent.value_network.load_weights(value_weights_path)
print(f"PPO policy weights loaded from {policy_weights_path}")
print(f"PPO value weights loaded from {value_weights_path}")

# Reset the environment and get the initial state
state, player_id = env.reset()

# Ensure we're working with player 0's state
while player_id != 0:
    action = np.random.choice(action_num)
    state, player_id = env.step(action)

# Now, 'state' is the observation for player 0
target_state = state

# Extract game state information
player_hand = target_state['raw_obs']['hand']  # Player's private card(s)
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

# Main function to render the state and save the image
def render_state_and_save():
    window.fill((34, 139, 34))  # Green background for poker table

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

    # Save the display surface as an image
    os.makedirs('evaluation_results/agent_visualization/', exist_ok=True)
    image_path = 'evaluation_results/agent_visualization/state_visualization.png'
    pygame.image.save(window, image_path)
    print(f"State visualization saved as '{image_path}'")

    # Keep the window open for a brief moment or until closed
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

# Run the visualization and save the image
render_state_and_save()

# Number of trials
num_trials = 100

# DQN Agent Actions
dqn_actions = []

for _ in range(num_trials):
    action = dqn_agent.step(target_state)
    dqn_actions.append(action)

# Count the frequency of each action
unique_actions, counts = np.unique(dqn_actions, return_counts=True)
dqn_action_freq = dict(zip(unique_actions, counts))

print("\nDQN Agent Action Frequencies over {} trials:".format(num_trials))
for action in sorted(dqn_action_freq.keys()):
    print("Action {}: {} times".format(action, dqn_action_freq[action]))

# PPO Agent Actions
ppo_actions = []

for _ in range(num_trials):
    action = ppo_agent.step(target_state)
    ppo_actions.append(action)

# Count the frequency of each action
unique_actions, counts = np.unique(ppo_actions, return_counts=True)
ppo_action_freq = dict(zip(unique_actions, counts))

print("\nPPO Agent Action Frequencies over {} trials:".format(num_trials))
for action in sorted(ppo_action_freq.keys()):
    print("Action {}: {} times".format(action, ppo_action_freq[action]))

# Set up action labels
actions = ['Fold', 'Check', 'Call', 'Raise']
action_indices = np.arange(len(actions))

# Ensure the action arrays match the number of actions
num_actions = len(actions)

# Visualize DQN Agent's Actions
dqn_action_probs = np.zeros(num_actions)
for action_idx in dqn_action_freq:
    dqn_action_probs[action_idx] = dqn_action_freq[action_idx] / num_trials

plt.figure(figsize=(6, 4))
plt.bar(action_indices, dqn_action_probs, color='blue')
plt.xticks(action_indices, actions)
plt.ylim(0, 1)
plt.title('DQN Agent Action Distribution')
plt.ylabel('Frequency')
plt.xlabel('Actions')

# Save the figure
os.makedirs('evaluation_results/agent_visualization/', exist_ok=True)
plt.savefig('evaluation_results/agent_visualization/dqn_action_distribution.png')
plt.close()
print("DQN action distribution saved as 'dqn_action_distribution.png'")

# Visualize PPO Agent's Actions
ppo_action_probs = np.zeros(num_actions)
for action_idx in ppo_action_freq:
    ppo_action_probs[action_idx] = ppo_action_freq[action_idx] / num_trials

plt.figure(figsize=(6, 4))
plt.bar(action_indices, ppo_action_probs, color='green')
plt.xticks(action_indices, actions)
plt.ylim(0, 1)
plt.title('PPO Agent Action Distribution')
plt.ylabel('Frequency')
plt.xlabel('Actions')

# Save the figure
plt.savefig('evaluation_results/agent_visualization/ppo_action_distribution.png')
plt.close()
print("PPO action distribution saved as 'ppo_action_distribution.png'")

# Visualize PPO Agent's Action Probabilities Directly from Policy Network
# Get the observation
state_obs = np.expand_dims(target_state['obs'], axis=0).astype(np.float32)

# Get the logits from the policy network
logits = ppo_agent.policy_network(state_obs)

# Compute action probabilities
ppo_action_probs_direct = tf.nn.softmax(logits).numpy()[0]

# Print the action probabilities from the policy network
print("\nPPO Agent Action Probabilities from Policy Network:")
for idx, prob in enumerate(ppo_action_probs_direct):
    print(f"Action {idx} ({actions[idx]}): {prob:.4f}")

plt.figure(figsize=(6, 4))
plt.bar(action_indices, ppo_action_probs_direct, color='orange')
plt.xticks(action_indices, actions)
plt.ylim(0, 1)
plt.title('PPO Agent Action Probabilities from Policy Network')
plt.ylabel('Probability')
plt.xlabel('Actions')

# Save the figure
plt.savefig('evaluation_results/agent_visualization/ppo_policy_action_probabilities.png')
plt.close()
print("PPO policy action probabilities saved as 'ppo_policy_action_probabilities.png'")
