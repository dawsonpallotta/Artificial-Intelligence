
""" 
Wumpus World Agent
Dawson Pallotta
Artificial Intelligence
May 5 2024

This program defines a rational agent for the Wumpus World environment.
The agent uses logical inference and reinforcement learning to explore 
the grid, avoid hazards, collect the gold, and exit safely.
The agent uses percepts to determine the state of the surrounding cells. 
It maintains a map of the grid, which is updated based on percepts.
The agent then uses logical inference to decide the safest next move. 
Reinforcement learning helps the agent improve its decision-making 
over time.
"""


import random
import itertools
from typing import List, Tuple, Dict


class WWAgent:
    def __init__(self, gold_position=(0, 0), pits=[(1, 1)], wumpus=(2, 2)):
        # inittializes the agent with its initial parameter
        # uses tuples to indicate the cordinates of the Wumpus, pits, and gold
        # this makes the agent assume that the grid is 4x4 in size, and sets the start for the agent at (3,0) each simulation
        self.position = (3, 0)
        self.facing = 'right'
        self.percepts = (None, None, None, None, None)
        self.map = [[(None, None, None, None, None) for _ in range(4)] for _ in range(4)]
        self.visited = set()
        self.safe_moves = set()
        self.q_table: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.probability_threshold = 0.7
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.episode_count = 0
        self.has_gold = False
        self.exit_position = (0, 3)
        self.gold_position = gold_position
        self.pits = pits
        self.wumpus = wumpus
        self.is_alive = True

    def update(self, percepts):
        # updates the agent's state based on the given percepts
        # percepts are listed as tuples that represent the precepts of the currect room
        # forces the percepts to be a tuple list with a length of 5
        # works by updatign the map with the new percepts, marks the current position as a visited position, and checks to see if the agent has encountered the gold, a stench, or a breeze
        self.percepts = percepts
        x, y = self.position
        if 0 <= y < len(self.map) and 0 <= x < len(self.map[y]):
            self.map[y][x] = self.percepts
        self.visited.add(self.position)
        if 'glitter' in percepts and self.position == self.gold_position:
            self.has_gold = True
        if 'stench' in percepts or 'breeze' in percepts:
            self.is_alive = self.position not in self.pits and self.position != self.wumpus

    def calculate_next_position(self, move: str) -> Tuple[int, int]:
        # calculates the next position based on the move and current derection the agent is facing
        # takes the string of represented oves of move, left, right, up, and down and returns a tuple that represents the next potiion coordinates
        # forces the agent to stay in the same position if it tries to move beyond the grid boundries
        # works by determining the next position of the agent based on the direction the agent is currently facing and the move it takes
        x, y = self.position
        if move == 'move':
            if self.facing == 'up':
                return x, y - 1
            elif self.facing == 'right':
                return x + 1, y
            elif self.facing == 'down':
                return x, y + 1
            elif self.facing == 'left':
                return x - 1, y
        return x, y
    
    def move(self, move):
        self.position = self.calculate_next_position(move)
        print(f"Agent's current position: {self.position}")

    def calculate_next_direction(self, turn: str) -> str:
        # calculates the next direction the agent faces based on the turn
        # takes the string of left or right that represents the turn and retruns a string that shows the next direction the agent will face based on the turn
        # ensures that the directions to take are up, right, down, or left
        # determines the next direction the agent faces based on its current direction and the turn
        directions = ['up', 'right', 'down', 'left']
        current_index = directions.index(self.facing)
        if turn == 'left':
            next_direction = directions[(current_index - 1) % 4]
        elif turn == 'right':
            next_direction = directions[(current_index + 1) % 4]
        else:
            next_direction = self.facing
        print("Agent:", turn, "-->", self.position[1], self.position[0], next_direction)
        return next_direction

    def model_check(self, move: Tuple[int, int]) -> float:
        # calcuklates the probability of safety for a given move
        # takes the the argument of move as a tupel to represent the position the agent checks and returns a float to represent the prob that the move is safe
        # makes sure the move must be inside of the 4x4
        # uses logicol propositiosn to determine the probability of the saftey of a move
        x, y = move
        if x < 0 or x >= 4 or y < 0 or y >= 4:
            return 0.0
        percepts = self.map[y][x]
        symbols = [f'b{x}{y}']
        alpha = [f'b{x}{y}']
        models = [list(zip(symbols, vals)) for vals in itertools.product([True, False], repeat=len(symbols))]
        total_kb_true = 0
        total_alpha_true = 0
        for model in models:
            kb = self.isTrue(alpha, model)
            if kb:
                total_kb_true += 1
            if kb and self.isTrue(alpha, model):
                total_alpha_true += 1
        return total_alpha_true / total_kb_true if total_kb_true > 0 else 0.0

    def isTrue(self, prop, model):
        # evaluates if a logical proposition is true in a gioven model
        # takes the arguments of prop and model, which are a list of string to represent the logical proposition for the argument of prop, and a list of tuples to represent the model for the model argument
        # returns a boolean value that indicates whether or not the proposition is in the model
        # set a limitation that the proposition should be a valid logical expression
        # works by evaluating the logical proposition via recursion
        if isinstance(prop, str):
            return (prop, True) in model
        if isinstance(prop, list) and len(prop) == 2 and prop[0] == 'not':
            return not self.isTrue(prop[1], model)
        elif isinstance(prop, list) and len(prop) == 3:
            if prop[1] == 'and':
                return self.isTrue(prop[0], model) and self.isTrue(prop[2], model)
            elif prop[1] == 'or':
                return self.isTrue(prop[0], model) or self.isTrue(prop[2], model)
            elif prop[1] == 'implies':
                return (not self.isTrue(prop[0], model)) or self.isTrue(prop[2], model)
            elif prop[1] == 'iff':
                left = (not self.isTrue(prop[0], model)) or self.isTrue(prop[2], model)
                right = (not self.isTrue(prop[2], model)) or self.isTrue(prop[0], model)
                return left and right
        return False

    def ttc(self, symbols, model, alpha):
        # generates all possible models that satisfy a given proposition
        # takes the arguments of symbols - which are shown as a list of string to represent the proposition, model - a list of tuples to represent the currrent model, and alpha - a list of strings that represent teh logical proposition
        # returns a list of models that are satisfying the proposition
        # limits so the symbols used need to be a valid logical proposition
        # works by usign truth table to generate the models that satidy the proposition
        models = []
        if len(symbols) == 0:
            if self.isTrue(alpha, model):
                models.append(model)
        else:
            p = symbols[0]
            rest = list(symbols[1:])
            models += self.ttc(rest, model + [(p, True)], alpha)
            models += self.ttc(rest, model + [(p, False)], alpha)
        return models

    def select_move(self) -> str:
        # selects the best move for the agent based on the logical inference and reinforcement learning
        # returns a string that represent tyeh selected move
        # makes the move to slect has to be one of the following - move, left, right, grab, or exit
        # set up so that it evaluates the possible moves and selects the one with the highest prob of saftey and q value
        possible_moves = ['move', 'left', 'right', 'grab', 'exit']
        best_move = None
        best_score = -float('inf')
        safe_moves = []

        for move in possible_moves:
            next_position = self.calculate_next_position(move)
            probability_safe = self.model_check(next_position)
            state = (next_position[0], next_position[1])

            if probability_safe >= self.probability_threshold:
                safe_moves.append((move, probability_safe))
                if state not in self.q_table:
                    self.q_table[state] = {m: 0.5 for m in possible_moves}
                score = self.q_table[state].get(move, 0.5)
                if score > best_score:
                    best_score = score
                    best_move = move

        if random.random() < self.epsilon or best_move is None:
            safe_moves.extend([(m, 0.5) for m in possible_moves if m not in [s[0] for s in safe_moves]])
            best_move = random.choice(safe_moves)[0]

        return best_move

    def action(self) -> str:
        # executes the selected action and updates the agents' current location
        # returns a string showing th exected action
        # works by executing the selected move and updates the agents' state, and then updates the q-table based on the reward atained from te action
        current_state = (self.position[0], self.position[1])
        move = self.select_move()
        if move == 'move':
            self.position = self.calculate_next_position(move)
            self.is_alive = self.position not in self.pits and self.position != self.wumpus
        elif move == 'left' or move == 'right':
            self.facing = self.calculate_next_direction(move)
        elif move == 'grab' and self.position == self.gold_position:
            self.has_gold = True
        elif move == 'exit' and self.position == self.exit_position:
            pass
        if not self.is_alive:
            return 'die'
        reward = -0.1
        if move == 'grab' and self.has_gold:
            reward = 1.0
        elif 'breeze' in self.percepts or 'stench' in self.percepts:
            reward = -1.0
        next_position = self.calculate_next_position(move)
        next_state = (next_position[0], next_position[1])
        self.q_table.setdefault(current_state, {})
        self.q_table[current_state][move] = (
            self.q_table[current_state].get(move, 0.5)
            + self.learning_rate * (reward + self.discount_factor * max(self.q_table.get(next_state, {}).values(), default=0) - self.q_table[current_state].get(move, 0.5))
        )
        self.episode_count += 1
        self.epsilon = min(1.0, self.epsilon + 0.01)
        return move

    def set_learning_rate(self, learning_rate: float):
        # sets the learning rate for the agents reinforcement learning
        # limits the learning rate to be a flat value from 0 to 1
        self.learning_rate = learning_rate

    def set_discount_factor(self, discount_factor: float):
        # sets the discount rate for the agents reinforcement learnign
        #  limits the discount rate to be a flat value from 0 to 1
        self.discount_factor = discount_factor

    def set_probability_threshold(self, threshold: float):
        # sets the prob threshold for the agents reinforcement learning
        # limits the prob threshold to be a flat value from 0 to 1
        self.probability_threshold = threshold

    def set_exploration_rate(self, rate: float):
        # sets the exploration rate of the reinforcement learning for the agent
        # limits the exploration rate to be a flat value from 0 to 1
        self.epsilon = rate

    def set_epsilon(self, epsilon: float):
        # sets the epsilon value for the reinforcement learning for the agent
        # limits the epsilon value to be a flat value from 0 to 1
        self.epsilon = epsilon
        
# """ 
# Wumpus World Agent
# Dawson Pallotta
# Artificial Intelligence
# May 5 2024

# This program defines a rational agent for the Wumpus World environment.
# The agent uses logical inference and reinforcement learning to explore 
# the grid, avoid hazards, collect the gold, and exit safely.
# The agent uses percepts to determine the state of the surrounding cells. 
# It maintains a map of the grid, which is updated based on percepts.
# The agent then uses logical inference to decide the safest next move. 
# Reinforcement learning helps the agent improve its decision-making 
# over time.
# """


# import random
# import itertools
# from typing import List, Tuple, Dict


# class WWAgent:
#     def __init__(self, gold_position=(0, 0), pits=[(1, 1)], wumpus=(2, 2)):
#         # inittializes the agent with its initial parameter
#         # uses tuples to indicate the cordinates of the Wumpus, pits, and gold
#         # this makes the agent assume that the grid is 4x4 in size, and sets the start for the agent at (3,0) each simulation
#         self.position = (3, 0)
#         self.facing = 'right'
#         self.percepts = (None, None, None, None, None)
#         self.map = [[(None, None, None, None, None) for _ in range(4)] for _ in range(4)]
#         self.visited = set()
#         self.safe_moves = set()
#         self.q_table: Dict[Tuple[int, int], Dict[str, float]] = {}
#         self.probability_threshold = 0.7
#         self.learning_rate = 0.1
#         self.discount_factor = 0.9
#         self.epsilon = 0.1
#         self.episode_count = 0
#         self.has_gold = False
#         self.exit_position = (0, 3)
#         self.gold_position = gold_position
#         self.pits = pits
#         self.wumpus = wumpus
#         self.is_alive = True

#     def update(self, percepts):
#         # updates the agent's state based on the given percepts
#         # percepts are listed as tuples that represent the precepts of the currect room
#         # forces the percepts to be a tuple list with a length of 5
#         # works by updatign the map with the new percepts, marks the current position as a visited position, and checks to see if the agent has encountered the gold, a stench, or a breeze
#         self.percepts = percepts
#         x, y = self.position
#         if 0 <= y < len(self.map) and 0 <= x < len(self.map[y]):
#             self.map[y][x] = self.percepts
#         self.visited.add(self.position)
#         if 'glitter' in percepts and self.position == self.gold_position:
#             self.has_gold = True
#         if 'stench' in percepts or 'breeze' in percepts:
#             self.is_alive = self.position not in self.pits and self.position != self.wumpus

#     def calculate_next_position(self, move: str) -> Tuple[int, int]:
#         # calculates the next position based on the move and current derection the agent is facing
#         # takes the string of represented oves of move, left, right, up, and down and returns a tuple that represents the next potiion coordinates
#         # forces the agent to stay in the same position if it tries to move beyond the grid boundries
#         # works by determining the next position of the agent based on the direction the agent is currently facing and the move it takes
#         x, y = self.position
#         if move == 'move':
#             if self.facing == 'up':
#                 return x, y - 1
#             elif self.facing == 'right':
#                 return x + 1, y
#             elif self.facing == 'down':
#                 return x, y + 1
#             elif self.facing == 'left':
#                 return x - 1, y
#         return x, y

#     def calculate_next_direction(self, turn: str) -> str:
#         # calculates the next direction the agent faces based on the turn
#         # takes the string of left or right that represents the turn and retruns a string that shows the next direction the agent will face based on the turn
#         # ensures that the directions to take are up, right, down, or left
#         # determines the next direction the agent faces based on its current direction and the turn
#         directions = ['up', 'right', 'down', 'left']
#         current_index = directions.index(self.facing)
#         if turn == 'left':
#             return directions[(current_index - 1) % 4]
#         elif turn == 'right':
#             return directions[(current_index + 1) % 4]
#         return self.facing

#     def model_check(self, move: Tuple[int, int]) -> float:
#         # calcuklates the probability of safety for a given move
#         # takes the the argument of move as a tupel to represent the position the agent checks and returns a float to represent the prob that the move is safe
#         # makes sure the move must be inside of the 4x4
#         # uses logicol propositiosn to determine the probability of the saftey of a move
#         x, y = move
#         if x < 0 or x >= 4 or y < 0 or y >= 4:
#             return 0.0
#         percepts = self.map[y][x]
#         symbols = [f'b{x}{y}']
#         alpha = [f'b{x}{y}']
#         models = [list(zip(symbols, vals)) for vals in itertools.product([True, False], repeat=len(symbols))]
#         total_kb_true = 0
#         total_alpha_true = 0
#         for model in models:
#             kb = self.isTrue(alpha, model)
#             if kb:
#                 total_kb_true += 1
#             if kb and self.isTrue(alpha, model):
#                 total_alpha_true += 1
#         return total_alpha_true / total_kb_true if total_kb_true > 0 else 0.0

#     def isTrue(self, prop, model):
#         # evaluates if a logical proposition is true in a gioven model
#         # takes the arguments of prop and model, which are a list of string to represent the logical proposition for the argument of prop, and a list of tuples to represent the model for the model argument
#         # returns a boolean value that indicates whether or not the proposition is in the model
#         # set a limitation that the proposition should be a valid logical expression
#         # works by evaluating the logical proposition via recursion
#         if isinstance(prop, str):
#             return (prop, True) in model
#         if isinstance(prop, list) and len(prop) == 2 and prop[0] == 'not':
#             return not self.isTrue(prop[1], model)
#         elif isinstance(prop, list) and len(prop) == 3:
#             if prop[1] == 'and':
#                 return self.isTrue(prop[0], model) and self.isTrue(prop[2], model)
#             elif prop[1] == 'or':
#                 return self.isTrue(prop[0], model) or self.isTrue(prop[2], model)
#             elif prop[1] == 'implies':
#                 return (not self.isTrue(prop[0], model)) or self.isTrue(prop[2], model)
#             elif prop[1] == 'iff':
#                 left = (not self.isTrue(prop[0], model)) or self.isTrue(prop[2], model)
#                 right = (not self.isTrue(prop[2], model)) or self.isTrue(prop[0], model)
#                 return left and right
#         return False

#     def ttc(self, symbols, model, alpha):
#         # generates all possible models that satisfy a given proposition
#         # takes the arguments of symbols - which are shown as a list of string to represent the proposition, model - a list of tuples to represent the currrent model, and alpha - a list of strings that represent teh logical proposition
#         # returns a list of models that are satisfying the proposition
#         # limits so the symbols used need to be a valid logical proposition
#         # works by usign truth table to generate the models that satidy the proposition
#         models = []
#         if len(symbols) == 0:
#             if self.isTrue(alpha, model):
#                 models.append(model)
#         else:
#             p = symbols[0]
#             rest = list(symbols[1:])
#             models += self.ttc(rest, model + [(p, True)], alpha)
#             models += self.ttc(rest, model + [(p, False)], alpha)
#         return models

#     def select_move(self) -> str:
#         # selects the best move for the agent based on the logical inference and reinforcement learning
#         # returns a string that represent tyeh selected move
#         # makes the move to slect has to be one of the following - move, left, right, grab, or exit
#         # set up so that it evaluates the possible moves and selects the one with the highest prob of saftey and q value
#         possible_moves = ['move', 'left', 'right', 'grab', 'exit']
#         best_move = None
#         best_score = -float('inf')
#         safe_moves = []

#         for move in possible_moves:
#             next_position = self.calculate_next_position(move)
#             probability_safe = self.model_check(next_position)
#             state = (next_position[0], next_position[1])

#             if probability_safe >= self.probability_threshold:
#                 safe_moves.append((move, probability_safe))
#                 if state not in self.q_table:
#                     self.q_table[state] = {m: 0.5 for m in possible_moves}
#                 score = self.q_table[state].get(move, 0.5)
#                 if score > best_score:
#                     best_score = score
#                     best_move = move

#         if random.random() < self.epsilon or best_move is None:
#             safe_moves.extend([(m, 0.5) for m in possible_moves if m not in [s[0] for s in safe_moves]])
#             best_move = random.choice(safe_moves)[0]

#         return best_move

#     def action(self) -> str:
#         # executes the selected action and updates the agents' current location
#         # returns a string showing th exected action
#         # works by executing the selected move and updates the agents' state, and then updates the q-table based on the reward atained from te action
#         current_state = (self.position[0], self.position[1])
#         move = self.select_move()
#         if move == 'move':
#             self.position = self.calculate_next_position(move)
#             self.is_alive = self.position not in self.pits and self.position != self.wumpus
#         elif move == 'left' or move == 'right':
#             self.facing = self.calculate_next_direction(move)
#         elif move == 'grab' and self.position == self.gold_position:
#             self.has_gold = True
#         elif move == 'exit' and self.position == self.exit_position:
#             pass
#         if not self.is_alive:
#             return 'die'
#         reward = -0.1
#         if move == 'grab' and self.has_gold:
#             reward = 1.0
#         elif 'breeze' in self.percepts or 'stench' in self.percepts:
#             reward = -1.0
#         next_position = self.calculate_next_position(move)
#         next_state = (next_position[0], next_position[1])
#         self.q_table.setdefault(current_state, {})
#         self.q_table[current_state][move] = (
#             self.q_table[current_state].get(move, 0.5)
#             + self.learning_rate * (reward + self.discount_factor * max(self.q_table.get(next_state, {}).values(), default=0) - self.q_table[current_state].get(move, 0.5))
#         )
#         self.episode_count += 1
#         self.epsilon = min(1.0, self.epsilon + 0.01)
#         return move

#     def set_learning_rate(self, learning_rate: float):
#         # sets the learning rate for the agents reinforcement learning
#         # limits the learning rate to be a flat value from 0 to 1
#         self.learning_rate = learning_rate

#     def set_discount_factor(self, discount_factor: float):
#         # sets the discount rate for the agents reinforcement learnign
#         #  limits the discount rate to be a flat value from 0 to 1
#         self.discount_factor = discount_factor

#     def set_probability_threshold(self, threshold: float):
#         # sets the prob threshold for the agents reinforcement learning
#         # limits the prob threshold to be a flat value from 0 to 1
#         self.probability_threshold = threshold

#     def set_exploration_rate(self, rate: float):
#         # sets the exploration rate of the reinforcement learning for the agent
#         # limits the exploration rate to be a flat value from 0 to 1
#         self.epsilon = rate

#     def set_epsilon(self, epsilon: float):
#         # sets the epsilon value for the reinforcement learning for the agent
#         # limits the epsilon value to be a flat value from 0 to 1
#         self.epsilon = epsilon