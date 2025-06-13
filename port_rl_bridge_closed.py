import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PortEnvironment:
    def __init__(self, close_vincent_bridge_at=12):
        # Define infrastructure capacities (containers/day)
        self.base_capacities = {
            "Vincent Thomas Bridge": 4000,
            "Gerald Desmond Bridge": 5000,
            "Long Beach Gateway": 6000,
            "Railroad": 10000
        }

        # Initially include all routes
        self.routes = ["Vincent Thomas Bridge", "Gerald Desmond Bridge", "Long Beach Gateway", "Railroad"]

        # Define when to close the Vincent Thomas Bridge (hour of the day)
        self.close_vincent_bridge_at = close_vincent_bridge_at
        self.vincent_bridge_closed = False

        # Traffic demand parameters - now completely dynamic
        self.base_demand = 15000  # Base daily demand
        self.demand_variation = 0.3  # ±30% variation

        # Time-dependent factors - much more extreme variation
        self.time_factors = {
            "morning_rush": {"start": 6, "end": 10, "factor": 2.0},
            "midday": {"start": 10, "end": 15, "factor": 1.0},
            "evening_rush": {"start": 15, "end": 19, "factor": 2.5},
            "night": {"start": 19, "end": 6, "factor": 0.75}
        }

        # More extreme capacity fluctuations
        self.capacity_variance = 0.3  # ±30% capacity variance

        # Add time tracking
        self.current_hour = 0  # Start at 0 AM
        self.episode_length = 24  # One day cycle
        self.episode_step = 0

        # Initialize queue to track waiting containers (explicit queue modeling)
        self.waiting_queue = 0
        self.max_queue_size = 5000  # Maximum queue size before severe penalties
        self.overflow_count = 0

        # Traffic congestion modeling
        self.route_congestion = {route: 0.0 for route in self.routes}
        self.congestion_recovery_rate = 0.2  # How quickly congestion recovers per hour

        # Action space: 5 discrete actions per route
        self.action_dim = 11

        # Track metrics before and after bridge closure
        self.pre_closure_metrics = {
            "throughput": 0,
            "queue_avg": 0,
            "overflow": 0
        }
        self.post_closure_metrics = {
            "throughput": 0,
            "queue_avg": 0,
            "overflow": 0
        }

        # Track impact of closure
        self.pre_closure_hours = 0
        self.post_closure_hours = 0

        # State space: [current traffic on routes, congestion on routes, queue size, RTG rate, current hour, bridge_closed]
        self.state_dim = len(self.routes) * 2 + 4

        # Current state
        self.state = None
        self.rtg_rate = None
        self.daily_capacities = None

        # Historical stats
        self.hourly_throughput = []
        self.hourly_queue = []
        self.route_usage_history = {route: [] for route in self.routes}

    def _close_vincent_bridge(self):
        """Close the Vincent Thomas Bridge and adjust environment"""
        if "Vincent Thomas Bridge" in self.routes:
            # Remove the bridge from available routes
            self.routes.remove("Vincent Thomas Bridge")

            # Flag the bridge as closed
            self.vincent_bridge_closed = True

            # Update congestion modeling
            self.route_congestion = {route: self.route_congestion.get(route, 0.0) for route in self.routes}

            # Reflect closure in route usage history
            self.route_usage_history = {route: self.route_usage_history.get(route, []) for route in self.routes}

            # print(f"Vincent Thomas Bridge closed at hour {self.current_hour}")

            # Redistribute any waiting traffic from the bridge to other routes
            # This will cause an immediate congestion spike on other routes
            # for route in self.routes:
            #     self.route_congestion[route] += 0.2  # Add 20% congestion to other routes

    def get_time_factor(self):
        """Returns traffic multiplier based on current time of day"""
        hour = self.current_hour

        if self.time_factors["morning_rush"]["start"] <= hour < self.time_factors["morning_rush"]["end"]:
            return self.time_factors["morning_rush"]["factor"]
        elif self.time_factors["midday"]["start"] <= hour < self.time_factors["midday"]["end"]:
            return self.time_factors["midday"]["factor"]
        elif self.time_factors["evening_rush"]["start"] <= hour < self.time_factors["evening_rush"]["end"]:
            return self.time_factors["evening_rush"]["factor"]
        else:
            return self.time_factors["night"]["factor"]

    def reset(self, rtg_rate=16000):
        """Reset environment to initial state with given RTG rate"""
        self.rtg_rate = rtg_rate

        # Reset time
        self.current_hour = 0  # Start at 0 AM
        self.episode_step = 0

        # Reset queue
        self.waiting_queue = 0
        self.overflow_count = 0

        # Reset bridge status
        self.vincent_bridge_closed = False

        # Reset all routes to include Vincent Thomas Bridge again
        self.routes = ["Vincent Thomas Bridge", "Gerald Desmond Bridge", "Long Beach Gateway", "Railroad"]

        # Reset congestion
        self.route_congestion = {route: 0.0 for route in self.routes}

        # Reset historical stats
        self.hourly_throughput = []
        self.hourly_queue = []
        self.route_usage_history = {route: [] for route in self.routes}

        # Reset closure metrics
        self.pre_closure_metrics = {
            "throughput": 0,
            "queue_avg": 0,
            "overflow": 0
        }
        self.post_closure_metrics = {
            "throughput": 0,
            "queue_avg": 0,
            "overflow": 0
        }
        self.pre_closure_hours = 0
        self.post_closure_hours = 0

        # Generate daily capacity variations for the entire day
        self.daily_capacities = {
            route: [capacity * np.random.uniform(1 - self.capacity_variance, 1 + self.capacity_variance)
                    for _ in range(24)]
            for route, capacity in self.base_capacities.items()
        }

        # Initialize state
        self._update_state()

        return self.state

    def _update_state(self):
        """Update the environment state"""
        # Collect route traffic and congestion information
        route_traffic = []
        route_congestion = []

        for route in self.routes:
            # Current hour's capacity
            route_capacity = self.daily_capacities[route][self.current_hour]
            # Effective capacity considering congestion
            effective_capacity = route_capacity * (1 - self.route_congestion[route])
            # Normalized by base capacity
            route_traffic.append(effective_capacity / self.base_capacities[route])
            route_congestion.append(self.route_congestion[route])

        # Normalized queue size
        norm_queue = self.waiting_queue / self.max_queue_size

        # Normalized RTG rate
        norm_rtg = self.rtg_rate / 25000  # Assuming max RTG rate is 25000

        # Time of day (normalized)
        norm_hour = self.current_hour / 24

        # Bridge closure state (binary)
        bridge_closed = 1.0 if self.vincent_bridge_closed else 0.0

        # Ensure the state vector has the right length
        # For routes no longer in self.routes, we add zeros
        all_routes = ["Vincent Thomas Bridge", "Gerald Desmond Bridge", "Long Beach Gateway", "Railroad"]
        full_route_traffic = []
        full_route_congestion = []

        for route in all_routes:
            if route in self.routes:
                idx = self.routes.index(route)
                full_route_traffic.append(route_traffic[idx])
                full_route_congestion.append(route_congestion[idx])
            else:
                # For closed routes, we use 0 for traffic and 1.0 for congestion
                full_route_traffic.append(0.0)
                full_route_congestion.append(1.0)

        # Combine all state components
        self.state = np.array(
            full_route_traffic + full_route_congestion + [norm_queue, norm_rtg, norm_hour, bridge_closed],
            dtype=np.float32)

    def step(self, actions):
        """
        Take action and return new state, reward, done

        Args:
            actions: List of discrete actions (0-4) for each route
        """
        # Convert discrete actions to allocation percentages
        # 0: 0%, 1: 25%, 2: 50%, 3: 75%, 4: 100% of capacity
        # Check for bridge closure before moving to next hour
        if not self.vincent_bridge_closed and (self.current_hour + 1) >= self.close_vincent_bridge_at:
            self._close_vincent_bridge()
        # # Check if Vincent Bridge needs to be closed
        # if not self.vincent_bridge_closed and self.current_hour == self.close_vincent_bridge_at:
        #     self._close_vincent_bridge()

        action_percentages = [a / 10 for a in actions[:len(self.routes)]]

        # Calculate time-dependent demand
        time_factor = self.get_time_factor()
        hourly_demand = (self.base_demand / 24) * time_factor * np.random.uniform(1 - self.demand_variation,
                                                                                  1 + self.demand_variation)

        # Add new arrivals to queue
        self.waiting_queue += hourly_demand

        # Process containers through each route based on action percentages
        total_processed = 0
        route_processed = {}
        route_usage = {}
        route_overutilization = {}

        for i, route in enumerate(self.routes):
            # Get current hour's capacity
            base_capacity = self.daily_capacities[route][self.current_hour] / 24  # Hourly capacity

            # Apply congestion effects to capacity
            effective_capacity = base_capacity * (1 - self.route_congestion[route])

            # Calculate how much to process based on action
            target_processing = effective_capacity * action_percentages[i]

            # Can't process more than what's in the queue
            actual_processing = min(target_processing, self.waiting_queue)

            # Update queue
            self.waiting_queue -= actual_processing
            total_processed += actual_processing
            route_processed[route] = actual_processing

            # Calculate usage percentage
            route_usage[route] = (actual_processing / base_capacity) * 100 if base_capacity > 0 else 0

            # Update route congestion based on utilization
            utilization_ratio = actual_processing / base_capacity if base_capacity > 0 else 0

            # Routes get congested when overutilized
            if utilization_ratio > 0.85:  # Congestion increases when >80% utilized
                congestion_increase = (utilization_ratio - 0.85) * 0.5  # Max 0.1 increase at 100% utilization
                self.route_congestion[route] = min(0.9, self.route_congestion[route] + congestion_increase)
                route_overutilization[route] = True
            else:
                # Routes recover from congestion when underutilized
                self.route_congestion[route] = max(0, self.route_congestion[route] - self.congestion_recovery_rate)
                route_overutilization[route] = False

            # Store usage history
            self.route_usage_history[route].append(route_usage[route])

        # Check for queue overflow
        if self.waiting_queue > self.max_queue_size:
            overflow = self.waiting_queue - self.max_queue_size
            self.waiting_queue = self.max_queue_size
            self.overflow_count += overflow
        else:
            overflow = 0

        # Store historical data
        self.hourly_throughput.append(total_processed)
        self.hourly_queue.append(self.waiting_queue)

        # Update metrics before/after closure
        if self.vincent_bridge_closed:
            self.post_closure_metrics["throughput"] += total_processed
            self.post_closure_metrics["queue_avg"] += self.waiting_queue
            self.post_closure_metrics["overflow"] += overflow
            self.post_closure_hours += 1
        else:
            self.pre_closure_metrics["throughput"] += total_processed
            self.pre_closure_metrics["queue_avg"] += self.waiting_queue
            self.pre_closure_metrics["overflow"] += overflow
            self.pre_closure_hours += 1

        # Calculate reward components
        # 1. Throughput reward - process as many containers as possible
        throughput_reward = total_processed / hourly_demand if hourly_demand > 0 else 0

        # 2. Queue management penalty - minimize waiting queue
        queue_penalty = (self.waiting_queue / self.max_queue_size) ** 2  # Quadratic penalty

        # 3. Overflow penalty - major penalty for overflow
        overflow_penalty = (overflow / hourly_demand) * 5 if hourly_demand > 0 else 0

        # 4. Balanced utilization reward - avoid congestion
        balance_reward = 0
        if sum(route_usage.values()) > 0:
            # Calculate standard deviation of route usage
            usage_values = list(route_usage.values())
            usage_std = np.std(usage_values) / np.mean(usage_values) if np.mean(usage_values) > 0 else 0
            balance_reward = 1 / (1 + usage_std)

        # 5. Congestion penalty
        congestion_penalty = sum(self.route_congestion.values()) / len(self.routes)

        # Combine all reward components
        reward = (0.4 * throughput_reward) + (0.3 * balance_reward) - (0.2 * queue_penalty) - (
                0.1 * congestion_penalty) - overflow_penalty

        # # Check if Vincent Bridge needs to be closed
        # if not self.vincent_bridge_closed and self.current_hour == self.close_vincent_bridge_at:
        #     self._close_vincent_bridge()

        # Update time for next state
        self.current_hour = (self.current_hour + 1) % 24
        self.episode_step += 1

        # Update state
        self._update_state()

        # Episode ends after episode_length steps
        done = self.episode_step >= self.episode_length

        # Additional info for logging
        info = {
            "processed": route_processed,
            "usage_percentages": route_usage,
            "total_processed": total_processed,
            "queue_size": self.waiting_queue,
            "overflow": overflow,
            "congestion": self.route_congestion.copy(),
            "overutilized": route_overutilization,
            "throughput_reward": throughput_reward,
            "balance_reward": balance_reward,
            "queue_penalty": queue_penalty,
            "congestion_penalty": congestion_penalty,
            "overflow_penalty": overflow_penalty,
            "time_factor": time_factor,
            "hourly_demand": hourly_demand,
            "bridge_closed": self.vincent_bridge_closed
        }

        # Calculate final metrics if episode is done
        if done:
            if self.pre_closure_hours > 0:
                self.pre_closure_metrics["queue_avg"] /= self.pre_closure_hours
            if self.post_closure_hours > 0:
                self.post_closure_metrics["queue_avg"] /= self.post_closure_hours

            # Add to info
            info["pre_closure_metrics"] = self.pre_closure_metrics
            info["post_closure_metrics"] = self.post_closure_metrics

        return self.state, reward, done, info


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), \
            np.array(self.probs), np.array(self.vals), \
            np.array(self.rewards), np.array(self.dones), \
            batches


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, fc1_dims=128, fc2_dims=128):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.policy = nn.Linear(fc2_dims, n_actions)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.policy.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        policy = F.softmax(self.policy(x), dim=-1)

        return policy


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=128, fc2_dims=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.value = nn.Linear(fc2_dims, 1)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.value.weight)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)

        return value


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0001, beta=0.0001,
                 gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10, entropy_coef=0.01):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

        self.actor = ActorNetwork(n_actions, input_dims).to(device)
        self.critic = CriticNetwork(input_dims).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=beta)

        self.memory = PPOMemory(batch_size)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self, name=""):
        print(f'... saving models for {name} ...')
        torch.save(self.actor.state_dict(), f'results_1/actor_{name}.pth')
        torch.save(self.critic.state_dict(), f'results_1/critic_{name}.pth')

    def load_models(self, name=""):
        print(f'... loading models for {name} ...')
        self.actor.load_state_dict(torch.load(f'actor_{name}.pth'))
        self.critic.load_state_dict(torch.load(f'critic_{name}.pth'))

    def choose_action(self, observation):
        state = torch.FloatTensor(observation).to(device)

        probs = self.actor(state)
        value = self.critic(state)

        dist = Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), probs.detach().cpu().numpy(), value.item(), log_prob.detach().item()

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # Calculate advantages
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.FloatTensor(advantage).to(device)

            # Update actor and critic networks
            for batch in batches:
                states = torch.FloatTensor(state_arr[batch]).to(device)
                old_probs = torch.FloatTensor(old_prob_arr[batch]).to(device)
                actions = torch.LongTensor(action_arr[batch]).to(device)

                # Get new probabilities and values
                probs = self.actor(states)
                values = self.critic(states).squeeze()

                # Create distribution and get new log probs
                dist = Categorical(probs)
                new_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Calculate critic loss
                critic_loss = F.mse_loss(values, torch.FloatTensor(reward_arr[batch]).to(device).squeeze())

                # Calculate actor loss (PPO clipped objective)
                prob_ratio = torch.exp(new_probs - torch.log(old_probs[range(len(batch)), action_arr[batch]] + 1e-10))
                weighted_probs = advantage[batch] * prob_ratio
                clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                weighted_clipped_probs = clipped_probs * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Add entropy bonus to encourage exploration
                actor_loss = actor_loss - self.entropy_coef * entropy

                # Update networks
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        self.memory.clear_memory()


class MultiAgentPPO:
    def __init__(self, env, n_agents=4, max_episodes=1000):
        self.env = env
        self.n_agents = n_agents  # One agent per route, including Vincent Thomas Bridge
        self.max_episodes = max_episodes

        # Create agents for each route
        self.agents = [
            Agent(
                n_actions=env.action_dim,  # 5 discrete actions
                input_dims=env.state_dim,
                entropy_coef=0.05  # Higher entropy coefficient for more exploration
            )
            for _ in range(n_agents)
        ]

        # Training history
        self.episode_rewards = []
        self.avg_queue_sizes = []
        self.processed_containers = []
        self.overflow_counts = []
        self.policy_entropy = []

        # Bridge closure metrics
        self.pre_closure_metrics = []
        self.post_closure_metrics = []

    def train(self, rtg_rates=None, render_freq=100, closure_hours=None):
        if rtg_rates is None:
            rtg_rates = [16000]  # Default RTG rate

        if closure_hours is None:
            closure_hours = [12, 14, 16]  # Different hours to test bridge closure

        # Initialize fixed baseline policy for comparison
        fixed_actions = [2, 2, 2, 2]  # 50% allocation for all routes

        # Track baseline performance
        baseline_rewards = []

        for episode in range(self.max_episodes):
            # Choose a random RTG rate for this episode
            rtg_rate = random.choice(rtg_rates)

            # Choose a random closure hour
            close_at = random.choice(closure_hours)
            self.env.close_vincent_bridge_at = close_at

            observation = self.env.reset(rtg_rate)

            done = False
            episode_score = 0
            episode_queue = []
            episode_throughput = 0

            # Use pure random exploration for first 200 episodes
            use_random = episode < 200

            while not done:
                # Get actions from all agents - only use actions for active routes
                multi_actions = []
                multi_probs = []
                multi_values = []
                multi_log_probs = []

                # We need to keep track of our active routes indices
                all_routes = ["Vincent Thomas Bridge", "Gerald Desmond Bridge", "Long Beach Gateway", "Railroad"]

                for route_index, route in enumerate(all_routes):
                    # Skip actions for closed routes
                    if route not in self.env.routes:
                        continue

                    agent = self.agents[route_index]

                    if use_random:
                        # Random exploration in early episodes
                        action = np.random.randint(0, self.env.action_dim)
                        probs = np.zeros(self.env.action_dim)
                        probs[action] = 1.0
                        value = 0.0
                        log_prob = 0.0
                    else:
                        action, probs, value, log_prob = agent.choose_action(observation)

                    multi_actions.append(action)
                    multi_probs.append(probs)
                    multi_values.append(value)
                    multi_log_probs.append(log_prob)

                # Take action in environment
                observation_, reward, done, info = self.env.step(multi_actions)
                episode_score += reward
                episode_queue.append(info['queue_size'])
                episode_throughput += info['total_processed']

                # Store transitions for each agent (skip if using random actions)
                if not use_random:
                    # We need to handle the bridge being removed
                    active_route_idx = 0
                    for route_index, route in enumerate(all_routes):
                        if route not in self.env.routes:
                            continue

                        agent = self.agents[route_index]
                        agent.store_transition(observation, multi_actions[active_route_idx],
                                               multi_probs[active_route_idx],
                                               multi_values[active_route_idx], reward, done)
                        active_route_idx += 1

                observation = observation_

            # Run a baseline episode with fixed policy every 10 episodes
            if episode % 10 == 0:
                observation = self.env.reset(rtg_rate)
                done = False
                baseline_score = 0

                while not done:
                    # Only use actions for active routes
                    active_actions = []
                    for route in all_routes:
                        if route in self.env.routes:
                            idx = all_routes.index(route)
                            active_actions.append(fixed_actions[idx])

                    observation_, reward, done, info = self.env.step(active_actions)
                    baseline_score += reward
                    observation = observation_

                baseline_rewards.append(baseline_score)

            # Learn at the end of each episode (skip if using random actions)
            if not use_random:
                for agent in self.agents:
                    agent.learn()

            # Store results for this episode
            self.episode_rewards.append(episode_score)
            self.avg_queue_sizes.append(np.mean(episode_queue))
            self.processed_containers.append(episode_throughput)
            self.overflow_counts.append(self.env.overflow_count)

            # Store closure metrics
            if done and info.get("pre_closure_metrics") and info.get("post_closure_metrics"):
                self.pre_closure_metrics.append(info["pre_closure_metrics"])
                self.post_closure_metrics.append(info["post_closure_metrics"])

            # Calculate policy entropy to track exploration
            if episode % 20 == 0 and not use_random:
                entropy = 0
                for agent in self.agents:
                    # Sample several states
                    states = [self.env.reset(rtg_rate) for _ in range(10)]
                    probs_list = []
                    for state in states:
                        probs = agent.actor(torch.FloatTensor(state).to(device)).detach().cpu().numpy()
                        probs_list.append(probs)

                    # Calculate entropy
                    mean_probs = np.mean(probs_list, axis=0)
                    ent = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
                    entropy += ent

                avg_entropy = entropy / self.n_agents
                self.policy_entropy.append(avg_entropy)

            # Log progress
            if episode % render_freq == 0:
                avg_score = np.mean(self.episode_rewards[-100:]) if episode >= 100 else np.mean(self.episode_rewards)
                avg_queue = np.mean(self.avg_queue_sizes[-100:]) if episode >= 100 else np.mean(self.avg_queue_sizes)
                avg_throughput = np.mean(self.processed_containers[-100:]) if episode >= 100 else np.mean(
                    self.processed_containers)

                baseline_avg = np.mean(baseline_rewards[-10:]) if baseline_rewards else 0

                if self.policy_entropy:
                    current_entropy = self.policy_entropy[-1]
                    print(f'Episode: {episode}, Score: {episode_score:.3f}, Baseline: {baseline_avg:.3f}, '
                          f'Avg Score: {avg_score:.3f}, Avg Queue: {avg_queue:.1f}, '
                          f'Avg Throughput: {avg_throughput:.1f}, Policy Entropy: {current_entropy:.3f}')
                else:
                    print(f'Episode: {episode}, Score: {episode_score:.3f}, Baseline: {baseline_avg:.3f}, '
                          f'Avg Score: {avg_score:.3f}, Avg Queue: {avg_queue:.1f}, '
                          f'Avg Throughput: {avg_throughput:.1f}')

                # Save models periodically
                if episode % 1000 == 0 and episode > 0:
                    for i, agent in enumerate(self.agents):
                        agent.save_models(f"route_{i}_ep_{episode}")

    def test(self, rtg_rates, bridge_closure_hours=None, n_test_episodes=3):
        """Test the trained agents on various RTG rates"""
        results = []

        if bridge_closure_hours is None:
            bridge_closure_hours = [12]  # Default closure hour

        for rtg_rate in rtg_rates:
            rtg_results = []

            for close_hour in bridge_closure_hours:
                self.env.close_vincent_bridge_at = close_hour
                print(f"Testing with RTG rate: {rtg_rate}, Bridge closure at: {close_hour}:00")

                episode_results = []
                for _ in range(n_test_episodes):
                    observation = self.env.reset(rtg_rate)

                    hourly_data = []

                    done = False
                    while not done:
                        # Get actions from all agents - only use actions for active routes
                        multi_actions = []

                        all_routes = ["Vincent Thomas Bridge", "Gerald Desmond Bridge", "Long Beach Gateway",
                                      "Railroad"]

                        for route_index, route in enumerate(all_routes):
                            # Skip actions for closed routes
                            if route not in self.env.routes:
                                continue

                            agent = self.agents[route_index]
                            action, _, _, _ = agent.choose_action(observation)
                            multi_actions.append(action)

                        # Take action in environment
                        observation_, reward, done, info = self.env.step(multi_actions)

                        # Store hourly data
                        hourly_data.append({
                            "hour": self.env.current_hour,
                            "processed": info['processed'].copy(),
                            "usage": info['usage_percentages'].copy(),
                            "queue": info['queue_size'],
                            "congestion": info['congestion'].copy(),
                            "demand": info['hourly_demand'],
                            "reward": reward,
                            "bridge_closed": info['bridge_closed']
                        })

                        observation = observation_

                    # Get closure metrics
                    closure_info = {}
                    if "pre_closure_metrics" in info and "post_closure_metrics" in info:
                        closure_info = {
                            "pre_closure": info["pre_closure_metrics"],
                            "post_closure": info["post_closure_metrics"]
                        }

                    # Process hourly data for this episode
                    processed_by_hour = {}
                    usage_by_hour = {}
                    queue_by_hour = {}
                    congestion_by_hour = {}
                    bridge_status_by_hour = {}

                    for data in hourly_data:
                        hour = data["hour"]
                        if hour not in processed_by_hour:
                            processed_by_hour[hour] = {route: [] for route in all_routes}
                            usage_by_hour[hour] = {route: [] for route in all_routes}
                            queue_by_hour[hour] = []
                            congestion_by_hour[hour] = {route: [] for route in all_routes}
                            bridge_status_by_hour[hour] = []

                        for route in all_routes:
                            # Add a check if the route exists in the processed/congestion data
                            if route in data["processed"]:
                                processed_by_hour[hour][route].append(data["processed"][route])
                            else:
                                processed_by_hour[hour][route].append(0)  # Add zero if route not present

                            if route in data["usage"]:
                                usage_by_hour[hour][route].append(data["usage"][route])
                            else:
                                usage_by_hour[hour][route].append(0)  # Add zero if route not present

                            if route in data["congestion"]:
                                congestion_by_hour[hour][route].append(data["congestion"][route])
                            else:
                                congestion_by_hour[hour][route].append(1.0)  # Add full congestion if route not present

                        queue_by_hour[hour].append(data["queue"])
                        bridge_status_by_hour[hour].append(data["bridge_closed"])

                    # Average the hourly metrics
                    avg_processed_by_hour = {}
                    avg_usage_by_hour = {}
                    avg_queue_by_hour = {}
                    avg_congestion_by_hour = {}
                    bridge_closed_by_hour = {}

                    for hour in processed_by_hour:
                        avg_processed_by_hour[hour] = {
                            route: np.mean(processed_by_hour[hour][route]) if processed_by_hour[hour][route] else 0
                            for route in all_routes
                        }

                        avg_usage_by_hour[hour] = {
                            route: np.mean(usage_by_hour[hour][route]) if usage_by_hour[hour][route] else 0
                            for route in all_routes
                        }

                        avg_queue_by_hour[hour] = np.mean(queue_by_hour[hour])

                        avg_congestion_by_hour[hour] = {
                            route: np.mean(congestion_by_hour[hour][route]) if congestion_by_hour[hour][route] else 0
                            for route in all_routes
                        }

                        bridge_closed_by_hour[hour] = any(bridge_status_by_hour[hour])

                    episode_results.append({
                        "processed_by_hour": avg_processed_by_hour,
                        "usage_by_hour": avg_usage_by_hour,
                        "queue_by_hour": avg_queue_by_hour,
                        "congestion_by_hour": avg_congestion_by_hour,
                        "bridge_closed_by_hour": bridge_closed_by_hour,
                        "overflow_count": self.env.overflow_count,
                        "total_throughput": sum(self.env.hourly_throughput),
                        "closure_metrics": closure_info
                    })

                # Average across all test episodes for this RTG rate and closure hour
                avg_processed = {}
                avg_usage = {}
                avg_queue = {}
                avg_congestion = {}
                bridge_closed = {}
                avg_overflow = 0
                avg_throughput = 0

                # Also track closure impact metrics
                pre_closure_throughput = 0
                post_closure_throughput = 0
                pre_closure_queue = 0
                post_closure_queue = 0
                pre_closure_overflow = 0
                post_closure_overflow = 0

                pre_closure_samples = 0
                post_closure_samples = 0

                for result in episode_results:
                    for hour in range(24):
                        if hour not in avg_processed:
                            avg_processed[hour] = {route: 0 for route in all_routes}
                            avg_usage[hour] = {route: 0 for route in all_routes}
                            avg_queue[hour] = 0
                            avg_congestion[hour] = {route: 0 for route in all_routes}
                            bridge_closed[hour] = False

                        if hour in result["processed_by_hour"]:
                            for route in all_routes:
                                avg_processed[hour][route] += result["processed_by_hour"][hour].get(route,
                                                                                                    0) / n_test_episodes
                                avg_usage[hour][route] += result["usage_by_hour"][hour].get(route, 0) / n_test_episodes
                                avg_congestion[hour][route] += result["congestion_by_hour"][hour].get(route,
                                                                                                      0) / n_test_episodes

                            avg_queue[hour] += result["queue_by_hour"][hour] / n_test_episodes
                            bridge_closed[hour] = bridge_closed[hour] or result["bridge_closed_by_hour"].get(hour,
                                                                                                             False)

                    avg_overflow += result["overflow_count"] / n_test_episodes
                    avg_throughput += result["total_throughput"] / n_test_episodes

                    # Process closure metrics
                    if "closure_metrics" in result and "pre_closure" in result["closure_metrics"] and "post_closure" in \
                            result["closure_metrics"]:
                        pre = result["closure_metrics"]["pre_closure"]
                        post = result["closure_metrics"]["post_closure"]

                        if pre:
                            pre_closure_throughput += pre.get("throughput", 0)
                            pre_closure_queue += pre.get("queue_avg", 0)
                            pre_closure_overflow += pre.get("overflow", 0)
                            pre_closure_samples += 1

                        if post:
                            post_closure_throughput += post.get("throughput", 0)
                            post_closure_queue += post.get("queue_avg", 0)
                            post_closure_overflow += post.get("overflow", 0)
                            post_closure_samples += 1

                # Calculate final closure impact metrics
                closure_impact = {}
                if pre_closure_samples > 0 and post_closure_samples > 0:
                    # Average the metrics
                    pre_closure_throughput /= pre_closure_samples
                    post_closure_throughput /= post_closure_samples
                    pre_closure_queue /= pre_closure_samples
                    post_closure_queue /= post_closure_samples
                    pre_closure_overflow /= pre_closure_samples
                    post_closure_overflow /= post_closure_samples

                    # Calculate percentage changes
                    throughput_change = ((
                                                     post_closure_throughput - pre_closure_throughput) / pre_closure_throughput) * 100 if pre_closure_throughput > 0 else 0
                    queue_change = ((
                                                post_closure_queue - pre_closure_queue) / pre_closure_queue) * 100 if pre_closure_queue > 0 else 0
                    overflow_change = ((
                                                   post_closure_overflow - pre_closure_overflow) / pre_closure_overflow) * 100 if pre_closure_overflow > 0 else 0

                    closure_impact = {
                        "throughput_change_pct": throughput_change,
                        "queue_change_pct": queue_change,
                        "overflow_change_pct": overflow_change,
                        "pre_closure": {
                            "throughput": pre_closure_throughput,
                            "queue_avg": pre_closure_queue,
                            "overflow": pre_closure_overflow
                        },
                        "post_closure": {
                            "throughput": post_closure_throughput,
                            "queue_avg": post_closure_queue,
                            "overflow": post_closure_overflow
                        }
                    }

                # Store combined results for this RTG rate and closure hour
                rtg_results.append({
                    "RTG_Rate": rtg_rate,
                    "Bridge_Closure_Hour": close_hour,
                    "Processed_By_Hour": avg_processed,
                    "Usage_By_Hour": avg_usage,
                    "Queue_By_Hour": avg_queue,
                    "Congestion_By_Hour": avg_congestion,
                    "Bridge_Closed_By_Hour": bridge_closed,
                    "Overflow_Count": avg_overflow,
                    "Total_Throughput": avg_throughput,
                    "Closure_Impact": closure_impact
                })

            results.extend(rtg_results)

        return results

    def visualize_results(self, results):
        """Create visualizations of the test results with bridge closure analysis"""
        # Group results by RTG rate
        rtg_results = {}
        for result in results:
            rtg_rate = result["RTG_Rate"]
            if rtg_rate not in rtg_results:
                rtg_results[rtg_rate] = []
            rtg_results[rtg_rate].append(result)

        # Plot throughput vs RTG rate for different closure hours
        plt.figure(figsize=(12, 8))
        closure_hours = sorted(list(set([r["Bridge_Closure_Hour"] for r in results])))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

        for i, hour in enumerate(closure_hours):
            rtg_rates = []
            throughputs = []
            for result in results:
                if result["Bridge_Closure_Hour"] == hour:
                    rtg_rates.append(result["RTG_Rate"])
                    throughputs.append(result["Total_Throughput"])

            marker = markers[i % len(markers)]
            plt.plot(rtg_rates, throughputs, marker=marker, linestyle='-', linewidth=2,
                     label=f'Bridge closure at {hour}:00')

        plt.xlabel('RTG Offloading Rate (containers/day)')
        plt.ylabel('Total Throughput (containers/day)')
        plt.title('Total Container Throughput vs RTG Rate with Different Bridge Closure Times')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('results_1/rl_throughput_by_closure.png')

        # Plot queue size vs RTG rate for different closure hours
        plt.figure(figsize=(12, 8))

        for i, hour in enumerate(closure_hours):
            rtg_rates = []
            avg_queues = []
            for result in results:
                if result["Bridge_Closure_Hour"] == hour:
                    rtg_rates.append(result["RTG_Rate"])
                    # Average queue size across all hours
                    avg_queue = np.mean([q for q in result["Queue_By_Hour"].values()])
                    avg_queues.append(avg_queue)

            marker = markers[i % len(markers)]
            plt.plot(rtg_rates, avg_queues, marker=marker, linestyle='-', linewidth=2,
                     label=f'Bridge closure at {hour}:00')

        plt.xlabel('RTG Offloading Rate (containers/day)')
        plt.ylabel('Average Queue Size (containers)')
        plt.title('Average Queue Size vs RTG Rate with Different Bridge Closure Times')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('results_1/rl_queue_by_closure.png')

        # Plot closure impact analysis
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)

        # Plot throughput change after closure
        for rtg_rate in sorted(rtg_results.keys()):
            closure_hours = []
            throughput_changes = []

            for result in rtg_results[rtg_rate]:
                if "Closure_Impact" in result and "throughput_change_pct" in result["Closure_Impact"]:
                    closure_hours.append(result["Bridge_Closure_Hour"])
                    throughput_changes.append(result["Closure_Impact"]["throughput_change_pct"])

            if closure_hours:  # Only plot if we have data
                plt.plot(closure_hours, throughput_changes, 'o-', linewidth=2,
                         label=f'RTG Rate: {rtg_rate}')

        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Bridge Closure Hour')
        plt.ylabel('Throughput Change (%)')
        plt.title('Impact of Bridge Closure on Throughput')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot queue change after closure
        plt.subplot(3, 1, 2)

        for rtg_rate in sorted(rtg_results.keys()):
            closure_hours = []
            queue_changes = []

            for result in rtg_results[rtg_rate]:
                if "Closure_Impact" in result and "queue_change_pct" in result["Closure_Impact"]:
                    closure_hours.append(result["Bridge_Closure_Hour"])
                    queue_changes.append(result["Closure_Impact"]["queue_change_pct"])

            if closure_hours:  # Only plot if we have data
                plt.plot(closure_hours, queue_changes, 'o-', linewidth=2,
                         label=f'RTG Rate: {rtg_rate}')

        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Bridge Closure Hour')
        plt.ylabel('Queue Size Change (%)')
        plt.title('Impact of Bridge Closure on Queue Size')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot overflow change after closure
        plt.subplot(3, 1, 3)

        for rtg_rate in sorted(rtg_results.keys()):
            closure_hours = []
            overflow_changes = []

            for result in rtg_results[rtg_rate]:
                if "Closure_Impact" in result and "overflow_change_pct" in result["Closure_Impact"]:
                    closure_hours.append(result["Bridge_Closure_Hour"])
                    overflow_changes.append(result["Closure_Impact"]["overflow_change_pct"])

            if closure_hours:  # Only plot if we have data
                plt.plot(closure_hours, overflow_changes, 'o-', linewidth=2,
                         label=f'RTG Rate: {rtg_rate}')

        plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Bridge Closure Hour')
        plt.ylabel('Overflow Change (%)')
        plt.title('Impact of Bridge Closure on Container Overflow')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig('results_1/rl_closure_impact.png')

        # Select a specific RTG rate and closure hour for hourly analysis
        mid_idx = len(results) // 2
        mid_result = results[mid_idx]
        rtg_rate = mid_result["RTG_Rate"]
        close_hour = mid_result["Bridge_Closure_Hour"]

        # Create visualization of hourly processing patterns
        plt.figure(figsize=(14, 8))
        # hours = list(range(24))
        hours = list(range(1, 25))  # 1–24 (exclude hour 0)
        all_routes = ["Vincent Thomas Bridge", "Gerald Desmond Bridge", "Long Beach Gateway", "Railroad"]

        # for route in all_routes:
        #     route_processed = [mid_result["Processed_By_Hour"][hour].get(route, 0) for hour in hours]
        #     plt.plot(hours, route_processed, 'o-', linewidth=2, label=f'{route}')

        for route in all_routes:
            route_processed = [mid_result["Processed_By_Hour"][hour].get(route, 0) for hour in range(1, 24)]
            route_processed.append(mid_result["Processed_By_Hour"][0].get(route, 0))  # hour 24 = hour 0
            plt.plot(hours, route_processed, 'o-', linewidth=2, label=f'{route}')

        # Add vertical line at bridge closure time
        plt.axvline(x=close_hour, color='r', linestyle='--', alpha=0.7,
                    label=f'Bridge Closure at {close_hour}:00')

        plt.xlabel('Hour of Day')
        plt.ylabel('Containers Processed per Hour')
        plt.title(f'Hourly Container Processing by Route (RTG Rate: {rtg_rate}, Bridge Closure: {close_hour}:00)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(hours)
        plt.savefig('results_1/rl_hourly_processing_with_closure.png')

        # Create visualization of hourly route utilization
        plt.figure(figsize=(14, 8))

        # for route in all_routes:
        #     route_usage = [mid_result["Usage_By_Hour"][hour].get(route, 0) for hour in hours]
        #     plt.plot(hours, route_usage, 'o-', linewidth=2, label=f'{route}')

        for route in all_routes:
            route_usage = [mid_result["Usage_By_Hour"][hour].get(route, 0) for hour in range(1, 24)]
            route_usage.append(mid_result["Usage_By_Hour"][0].get(route, 0))  # hour 24 = hour 0
            plt.plot(hours, route_usage, 'o-', linewidth=2, label=f'{route}')

        # Add vertical line at bridge closure time
        plt.axvline(x=close_hour, color='r', linestyle='--', alpha=0.7,
                    label=f'Bridge Closure at {close_hour}:00')

        plt.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='100% Capacity')
        plt.axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='85% Capacity (Congestion Threshold)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Route Usage (%)')
        plt.title(f'Hourly Route Utilization (RTG Rate: {rtg_rate}, Bridge Closure: {close_hour}:00)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(hours)
        plt.savefig('results_1/rl_hourly_usage_with_closure.png')

        # Create visualization of hourly queue size
        plt.figure(figsize=(14, 8))
        # queue_sizes = [mid_result["Queue_By_Hour"][hour] for hour in hours]
        queue_sizes = [mid_result["Queue_By_Hour"][hour] for hour in range(1, 24)]
        queue_sizes.append(mid_result["Queue_By_Hour"][0])  # hour 24 = hour 0
        plt.plot(hours, queue_sizes, 'o-', linewidth=2, color='purple')

        # Add vertical line at bridge closure time
        plt.axvline(x=close_hour, color='r', linestyle='--', alpha=0.7,
                    label=f'Bridge Closure at {close_hour}:00')

        plt.axhline(y=self.env.max_queue_size, color='r', linestyle='--', alpha=0.7,
                    label=f'Max Queue Size ({self.env.max_queue_size})')

        plt.xlabel('Hour of Day')
        plt.ylabel('Queue Size (containers)')
        plt.title(f'Hourly Queue Size (RTG Rate: {rtg_rate}, Bridge Closure: {close_hour}:00)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(hours)
        plt.savefig('results_1/rl_hourly_queue_with_closure.png')

        # Create visualization of hourly congestion levels
        plt.figure(figsize=(14, 8))

        for route in all_routes:
            # route_congestion = [mid_result["Congestion_By_Hour"][hour].get(route, 0) * 100 for hour in hours]
            route_congestion = [mid_result["Congestion_By_Hour"][hour].get(route, 0) * 100 for hour in range(1, 24)]
            route_congestion.append(mid_result["Congestion_By_Hour"][0].get(route, 0) * 100)  # hour 24 = hour 0
            plt.plot(hours, route_congestion, 'o-', linewidth=2, label=f'{route}')

        # Add vertical line at bridge closure time
        plt.axvline(x=close_hour, color='r', linestyle='--', alpha=0.7,
                    label=f'Bridge Closure at {close_hour}:00')

        plt.xlabel('Hour of Day')
        plt.ylabel('Route Congestion (%)')
        plt.title(f'Hourly Route Congestion (RTG Rate: {rtg_rate}, Bridge Closure: {close_hour}:00)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(hours)
        plt.savefig('results_1/rl_hourly_congestion_with_closure.png')

    def plot_training_progress(self):
        """Plot training progress over episodes"""
        episodes = range(len(self.episode_rewards))

        # Plot episode rewards
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, self.episode_rewards, linewidth=1, color='blue', alpha=0.5)
        # Add rolling average
        if len(episodes) > 100:
            rolling_avg = pd.Series(self.episode_rewards).rolling(window=100).mean()
            plt.plot(episodes, rolling_avg, linewidth=2, color='red', label='100-episode moving average')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress - Episode Rewards')
        plt.grid(True, alpha=0.3)
        if len(episodes) > 100:
            plt.legend()
        plt.savefig('results_1/training_rewards.png')

        # Plot queue sizes
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, self.avg_queue_sizes, linewidth=1, color='purple', alpha=0.5)
        # Add rolling average
        if len(episodes) > 100:
            rolling_avg = pd.Series(self.avg_queue_sizes).rolling(window=100).mean()
            plt.plot(episodes, rolling_avg, linewidth=2, color='red', label='100-episode moving average')
        plt.xlabel('Episode')
        plt.ylabel('Average Queue Size')
        plt.title('Training Progress - Queue Management')
        plt.grid(True, alpha=0.3)
        if len(episodes) > 100:
            plt.legend()
        plt.savefig('results_1/training_queue.png')

        # Plot throughput
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, self.processed_containers, linewidth=1, color='green', alpha=0.5)
        # Add rolling average
        if len(episodes) > 100:
            rolling_avg = pd.Series(self.processed_containers).rolling(window=100).mean()
            plt.plot(episodes, rolling_avg, linewidth=2, color='red', label='100-episode moving average')
        plt.xlabel('Episode')
        plt.ylabel('Containers Processed')
        plt.title('Training Progress - Throughput')
        plt.grid(True, alpha=0.3)
        if len(episodes) > 100:
            plt.legend()
        plt.savefig('results_1/training_throughput.png')

        # Plot overflow counts
        plt.figure(figsize=(12, 6))
        plt.plot(episodes, self.overflow_counts, linewidth=1, color='red', alpha=0.5)
        # Add rolling average
        if len(episodes) > 100:
            rolling_avg = pd.Series(self.overflow_counts).rolling(window=100).mean()
            plt.plot(episodes, rolling_avg, linewidth=2, color='orange', label='100-episode moving average')
        plt.xlabel('Episode')
        plt.ylabel('Overflow Count')
        plt.title('Training Progress - Overflow Management')
        plt.grid(True, alpha=0.3)
        if len(episodes) > 100:
            plt.legend()
        plt.savefig('results_1/training_overflow.png')

        # Plot policy entropy if available
        if self.policy_entropy:
            entropy_episodes = range(0, len(episodes), 20)[:len(self.policy_entropy)]
            plt.figure(figsize=(12, 6))
            plt.plot(entropy_episodes, self.policy_entropy, linewidth=2, color='purple')
            plt.xlabel('Episode')
            plt.ylabel('Policy Entropy')
            plt.title('Policy Entropy During Training (Lower = More Exploitation)')
            plt.grid(True, alpha=0.3)
            plt.savefig('results_1/policy_entropy.png')

        # Plot closure impact if we have enough data
        if len(self.pre_closure_metrics) > 10 and len(self.post_closure_metrics) > 10:
            plt.figure(figsize=(15, 15))

            # Throughput change
            plt.subplot(3, 1, 1)
            episodes = range(len(self.pre_closure_metrics))

            pre_throughputs = [m.get("throughput", 0) for m in self.pre_closure_metrics]
            post_throughputs = [m.get("throughput", 0) for m in self.post_closure_metrics]

            # Calculate percentage change
            throughput_changes = []
            for pre, post in zip(pre_throughputs, post_throughputs):
                if pre > 0:
                    change = ((post - pre) / pre) * 100
                    throughput_changes.append(change)
                else:
                    throughput_changes.append(0)

            plt.plot(episodes, throughput_changes, linewidth=1, color='blue', alpha=0.5)
            if len(episodes) > 100:
                rolling_avg = pd.Series(throughput_changes).rolling(window=100).mean()
                plt.plot(episodes, rolling_avg, linewidth=2, color='red', label='100-episode moving average')

            plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
            plt.xlabel('Episode')
            plt.ylabel('Overflow Change (%)')
            plt.title('Bridge Closure Impact - Overflow Change')
            plt.grid(True, alpha=0.3)
            if len(episodes) > 100:
                plt.legend()

            plt.tight_layout()
            plt.savefig('results_1/training_closure_impact.png')


# Compare with heuristic policy
def run_heuristic_policy(env, rtg_rate, policy_type="balanced"):
    """Run a heuristic policy on the environment"""
    observation = env.reset(rtg_rate)

    done = False
    total_reward = 0
    hourly_data = []

    while not done:
        if policy_type == "balanced":
            # Balanced allocation - 50% across all routes
            actions = [5 for _ in range(len(env.routes))]  # Action 2 corresponds to 50% allocation
        elif policy_type == "greedy":
            # Greedy allocation - prioritize faster routes first
            # Sort routes by capacity (highest to lowest)
            route_capacities = [(route, env.base_capacities[route]) for route in env.routes]
            route_capacities.sort(key=lambda x: x[1], reverse=True)

            # Assign higher allocations to higher capacity routes
            actions = []
            for i, (route, _) in enumerate(route_capacities):
                if i == 0:
                    actions.append(10)  # 100% for highest capacity
                elif i == 1:
                    actions.append(7)  # 75% for second highest
                else:
                    actions.append(5)  # 50% for the rest
        elif policy_type == "congestion_aware":
            # Allocate based on current congestion levels
            current_congestion = [env.route_congestion[route] for route in env.routes]
            # Inverse mapping of congestion to actions (higher congestion = lower allocation)
            actions = [min(10, max(0, int(10 * (1 - cong)))) for cong in current_congestion]
        elif policy_type == "adaptive":
            # After bridge closure, increase allocation to remaining routes
            if env.vincent_bridge_closed:
                actions = [7 for _ in range(len(env.routes))]  # 75% allocation after closure
            else:
                actions = [5 for _ in range(len(env.routes))]  # 50% allocation before closure
        elif policy_type == "dynamic":
            # Dynamic policy that adjusts based on current queue size and time of day
            actions = []
            for route in env.routes:
                # High allocation during rush hours or when queue is large
                if (6 <= env.current_hour < 10 or 15 <= env.current_hour < 19 or
                        env.waiting_queue > 0.5 * env.max_queue_size):
                    actions.append(10)  # 100% allocation during peak times
                elif env.waiting_queue > 0.3 * env.max_queue_size:
                    actions.append(7)  # 75% allocation for moderate queues
                else:
                    actions.append(5)  # 50% allocation for normal conditions
        else:
            # Default to balanced
            actions = [5 for _ in range(len(env.routes))]

        observation_, reward, done, info = env.step(actions)
        total_reward += reward

        hourly_data.append({
            "hour": env.current_hour,
            "processed": info['processed'].copy(),
            "usage": info['usage_percentages'].copy(),
            "queue": info['queue_size'],
            "congestion": info['congestion'].copy(),
            "overflow": info.get('overflow', 0),
            "bridge_closed": info.get('bridge_closed', False)
        })

        observation = observation_

    # Extract closure metrics if available
    closure_metrics = {}
    if "pre_closure_metrics" in info and "post_closure_metrics" in info:
        closure_metrics = {
            "pre_closure": info["pre_closure_metrics"],
            "post_closure": info["post_closure_metrics"]
        }

    return {
        "total_reward": total_reward,
        "hourly_data": hourly_data,
        "total_throughput": sum(env.hourly_throughput),
        "overflow_count": env.overflow_count,
        "final_queue": env.waiting_queue,
        "closure_metrics": closure_metrics
    }


# Example usage
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create environment
    env = PortEnvironment(close_vincent_bridge_at=12)  # Close bridge at noon by default

    print("Port Environment Initialized with:")
    print(f"  Routes: {env.routes}")
    print(f"  Base demand: {env.base_demand} containers/day (±{env.demand_variation * 100:.0f}% variation)")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Time factors: Morning Rush {env.time_factors['morning_rush']['factor']}, "
          f"Midday {env.time_factors['midday']['factor']}, "
          f"Evening Rush {env.time_factors['evening_rush']['factor']}, "
          f"Night {env.time_factors['night']['factor']}")
    print(f"  Capacity variance: ±{env.capacity_variance * 100:.0f}%")
    print(f"  Max queue size: {env.max_queue_size} containers")
    print(f"  Vincent Thomas Bridge will close at hour {env.close_vincent_bridge_at}")

    # Create multi-agent PPO
    mappo = MultiAgentPPO(env, n_agents=4, max_episodes=250)  # Now 4 agents for all routes including Vincent Thomas

    print("\nTraining multi-agent PPO...")
    print("First 200 episodes will use random actions for exploration")

    # Train with various RTG rates and bridge closure times
    train_rtg_rates = [12000, 14000, 16000, 18000, 20000]
    closure_hours = [10, 12, 14, 16]  # Test different bridge closure times
    mappo.train(train_rtg_rates, render_freq=50, closure_hours=closure_hours)

    # Plot training progress
    print("\nPlotting training progress...")
    mappo.plot_training_progress()

    # Test on specific RTG rates and bridge closure times
    print("\nTesting trained agents with various RTG rates and bridge closure times...")
    test_rtg_rates = [12000, 16000, 20000]
    test_closure_hours = [8, 12, 16]  # Test morning, noon, and afternoon closures
    results = mappo.test(test_rtg_rates, bridge_closure_hours=test_closure_hours)

    # Visualize results
    print("\nVisualizing test results...")
    mappo.visualize_results(results)

    # Save trained models
    print("\nSaving trained models...")
    for i, agent in enumerate(mappo.agents):
        agent.save_models(f"route_{i}")

    # Compare with heuristic policies
    print("\n----- COMPARING RL SOLUTION WITH HEURISTIC POLICIES -----")

    rtg_rate = 16000  # Use a middle RTG rate for comparison
    closure_hour = 12  # Noon closure for comparison
    env.close_vincent_bridge_at = closure_hour

    # Run different heuristic policies
    balanced_result = run_heuristic_policy(env, rtg_rate, "balanced")
    greedy_result = run_heuristic_policy(env, rtg_rate, "greedy")
    congestion_aware_result = run_heuristic_policy(env, rtg_rate, "congestion_aware")
    adaptive_result = run_heuristic_policy(env, rtg_rate, "adaptive")
    dynamic_result = run_heuristic_policy(env, rtg_rate, "dynamic")

    # Find RL result for this RTG rate and closure hour
    rl_result = next((r for r in results if r["RTG_Rate"] == rtg_rate and r["Bridge_Closure_Hour"] == closure_hour),
                     None)

    print(f"\nPolicy Comparison (RTG Rate: {rtg_rate}, Bridge Closure at: {closure_hour}:00):")
    print("  Policy          | Total Reward | Throughput | Overflow | Final Queue")
    print("  ----------------------------------------------------------------")

    if rl_result:
        print(
            f"  RL Solution     | N/A          | {rl_result['Total_Throughput']:.0f}      | {rl_result['Overflow_Count']:.0f}       | N/A")

    print(
        f"  Balanced        | {balanced_result['total_reward']:.2f}       | {balanced_result['total_throughput']:.0f}      | {balanced_result['overflow_count']:.0f}       | {balanced_result['final_queue']:.0f}")
    print(
        f"  Greedy          | {greedy_result['total_reward']:.2f}       | {greedy_result['total_throughput']:.0f}      | {greedy_result['overflow_count']:.0f}       | {greedy_result['final_queue']:.0f}")
    print(
        f"  Congestion-Aware| {congestion_aware_result['total_reward']:.2f}       | {congestion_aware_result['total_throughput']:.0f}      | {congestion_aware_result['overflow_count']:.0f}       | {congestion_aware_result['final_queue']:.0f}")
    print(
        f"  Adaptive        | {adaptive_result['total_reward']:.2f}       | {adaptive_result['total_throughput']:.0f}      | {adaptive_result['overflow_count']:.0f}       | {adaptive_result['final_queue']:.0f}")
    print(
        f"  Dynamic         | {dynamic_result['total_reward']:.2f}       | {dynamic_result['total_throughput']:.0f}      | {dynamic_result['overflow_count']:.0f}       | {dynamic_result['final_queue']:.0f}")

    # Analyze closure impact for each policy
    print("\n----- ANALYZING BRIDGE CLOSURE IMPACT BY POLICY -----")

    policies = {
        "RL Solution": rl_result,
        "Balanced": balanced_result,
        "Greedy": greedy_result,
        "Congestion-Aware": congestion_aware_result,
        "Adaptive": adaptive_result,
        "Dynamic": dynamic_result
    }

    print(f"\nMetrics Before and After Bridge Closure at {closure_hour}:00 (RTG Rate: {rtg_rate}):")
    print("  Policy          | Throughput Change | Queue Change | Overflow Change")
    print("  -----------------------------------------------------------------")

    for policy_name, result in policies.items():
        if not result or 'closure_metrics' not in result or not result['closure_metrics']:
            continue

        pre = result['closure_metrics'].get('pre_closure', {})
        post = result['closure_metrics'].get('post_closure', {})

        if not pre or not post:
            continue

        pre_throughput = pre.get('throughput', 0)
        post_throughput = post.get('throughput', 0)
        pre_queue = pre.get('queue_avg', 0)
        post_queue = post.get('queue_avg', 0)
        pre_overflow = pre.get('overflow', 0)
        post_overflow = post.get('overflow', 0)

        throughput_change = ((post_throughput - pre_throughput) / pre_throughput) * 100 if pre_throughput > 0 else 0
        queue_change = ((post_queue - pre_queue) / pre_queue) * 100 if pre_queue > 0 else 0
        overflow_change = ((post_overflow - pre_overflow) / pre_overflow) * 100 if pre_overflow > 0 else 0

        print(
            f"  {policy_name.ljust(16)}| {throughput_change:+.1f}%           | {queue_change:+.1f}%        | {overflow_change:+.1f}%")

    # Analyze peak hour performance
    print("\n----- ANALYZING PEAK HOUR PERFORMANCE -----")

    # Find a relevant RTG rate result for analysis
    mid_idx = len(results) // 2
    mid_result = results[mid_idx]
    rtg_rate = mid_result["RTG_Rate"]
    close_hour = mid_result["Bridge_Closure_Hour"]

    # Find the peak hour with highest queue size
    peak_hour = max(range(24), key=lambda h: mid_result["Queue_By_Hour"][h])

    print(f"\nPeak Hour Analysis (RTG Rate: {rtg_rate}, Bridge Closure: {close_hour}:00):")
    print(f"  Peak hour (highest queue): {peak_hour}:00")
    print(f"  Queue size: {mid_result['Queue_By_Hour'][peak_hour]:.0f} containers")
    print(f"  Bridge status: {'Closed' if peak_hour >= close_hour else 'Open'}")

    all_routes = ["Vincent Thomas Bridge", "Gerald Desmond Bridge", "Long Beach Gateway", "Railroad"]
    print("  Route utilization:")
    for route in all_routes:
        usage = mid_result["Usage_By_Hour"][peak_hour].get(route, 0)
        if route == "Vincent Thomas Bridge" and peak_hour >= close_hour:
            print(f"    {route}: CLOSED")
        else:
            congestion = mid_result["Congestion_By_Hour"][peak_hour].get(route, 0) * 100
            print(f"    {route}: {usage:.1f}% utilization, {congestion:.1f}% congestion")

    # Analyze post-closure adaptation
    print("\n----- ANALYZING POST-CLOSURE ADAPTATION -----")

    # Find hours just before and after closure
    before_closure = close_hour - 1
    after_closure = close_hour + 1

    print(f"\nBefore vs. After Closure Comparison (RTG Rate: {rtg_rate}, Bridge Closure: {close_hour}:00):")
    print(f"  Pre-closure hour: {before_closure}:00")
    print(f"  Post-closure hour: {after_closure}:00")

    # Queue comparison
    pre_queue = mid_result["Queue_By_Hour"][before_closure]
    post_queue = mid_result["Queue_By_Hour"][after_closure]
    queue_change = ((post_queue - pre_queue) / pre_queue) * 100 if pre_queue > 0 else 0
    print(f"  Queue size change: {pre_queue:.0f} -> {post_queue:.0f} containers ({queue_change:+.1f}%)")

    # Traffic redistribution
    print("  Traffic redistribution to remaining routes:")
    for route in all_routes:
        if route == "Vincent Thomas Bridge":
            continue  # Skip closed bridge

        pre_usage = mid_result["Usage_By_Hour"][before_closure].get(route, 0)
        post_usage = mid_result["Usage_By_Hour"][after_closure].get(route, 0)
        usage_change = post_usage - pre_usage

        pre_congestion = mid_result["Congestion_By_Hour"][before_closure].get(route, 0) * 100
        post_congestion = mid_result["Congestion_By_Hour"][after_closure].get(route, 0) * 100
        congestion_change = post_congestion - pre_congestion

        print(f"    {route}: Utilization {pre_usage:.1f}% -> {post_usage:.1f}% ({usage_change:+.1f}%), "
              f"Congestion {pre_congestion:.1f}% -> {post_congestion:.1f}% ({congestion_change:+.1f}%)")

    print("\n----- RECOMMENDATIONS BASED ON BRIDGE CLOSURE ANALYSIS -----")

    print("\n1. Optimal Bridge Closure Timing:")

    # Find best and worst closure hours based on throughput impact
    throughput_impacts = {}
    for result in results:
        if "Closure_Impact" in result and "throughput_change_pct" in result["Closure_Impact"]:
            hour = result["Bridge_Closure_Hour"]
            impact = result["Closure_Impact"]["throughput_change_pct"]
            rtg = result["RTG_Rate"]

            if hour not in throughput_impacts:
                throughput_impacts[hour] = []
            throughput_impacts[hour].append((rtg, impact))

    avg_impacts = {}
    for hour, impacts in throughput_impacts.items():
        avg_impacts[hour] = sum(impact for _, impact in impacts) / len(impacts)

    best_hour = min(avg_impacts.items(), key=lambda x: x[1])[0] if avg_impacts else None
    worst_hour = max(avg_impacts.items(), key=lambda x: x[1])[0] if avg_impacts else None

    if best_hour is not None and worst_hour is not None:
        print(f"  - If bridge closure is inevitable, schedule it at {best_hour}:00 for minimal impact")
        print(f"  - Avoid closing the bridge at {worst_hour}:00 which causes the largest disruption")

    print("\n2. RTG Rate Adjustments:")

    # Find optimal RTG rate for handling bridge closure
    rtg_impacts = {}
    for result in results:
        if "Closure_Impact" in result and "throughput_change_pct" in result["Closure_Impact"]:
            rtg = result["RTG_Rate"]
            impact = result["Closure_Impact"]["throughput_change_pct"]
            hour = result["Bridge_Closure_Hour"]

            if rtg not in rtg_impacts:
                rtg_impacts[rtg] = []
            rtg_impacts[rtg].append((hour, impact))

    avg_rtg_impacts = {}
    for rtg, impacts in rtg_impacts.items():
        avg_rtg_impacts[rtg] = sum(impact for _, impact in impacts) / len(impacts)

    best_rtg = min(avg_rtg_impacts.items(), key=lambda x: x[1])[0] if avg_rtg_impacts else None

    if best_rtg is not None:
        print(f"  - Optimal RTG rate to minimize bridge closure impact: {best_rtg} containers/day")
        print(f"  - Increase RTG rate by {((best_rtg / 16000) - 1) * 100:.1f}% from baseline during bridge closure periods")

    print("\n3. Traffic Management Strategy:")

    # Compare policy performance during closure
    # Fix for the "KeyError: 'overflow_count'" issue
    best_policy = min(policies.items(), key=lambda x: x[1]['Overflow_Count'] if x[1] and 'Overflow_Count' in x[1] else
    (x[1]['overflow_count'] if x[1] and 'overflow_count' in x[1] else float('inf')))[0]

    print(f"  - Best traffic management policy during bridge closure: {best_policy}")
    print("  - Recommended route allocations after bridge closure:")

    # Find optimal route allocations post-closure
    if rl_result:
        all_routes = ["Vincent Thomas Bridge", "Gerald Desmond Bridge", "Long Beach Gateway", "Railroad"]
        for route in all_routes:
            if route == "Vincent Thomas Bridge":
                continue  # Skip closed bridge

            # Find average allocation after closure
            post_allocations = []
            for hour in range(close_hour, 24):
                usage = rl_result["Usage_By_Hour"][hour].get(route, 0)
                post_allocations.append(usage)

            avg_allocation = sum(post_allocations) / len(post_allocations) if post_allocations else 0
            print(f"    {route}: {avg_allocation:.1f}% utilization")

    print("\n4. Infrastructure Investment Recommendations:")

    # Identify bottlenecks after bridge closure
    if rl_result:
        route_max_usage_post = {}
        for route in all_routes:
            if route == "Vincent Thomas Bridge":
                continue  # Skip closed bridge

            max_usage = max([rl_result["Usage_By_Hour"][hour].get(route, 0) for hour in range(close_hour, 24)])
            route_max_usage_post[route] = max_usage

        bottleneck_route = max(route_max_usage_post.items(), key=lambda x: x[1]) if route_max_usage_post else (None, 0)

        if bottleneck_route[0]:
            print(
                f"  - Primary bottleneck after bridge closure: {bottleneck_route[0]} (reaches {bottleneck_route[1]:.1f}% capacity)")

            if bottleneck_route[1] > 90:
                capacity_increase = ((bottleneck_route[1] - 85) / 85) * 100
                print(f"  - Recommended capacity increase: ~{capacity_increase:.0f}% to maintain safe operating margins")

            # Calculate capacity needed to compensate for Vincent Thomas Bridge
            vt_capacity = env.base_capacities["Vincent Thomas Bridge"]
            print(
                f"  - Additional capacity needed to compensate for Vincent Thomas Bridge closure: {vt_capacity} containers/day")
            print(f"  - Consider expanding {bottleneck_route[0]} capacity by at least {vt_capacity / 2} containers/day")