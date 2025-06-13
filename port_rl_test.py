import numpy as np
import torch
import json
import os
from collections import defaultdict

# Import the required classes from the provided code
from port_rl_bridge_closed import PortEnvironment, Agent

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_trained_models(save_path="results"):
    """Test the trained models and save the results to a JSON file."""
    # Create environment
    env = PortEnvironment()

    # Define test parameters
    rtg_rates = [12000, 14000, 16000, 18000, 20000]
    bridge_closure_hours = [8, 10, 12, 14, 16]
    n_test_episodes = 3

    # Load trained agents
    agents = []
    for i in range(4):  # 4 agents for 4 routes
        agent = Agent(
            n_actions=env.action_dim,
            input_dims=env.state_dim
        )
        model_path = f'results/actor_route_{i}.pth'
        if os.path.exists(model_path):
            print(f"Loading model for route {i}...")
            agent.actor.load_state_dict(torch.load(model_path, map_location=device))
            agents.append(agent)
        else:
            print(f"Model for route {i} not found at {model_path}")
            return

    all_results = []

    for rtg_rate in rtg_rates:
        for close_hour in bridge_closure_hours:
            print(f"Testing with RTG rate: {rtg_rate}, Bridge closure at: {close_hour}:00")

            env.close_vincent_bridge_at = close_hour
            episode_metrics = defaultdict(list)

            for episode in range(n_test_episodes):
                observation = env.reset(rtg_rate)

                hourly_data = []
                done = False

                while not done:
                    # Get actions from all agents for active routes
                    multi_actions = []
                    action_probs = []  # Store probabilities for policy visualization
                    all_routes = ["Vincent Thomas Bridge", "Gerald Desmond Bridge", "Long Beach Gateway", "Railroad"]

                    # Track which agent goes with which route
                    route_actions = {}
                    route_probs = {}

                    for route_index, route in enumerate(all_routes):
                        if route not in env.routes:
                            continue

                        agent = agents[route_index]
                        action, probs, _, _ = agent.choose_action(observation)
                        multi_actions.append(action)
                        action_probs.append(probs)

                        # Store for this specific route
                        route_actions[route] = int(action)
                        route_probs[route] = probs.tolist()  # Convert numpy array to list

                    # Take action in environment
                    observation_, reward, done, info = env.step(multi_actions)

                    # Store hourly data with policy information
                    hourly_data.append({
                        "hour": env.current_hour,
                        "queue_size": float(info['queue_size']),
                        "total_processed": float(info['total_processed']),
                        "bridge_closed": info['bridge_closed'],
                        "route_data": {
                            route: {
                                "usage": float(info['usage_percentages'].get(route, 0)),
                                "processed": float(info['processed'].get(route, 0)),
                                "congestion": float(info['congestion'].get(route, 0)),
                                "action": route_actions.get(route, 0),  # Add action taken
                                "policy": route_probs.get(route, [0] * env.action_dim)  # Add policy probabilities
                            } for route in all_routes if route in info['usage_percentages']
                        }
                    })

                    observation = observation_

                # Process hourly data to calculate averages by hour
                for hour_data in hourly_data:
                    hour = hour_data["hour"]
                    episode_metrics[f"hour_{hour}"].append(hour_data)

            # Calculate average metrics for each hour across episodes
            hourly_avg_metrics = {}
            for hour in range(24):
                hour_key = f"hour_{hour}"
                if hour_key in episode_metrics and episode_metrics[hour_key]:
                    hourly_data = episode_metrics[hour_key]

                    # Calculate averages for this hour
                    queue_sizes = [data["queue_size"] for data in hourly_data]
                    total_processed = [data["total_processed"] for data in hourly_data]
                    bridge_closed = any(data["bridge_closed"] for data in hourly_data)

                    # Average route data
                    route_metrics = defaultdict(dict)
                    for route in ["Vincent Thomas Bridge", "Gerald Desmond Bridge", "Long Beach Gateway", "Railroad"]:
                        usage_values = []
                        processed_values = []
                        congestion_values = []
                        action_values = []
                        policy_values = []

                        for data in hourly_data:
                            if route in data["route_data"]:
                                route_data = data["route_data"][route]
                                usage_values.append(route_data["usage"])
                                processed_values.append(route_data["processed"])
                                congestion_values.append(route_data["congestion"])
                                action_values.append(route_data["action"])
                                policy_values.append(route_data["policy"])

                        if usage_values:
                            # Average the policy probabilities
                            avg_policy = np.mean(policy_values, axis=0).tolist() if policy_values else []

                            # Find most common action
                            most_common_action = max(set(action_values),
                                                     key=action_values.count) if action_values else 0

                            route_metrics[route] = {
                                "usage": float(np.mean(usage_values)),
                                "processed": float(np.mean(processed_values)),
                                "congestion": float(np.mean(congestion_values)),
                                "action": most_common_action,
                                "policy": avg_policy
                            }

                    hourly_avg_metrics[hour] = {
                        "queue_size": float(np.mean(queue_sizes)),
                        "total_processed": float(np.mean(total_processed)),
                        "bridge_closed": bridge_closed,
                        "route_data": dict(route_metrics)
                    }

            # Add to results
            result = {
                "rtg_rate": rtg_rate,
                "bridge_closure_hour": close_hour,
                "hourly_metrics": hourly_avg_metrics
            }

            all_results.append(result)

    # Save results to a JSON file
    with open(os.path.join(save_path, 'test_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {os.path.join(save_path, 'test_results.json')}")
    return all_results


if __name__ == "__main__":
    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)
    test_trained_models()