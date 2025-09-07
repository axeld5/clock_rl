import re

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards

def format_hour_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\d{1,2}:\d{2}:\d{2}.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards

def accuracy_reward(completions: list[str], solution: list[str], **kwargs) -> list[float]:
    """Reward function that checks if the completion matches the ground truth.
    Ground truth is expected to be in the format matching format_hour_reward pattern.
    """
    rewards = []

    # Pattern to extract time from answer section
    time_pattern = r"<answer>\n.*?(\d{1,2}:\d{2}:\d{2}).*?\n</answer>"

    for completion, sol in zip(completions, solution):
        # Extract time from completion
        completion_match = re.search(time_pattern, completion, re.DOTALL | re.MULTILINE)
        completion_time = completion_match.group(1) if completion_match else None

        # Extract time from solution
        solution_time = sol

        # Compare extracted times
        if completion_time and solution_time:
            reward = 1.0 if completion_time == solution_time else 0.0
        else:
            # Fallback to exact text match if time extraction fails
            reward = 1.0 if completion.strip() == sol.strip() else 0.0

        rewards.append(reward)

    return rewards

def test_reward_functions():
    """Test the reward functions with correct format and ground truth answers."""

    # Test cases with correct format and ground truth answers
    correct_completions = [
        "<think>\nI need to look at the clock and determine the time.\n</think>\n<answer>\n12:34:56\n</answer>",
        "<think>\nThe clock shows the hour, minute, and second hands.\n</think>\n<answer>\n03:15:45\n</answer>",
        "<think>\nAnalyzing the clock image carefully.\n</think>\n<answer>\n09:27:13\n</answer>",
        "<think>\nLet me calculate the time based on hand positions.\n</think>\n<answer>\n11:58:22\n</answer>",
    ]

    # Corresponding ground truth solutions
    solutions = [
        "12:34:56",
        "03:15:45",
        "09:27:13",
        "11:58:22",
    ]

    print("Testing reward functions with correct format and ground truth answers...")

    # Test format_reward
    print("\n=== Testing format_reward ===")
    format_rewards = format_reward(correct_completions)
    print(f"Format rewards: {format_rewards}")
    assert all(reward == 1.0 for reward in format_rewards), "All format rewards should be 1.0"
    print("✓ format_reward: All correct completions have reward 1.0")

    # Test format_hour_reward
    print("\n=== Testing format_hour_reward ===")
    format_hour_rewards = format_hour_reward(correct_completions)
    print(f"Format hour rewards: {format_hour_rewards}")
    assert all(reward == 1.0 for reward in format_hour_rewards), "All format hour rewards should be 1.0"
    print("✓ format_hour_reward: All correct completions have reward 1.0")

    # Test accuracy_reward
    print("\n=== Testing accuracy_reward ===")
    accuracy_rewards = accuracy_reward(correct_completions, solutions)
    print(f"Accuracy rewards: {accuracy_rewards}")
    assert all(reward == 1.0 for reward in accuracy_rewards), "All accuracy rewards should be 1.0"
    print("✓ accuracy_reward: All correct completions have reward 1.0")

    print("\n=== Testing edge cases ===")

    # Test with incorrect format
    incorrect_format_completions = [
        "12:34:56",  # No tags
        "<think>thinking</think>12:34:56",  # Missing answer tags
        "<answer>12:34:56</answer>",  # Missing think tags
        "<think>\nthinking\n</think>\n<answer>\n12:34:56",  # Missing closing answer tag
    ]

    print("\nTesting format_reward with incorrect formats:")
    incorrect_format_rewards = format_reward(incorrect_format_completions)
    print(f"Incorrect format rewards: {incorrect_format_rewards}")
    assert all(reward == 0.0 for reward in incorrect_format_rewards), "All incorrect format rewards should be 0.0"
    print("✓ format_reward: Incorrect formats have reward 0.0")

    print("\nTesting format_hour_reward with incorrect formats:")
    incorrect_format_hour_rewards = format_hour_reward(incorrect_format_completions)
    print(f"Incorrect format hour rewards: {incorrect_format_hour_rewards}")
    assert all(reward == 0.0 for reward in incorrect_format_hour_rewards), "All incorrect format hour rewards should be 0.0"
    print("✓ format_hour_reward: Incorrect formats have reward 0.0")

    # Test with wrong time in accuracy_reward
    wrong_time_completion = "<think>\nthinking\n</think>\n<answer>\n12:34:56\n</answer>"
    correct_solution = "<think>\ncorrect\n</think>\n<answer>\n11:22:33\n</answer>"

    print("\nTesting accuracy_reward with wrong time:")
    wrong_time_reward = accuracy_reward([wrong_time_completion], [correct_solution])
    print(f"Wrong time reward: {wrong_time_reward}")
    assert wrong_time_reward[0] == 0.0, "Wrong time should have reward 0.0"
    print("✓ accuracy_reward: Wrong time has reward 0.0")

    # Test with missing time pattern
    no_time_completion = "<think>\nthinking\n</think>\n<answer>\nNo time visible\n</answer>"
    no_time_solution = "<think>\ncorrect\n</think>\n<answer>\n12:34:56\n</answer>"

    print("\nTesting accuracy_reward with missing time pattern:")
    no_time_reward = accuracy_reward([no_time_completion], [no_time_solution])
    print(f"No time reward: {no_time_reward}")
    assert no_time_reward[0] == 0.0, "Missing time should have reward 0.0"
    print("✓ accuracy_reward: Missing time pattern has reward 0.0")

    print("\n🎉 All tests passed! The reward functions work correctly.")

if __name__ == "__main__":
    test_reward_functions()
