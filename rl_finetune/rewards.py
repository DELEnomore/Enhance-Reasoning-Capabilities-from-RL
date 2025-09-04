from latex2sympy2_extended import NormalizationConfig
from math_verify import parse, LatexExtractionConfig, verify


def accuracy_reward(completions, answer, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []
    for content, sol in zip(completions, answer):
        gold_parsed = parse(
            sol,
            parsing_timeout=None,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            answer_parsed = parse(
                content,
                parsing_timeout=None,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

if __name__ == '__main__':
    sol = r'B'
    gold_parsed = parse(
        sol,
        parsing_timeout=None,
        extraction_mode="first_match",
    )
    answer_parsed = parse(
        r'\boxed{b}',
        parsing_timeout=None,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    equations=True,
                    boxed="all",
                    units=True,
                ),
                # Ensures that boxed is tried first
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
    )

    print(verify(gold_parsed, answer_parsed))