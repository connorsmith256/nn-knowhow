from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate, prompt_template, system_message
)

def record_to_sample(record):
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(
        input=input,
        target=target,
        metadata={"reasoning": reasoning.strip()}
    )


def sample_to_fewshot(sample):
    return (
        f"{sample.input}\n\nReasoning:\n"
        + f"{sample.metadata['reasoning']}\n\n"
        + f"ANSWER: {sample.target}"
    )

MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your
response should be of the form "ANSWER: $ANSWER" (without quotes)
where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form
"ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to
the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

@task
def gsm8k(fewshot=10, fewshot_seed=42):
    # build plan dynamically (may or may not be doing fewshot)
    plan = [prompt_template(MATH_PROMPT_TEMPLATE), generate()]
    if fewshot:
        fewshots = hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        plan.insert(
            0,
            system_message(
                "\n\n".join([sample_to_fewshot(sample) for sample in fewshots])
            ),
        )

    # define task
    return Task(
        dataset=hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="test",
            sample_fields=record_to_sample,
        ),
        plan=plan,
        scorer=match(numeric=True),
    )