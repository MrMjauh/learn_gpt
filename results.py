import os
from datetime import datetime

def start_training_results(
    num_heads,
    num_blocks,
    batch_size,
    tokenizer_id,
    context_window,
    embedding_dim,
    dropout_rate
):
    today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("./training", exist_ok=True)
    filename = f"./training/{today}.txt"
    
    headers = [
        "num_heads",
        "num_blocks",
        "batch_size",
        "tokenizer_id",
        "context_window",
        "embedding_dim",
        "dropout_rate"
    ]
    
    values = [
        str(num_heads),
        str(num_blocks),
        str(batch_size),
        str(tokenizer_id),
        str(context_window),
        str(embedding_dim),
        str(dropout_rate)
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        for h, v in zip(headers, values):
            f.write(f"{h}={v}\n")
        f.write(f"comment=...\n")
        f.write("\n")

    print(f"Created new training file: {filename}")
    print("Training parameters:")
    for h, v in zip(headers, values):
        print(f"{h}: {v}")

    return filename


def add_evaluation_results(
    filename: str,
    generated_text: str,
    loss: float,
    perplexity: float,
    iteration: int,
    max_iterations
):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"===Evaluation ({iteration}/{max_iterations})===\n")
        f.write("Example\n")
        f.write("```\n")
        f.write(f"{generated_text}\n")
        f.write("```\n")
        f.write(f"loss = {loss}\n")
        f.write(f"perplexity = {perplexity}\n\n")

    print("Evaluation:")
    print(f"Generated text:\n{generated_text}")
    print(f"loss = {loss}")
    print(f"perplexity = {perplexity}")

def end_training_results(filename: str, elapsed_time: float):
    # Read current content
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Prepend the training time info
    new_content = f"elapsed_time = {elapsed_time}s\n" + content
    
    # Write back to the same file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Training done in ${elapsed_time}s")