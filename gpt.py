import time
import torch
import torch.nn as nn
from tqdm import tqdm
from results import start_training_results, add_evaluation_results, end_training_results
from tokenizer.tiktoken_tokenizer import encode, decode, vocab_size, tokenizer_id
import torch.optim as optim

torch.manual_seed(1)

dropout_rate = 0.1
num_heads = 6
num_blocks = 6
context_window = 64
embedding_dim = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("./resources/wiki.train.txt", "r", encoding="utf-8") as file:
    content = file.read() 
data = torch.tensor(encode(content), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        # Linear layers to project input to queries, keys, values
        # TODO, does not need to be embedding_dim sized
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        # Triangulation matrix, avoid tokens to attend future ones (upper matrix with ones)
        self.register_buffer(
            "tril", torch.tril(torch.ones(context_window, context_window))
        )
        self.dropout = nn.Dropout(dropout_rate)

    # See https://jalammar.github.io/illustrated-transformer/#multi-head-attention for nice math explanations
    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape

        # Will be batch_size x embedding_dim
        k_out = self.key(x)
        q_out = self.query(x)
        v_out = self.value(x)
        # compute attention scores ("affinities")

        scores_out = torch.matmul(q_out, k_out.transpose(-2, -1)) / (
            self.head_size**0.5
        )
        # This step is important to make sure it does not 'attend' to future words. Crucial for prediction tasks
        scores_out = scores_out.masked_fill(
            self.tril[:seq_len, :seq_len] == 0, float("-inf")
        )
        attn_weights_out = torch.softmax(scores_out, dim=-1)
        dropout_out = self.dropout(attn_weights_out)
        out = torch.matmul(dropout_out, v_out)
        return out


class DecoderBlock(nn.Module):

    def __init__(self):
        super().__init__()

        # Multi head
        head_size = embedding_dim // num_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Brings the dimension back down
        self.proj = nn.Linear(head_size * num_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.ln1 = nn.LayerNorm(embedding_dim)
        # TODO, tweaking internal_dim what can it do?
        internal_dim = 4 * embedding_dim
        self.feedForward = nn.Sequential(
            nn.Linear(embedding_dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )
        self.ln2 = nn.LayerNorm(embedding_dim)

    # Follows the attention paper transformer block
    def forward(self, x):
        # Multi-Head Attention
        # Dimension is (batch_size, seq_len, embedding_dim*num_heads)
        head_out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Followed by projection down to embedding_dim
        # Dimension is (batch_size, seq_len, embedding_dim)
        attn_out = self.proj(head_out)
        # Dropout is somewhat important for small datasets
        drop_out = self.dropout(attn_out)
        residual = x + drop_out
        # Add & Norm
        # Dimension is (batch_size, seq_len, embedding_dim)
        ln1_out = self.ln1(residual)
        # Feed Forward
        # Dimension is (batch_size, seq_len, embedding_dim)
        ff_out = self.feedForward(ln1_out)
        residual = ln1_out + ff_out
        # Dimension is (batch_size, seq_len, embedding_dim)
        # Add & Norm
        # Dimension is (batch_size, seq_len, embedding_dim)
        out = self.ln2(residual)
        return out


class Gpt(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.position_embedding = nn.Embedding(context_window, embedding_dim)
        # Can use sequential, but lets illustrate
        self.blocks = nn.Sequential(*[DecoderBlock() for _ in range(num_blocks)])

        self.ln_f = nn.LayerNorm(embedding_dim)  # final layer norm
        # From https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py#L152
        # Needed to make the training more smooth and avoid exploding gradients and so on
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        batch_size, seq = x.shape

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(seq, device=device))
        # Dimension is (batch_size, seq_len, embedding_dim)
        tok_pos_out = tok_emb + pos_emb

        # Transformer block N times
        block_out = self.blocks(tok_pos_out)

        ln_f_out = self.ln_f(block_out)  # (B,T,C)
        # Dimension is (batch_size, seq_len, vocab_size)
        logits = self.linear(ln_f_out)
        # We do not try and predict all words at the same time, just next one
        # For faster parallel training, use all subresults in the prediction
        # Since we output (seq_len, vocab_size) for each sample, we can create
        # a loss on all subsamples in the seq_len, given then that targets is slided by one
        # Example
        # Seq -> ["the", "cat", "sat", "on"]
        # Target -> ["cat", "sat", "on", "mat"]
        # Predictions for
        #     "the" -> "cat"
        #     "the cat" -> "sat"
        #     "the cat sat" -> "on"
        # This would add more samples for the loss gradient, but made it a bit more complicated to understand
        # TODO: Can we somehow modify the whole architecture so the last linear layer outputs the correct dimensions (batch_size, embedding_dim) -> softmax that
        # logits = logits[:, -1, :]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss


# Run training
print(f"Running training on device {device}")
print(f"Vocab size {vocab_size}")
max_iters = 20000
eval_iter = 1000
lr = 3e-4
batch_size = 32


def get_batch(dataset):
    ix = torch.randint(len(dataset) - context_window, (batch_size,))
    x = torch.stack([dataset[i : i + context_window] for i in ix])
    y = torch.stack([dataset[i + 1 : i + context_window + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


gpt = Gpt()
gpt = gpt.to(device)
optimizer = optim.Adam(gpt.parameters(), lr=lr)
results_file = start_training_results(
    num_heads,
    num_blocks,
    batch_size,
    tokenizer_id,
    context_window,
    embedding_dim,
    dropout_rate
)

def eval(iteration: int, max_iterations: int):
    with torch.no_grad():
        gpt.eval()
        # Generate a preview for each epoc, 128 seq length
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        for i in tqdm(range(0, 512), desc="Generating", unit="token"):
            logits, loss = gpt(context[:, -context_window:])
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
        generated_text = decode(context[0].tolist())

        losses = torch.zeros(32)
        for i in tqdm(range(0, 32), desc="Evaluate", unit="batch"):
            x, y = get_batch(val_data)
            logits, loss = gpt(x, y)
            losses[i] = loss.item()
        avg_loss = losses.mean()
        perplexity = torch.exp(torch.tensor(avg_loss))

        add_evaluation_results(
            results_file,
            generated_text,
            avg_loss,
            perplexity,
            iteration,
            max_iterations
        )

start = time.perf_counter()
for iteration in tqdm(range(max_iters), desc="Training Epochs", unit="iter"):
    if iteration % eval_iter == 0:
        eval(iteration, max_iters)

    # Training
    gpt.train()
    x, y = get_batch(train_data)
    optimizer.zero_grad()
    logits, loss = gpt(x, y)
    loss.backward()
    optimizer.step()

# Last eval after training is done
eval(
    max_iters,
    max_iters
)

end = time.perf_counter()
end_training_results(results_file, end - start)