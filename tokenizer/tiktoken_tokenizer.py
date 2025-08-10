import tiktoken

enc = tiktoken.get_encoding("gpt2")

def encode(data: str):
    return enc.encode(data)

def decode(tensor):
    return enc.decode(tensor)

tokenizer_id = "tiktoken"
vocab_size = enc.n_vocab