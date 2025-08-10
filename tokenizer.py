import re
from consts import TOKEN_END_Of_TEXT, TOKEN_PAD, TOKEN_UNKOWN

vocab = [
    TOKEN_END_Of_TEXT,
    TOKEN_UNKOWN,
    TOKEN_PAD,
    " ",
    # Lowercase letters
    "a","b","c","d","e","f","g","h","i","j","k","l","m",
    "n","o","p","q","r","s","t","u","v","w","x","y","z",
    # Uppercase letters
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    # Digits
    "0","1","2","3","4","5","6","7","8","9",
    # Punctuation
    ".", ",", "!", "?", ";", ":", "'", '"', "-", "(", ")", "[", "]", "{", "}", "/", "\\",
    # Common symbols
    "@", "#", "$", "%", "^", "&", "*", "_", "+", "=",
    # Whitespace characters (tab, newline)
    "\t", "\n",
    # Common currency symbols
    "€", "£", "¥", "¢",
    # Common math symbols
    "<", ">", "|", "~",
]

def get_token_id(chr: str):
    if chr in vocab:
        return vocab.index(chr)
    return vocab.index(TOKEN_UNKOWN)

def get_token(ix: int):
    return vocab[ix]

def pre_process(data: str) -> str:
    # Replace 2+ spaces with a single space
    data = re.sub(r"\s+", " ", data)
    # Remove leading and trailing spaces
    return data.strip()

def encode(data: str):
    data = pre_process(data)
    return [get_token_id(ch) for ch in data]

def decode(tensor):
    return "".join(get_token(ix) for ix in tensor)

vocab_size = len(vocab)
