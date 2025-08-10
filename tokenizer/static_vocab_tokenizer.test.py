from tokenizer import encode, decode, get_token_id, get_token

msg = "hello AbcDef"
encoded = encode("hello AbcDef *ÄEs")
decoded = decode(encoded)

print(decoded)
