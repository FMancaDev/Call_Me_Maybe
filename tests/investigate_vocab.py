import sys
import os
import json

sys.path.append("/home/fodemanca/Documentos/Call_Me_Maybe/llm_sdk")

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

# ============ SOBRE O VOCABULÁRIO ============
print("=" * 60)
print("SOBRE O VOCABULÁRIO")
print("=" * 60)

vocab_path = model.get_path_to_vocab_file()
print(f"\n1. Path para o vocabulário: {vocab_path}")

# Ler o ficheiro JSON do vocabulário
with open(vocab_path, 'r') as f:
    vocab_data = json.load(f)

print(f"\n2. Tipo de dados do vocabulário: {type(vocab_data)}")
print(f"   É um dicionário? {isinstance(vocab_data, dict)}")

# Ver a estrutura
print(f"\n3. Estrutura do vocabulário:")
print(f"   - Número de entradas: {len(vocab_data)}")
print(f"   - Chaves são strings? {all(isinstance(k, str) for k in list(vocab_data.keys())[:10])}")
print(f"   - Valores são inteiros? {all(isinstance(v, int) for v in list(vocab_data.values())[:10])}")

# Mostrar alguns exemplos
print(f"\n4. Exemplos de mapeamento token -> ID:")
sample_items = list(vocab_data.items())[:10]
for token_str, token_id in sample_items:
    print(f"   '{token_str}' -> {token_id}")

# Mostrar como fazer o inverso (ID -> string)
print(f"\n5. Como fazer ID -> string (inverso):")
id_to_token = {v: k for k, v in vocab_data.items()}
sample_ids = [14990, 0, 1, 2, 3]
for token_id in sample_ids:
    if token_id in id_to_token:
        print(f"   {token_id} -> '{id_to_token[token_id]}'")

# Verificar o token do "hello"
print(f"\n6. Token ID de 'hello': {vocab_data.get('hello', 'NÃO ENCONTRADO')}")

# ============ SOBRE O SDK ============
print("\n" + "=" * 60)
print("SOBRE O SDK")
print("=" * 60)

print("\n1. O que é que encode('hello') devolve:")
text = "hello"
encoded = model.encode(text)
print(f"   Texto: '{text}'")
print(f"   Resultado: {encoded}")
print(f"   Tipo: {type(encoded)}")
print(f"   Shape: {encoded.shape}")
print(f"   Significado do shape: [batch_size={encoded.shape[0]}, sequence_length={encoded.shape[1]}]")

# Testar com mais texto
print(f"\n2. Teste com texto mais longo:")
text2 = "hello world"
encoded2 = model.encode(text2)
print(f"   Texto: '{text2}'")
print(f"   Resultado: {encoded2}")
print(f"   Shape: {encoded.shape}")

# Decodificar para verificar
print(f"\n3. Decodificar de volta:")
decoded = model.decode(encoded[0].tolist())
print(f"   Decodificado: '{decoded}'")

print("\n4. O que é que get_logits_from_input_ids devolve:")
input_ids = encoded[0].tolist()
print(f"   Input IDs: {input_ids}")
logits = model.get_logits_from_input_ids(input_ids)
print(f"   Resultado: {logits[:5]}... (primeiros 5)")
print(f"   Tipo: {type(logits)}")
print(f"   Tamanho: {len(logits)}")
print(f"   Significado: Lista de {len(logits)} números, um para cada token no vocabulário")

print(f"\n5. Relação entre logits e vocabulário:")
print(f"   - O vocabulário tem {len(vocab_data)} tokens")
print(f"   - Os logits têm {len(logits)} valores")
print(f"   - Cada logit[i] corresponde à probabilidade/confiança do token com ID=i")
print(f"   - O token com maior logit é o que o modelo acha mais provável")

# Encontrar o token com maior logit
max_logit_idx = logits.index(max(logits))
id_to_token = {v: k for k, v in vocab_data.items()}
print(f"   - Token com maior logit: ID={max_logit_idx}, token='{id_to_token.get(max_logit_idx, 'N/A')}'")
print(f"   - Valor do logit: {logits[max_logit_idx]}")

# Mostrar top 5 tokens
print(f"\n6. Top 5 tokens mais prováveis:")
sorted_indices = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)[:5]
for rank, idx in enumerate(sorted_indices, 1):
    token_str = id_to_token.get(idx, 'N/A')
    print(f"   {rank}. ID={idx}, token='{token_str}', logit={logits[idx]:.4f}")
