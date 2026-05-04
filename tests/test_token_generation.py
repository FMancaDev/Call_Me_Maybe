import sys
sys.path.append("/home/fodemanca/Documentos/Call_Me_Maybe/llm_sdk")

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

print("=" * 60)
print("GERAÇÃO TOKEN-A-TOKEN MANUAL")
print("=" * 60)

print("\nComo funciona a geração token-a-token:")
print("1. Começamos com um prompt (texto inicial)")
print("2. Tokenizamos o prompt")
print("3. Num loop:")
print("   - Passamos os tokens atuais ao modelo")
print("   - Obtemos os logits para o próximo token")
print("   - Escolhemos o token com maior logit (greedy decoding)")
print("   - Adicionamos esse token à sequência")
print("   - Repetimos até atingirmos um token de paragem ou limite")

print("\n" + "=" * 60)
print("EXEMPLO PRÁTICO")
print("=" * 60)

# Prompt inicial
prompt = "The capital of France is"
print(f"\nPrompt inicial: '{prompt}'")

# Tokenizar o prompt
encoded = model.encode(prompt)
input_ids = encoded[0].tolist()
print(f"Tokens iniciais: {input_ids}")

# Gerar token por token
max_tokens = 10
generated_tokens = []
generated_text = ""

print(f"\nGerando {max_tokens} tokens token-a-token:")
print("-" * 60)

for step in range(max_tokens):
    # Obter logits para o próximo token
    logits = model.get_logits_from_input_ids(input_ids)

    # Escolher o token com maior logit (greedy decoding)
    next_token_id = logits.index(max(logits))

    # Adicionar à sequência
    input_ids.append(next_token_id)
    generated_tokens.append(next_token_id)

    # Decodificar o token para ver o texto
    token_text = model.decode([next_token_id])
    generated_text += token_text

    print(f"Passo {step + 1}: Token ID={next_token_id}, Texto='{token_text}', Logit={logits[next_token_id]:.4f}")

print("-" * 60)
print(f"\nTexto gerado: '{generated_text}'")
print(f"Tokens gerados: {generated_tokens}")

print("\n" + "=" * 60)
print("COMPARAÇÃO: GERAÇÃO AUTOMÁTICA VS MANUAL")
print("=" * 60)

# Geração automática (se o modelo tiver um método generate)
print("\nNota: O SDK não tem um método generate() automático,")
print("por isso a geração manual é a única forma de gerar texto.")

print("\n" + "=" * 60)
print("CONCLUSÃO")
print("=" * 60)
print("\nA geração token-a-token manual permite-nos:")
print("- Controlar exatamente qual token é escolhido em cada passo")
print("- Implementar constrained decoding (filtrar tokens válidos)")
print("- Implementar diferentes estratégias de sampling (temperature, top-k, etc.)")
print("- Parar a geração em condições específicas")
print("\nPara o constrained decoding, vamos:")
print("- Em cada passo, filtrar quais tokens são válidos")
print("- Escolher apenas entre os tokens válidos")
print("- Garantir que o JSON gerado é sempre válido")
