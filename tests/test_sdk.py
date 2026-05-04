import sys
import os

print("sys.path antes", sys.path[:3]) # primeiros paths

path_to_add = "/home/fodemanca/Documentos/Call_Me_Maybe/llm_sdk"
print(f"Path a adicinor: {path_to_add}")
print(f"Path existe? {os.path.exists(path_to_add)}")

sys.path.append(path_to_add)

print("sys.path depois:", sys.path[:3]) # primeiros 3 path

try:
    import llm_sdk
    from llm_sdk import Small_LLM_Model
    print("Import llm_sdk funcionou")
    print("conteudo do llm_sdk", dir(llm_sdk))
except ImportError as e:
    print(f"Erro ao importar llm_sdk: {e}")
    sys.exit(1)


# carregar modelo
print("A carregar modelo...")
model = Small_LLM_Model()

# testar o encode
print("\n--- Teste do encode ---")
text = "hello"
encoded = model.encode(text)
print(f"Texto: {text}")
print(f"Resultado do encode: {encoded}")
print(f"Tipo; {type(encoded)}")
print(f"shape: {encoded.shape}")

# explorar o vocabulario
print("\n--- Explorar vocabulario ---")
vocab_path = model.get_path_to_vocab_file()
print(f"Path para o vocabulario: {vocab_path}")

# testar os logits do input
print("--- Teste do get_logits_from_input_ids ---")
input_ids = encoded[0].tolist()
print(f"Input IDs: {input_ids}")

logits = model.get_logits_from_input_ids(input_ids)
print(f"Logits: {logits[:5]}... (primeiros 5)")
print(f"Tipo: {type(logits)}")
print(f"Tamanho: {len(logits)}")
