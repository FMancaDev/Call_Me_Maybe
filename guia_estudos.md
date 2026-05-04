## ANTES DE COMECAR O PROJETO É PRECISO ENTENDER

1. Sobre o vocabulário:

  - Como é que o ficheiro JSON do vocabulário está estruturado?
  - Como é que um token ID mapeia para uma string?
  - Quantos tokens tem o vocabulário?

  Sobre o SDK:

  - O que é que encode("hello") te devolve exatamente?
  - O que é que get_logits_from_input_ids devolve — que shape tem o tensor, o que
    representa cada dimensão?
  - Como é que os logits se relacionam com o vocabulário?


1. RESPOSTAS

    ## Sobre o vocabulário:

    - Como é que o ficheiro JSON do vocabulário está estruturado?
      É um dicionário Python (dict) onde:
      - As chaves são strings (os tokens)
      - Os valores são inteiros (os IDs dos tokens)
      - Exemplo: {"hello": 14990, "!": 0, '"': 1, ...}

    - Como é que um token ID mapeia para uma string?
      O ficheiro JSON tem mapeamento string -> ID. Para fazer ID -> string,
      preciso inverter o dicionário: {v: k for k, v in vocab_data.items()}
      Exemplo: ID 14990 -> "hello"

    - Quantos tokens tem o vocabulário?
      151,643 tokens

    ## Sobre o SDK:

    - O que é que encode("hello") te devolve exatamente?
      Devolve um tensor PyTorch com shape [1, 1]:
      - [batch_size=1, sequence_length=1]
      - O valor é [[14990]] (o token ID de "hello")
      - Tipo: torch.Tensor

    - O que é que get_logits_from_input_ids devolve — que shape tem o tensor, o que representa cada dimensão?
      Devolve uma lista Python (não tensor) com 151,936 números.
      Cada número é um "logit" - representa a confiança/probabilidade do modelo
      de que aquele token específico deve vir a seguir.
      - logit[i] corresponde ao token com ID=i
      - Maior logit = token mais provável

    - Como é que os logits se relacionam com o vocabulário?
      - O vocabulário tem 151,643 tokens (cada um com um ID único)
      - Os logits têm 151,936 valores (um pouco mais, provavelmente tokens especiais)
      - Cada logit[i] é a "confiança" do modelo de que o token com ID=i deve vir a seguir
      - Para escolher o próximo token, normalmente escolhemos o ID com maior logit
      - No constrained decoding, vamos filtrar quais IDs são válidos antes de escolher

    ## Sobre a geração token-a-token:

    - Como funciona a geração token-a-token manual?
      A geração token-a-token manual funciona assim:
      1. Começamos com um prompt (texto inicial)
      2. Tokenizamos o prompt com model.encode()
      3. Num loop:
         - Passamos os tokens atuais ao modelo com model.get_logits_from_input_ids()
         - Obtemos os logits para o próximo token (lista de confianças)
         - Escolhemos o token com maior logit (greedy decoding): next_token_id = logits.index(max(logits))
         - Adicionamos esse token à sequência: input_ids.append(next_token_id)
         - Repetimos até atingirmos um token de paragem ou limite
      4. Decodificamos os tokens gerados com model.decode()

    - Como é que a geração manual se relaciona com o constrained decoding?
      A geração manual é essencial para o constrained decoding porque:
      - Em cada passo, podemos filtrar quais tokens são válidos antes de escolher
      - Em vez de escolher o token com maior logit global, escolhemos o maior logit
        apenas entre os tokens válidos para o estado atual do JSON
      - Isso garante que o JSON gerado é sempre válido, mesmo que o modelo
        "queira" escolher um token inválido
      - Exemplo: se estamos a gerar {"name": "fn_add_"}, apenas tokens que
        continuem nomes de funções válidos são permitidos 
