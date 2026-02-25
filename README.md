# Classificador de Imagens com TensorFlow e FastAPI

Este repositório contém um estudo inicial de classificação de imagens utilizando redes neurais convolucionais (CNN). O objetivo não é fornecer um modelo de alta precisão para uso comercial, mas sim documentar meu processo de aprendizado com a biblioteca TensorFlow e a exposição de modelos através de APIs simples com FastAPI.

Limitações Conhecidas:

- O modelo foi treinado com um conjunto de dados reduzido e pode apresentar falsos positivos.

- A arquitetura é básica e serve como prova de conceito para o fluxo "Treino -> Exportação -> Deploy".

- Não há tratamento avançado de erros ou camadas de segurança na API.

## Funcionalidades

* **Treinamento de Modelo:** Arquitetura CNN customizada com normalização integrada (`Rescaling`) para evitar vazamento de dados.
* **Predição em Lote (CLI):** Script para testar imagens individuais ou diretórios inteiros via terminal, gerando relatórios em formato tabular (Pandas).
* **API REST:** Endpoint pronto para receber uploads de imagens e retornar a classe predita em formato JSON.

## 📁 Estrutura do Projeto

\`\`\`text
meu-projeto-ia/
├── data/               # Diretório do dataset (não versionado no Git)
│   ├── train/          # Imagens de treino separadas por pastas de classes
│   └── test/           # Imagens de validação/teste separadas por classes
├── train.py            # Script para treinar e salvar o modelo (.keras)
├── predict.py          # Script de predição via terminal (CLI)
├── api.py              # Código da API usando FastAPI
├── model.keras         # Modelo treinado salvo (gerado após o treino)
├── requirements.txt    # Dependências do projeto
└── README.md           # Documentação do projeto
\`\`\`

## Tecnologias Utilizadas

* **Python 3.x**
* **TensorFlow / Keras** (Construção e treino da Rede Neural)
* **FastAPI & Uvicorn** (Criação do servidor da API)
* **Pandas & NumPy** (Manipulação de dados e resultados)
* **Pillow (PIL)** (Processamento de imagens)

## ⚙️ Como Instalar e Rodar

### 1. Clone o repositório
```bash
git clone https://github.com/marcossantoswz/classificacao-imagens-tensorflow-api.git
cd classificacao-imagens-tensorflow-api
```

### 3. Organização dos Dados (Dataset)
Crie uma pasta chamada `data` na raiz do projeto e organize suas imagens da seguinte forma:
```text
data/
  train/
    classe_1/
    classe_2/
  test/
    classe_1/
    classe_2/
```

### 4. Treinando o Modelo
Para treinar a rede neural e gerar o arquivo `model.keras`, rode:
```bash
python train.py
```

### 5. Fazendo Predições (Terminal)
Para testar o modelo diretamente no terminal:
```bash
python predict.py
```

### 6. Rodando a API (FastAPI)
Para iniciar o servidor local da sua API:
```bash
python3 api.py
```
A API estará rodando em `http://localhost:8000`. Você pode acessar a documentação interativa (Swagger UI) em `http://localhost:8000/docs` para testar o upload de imagens diretamente pelo navegador!

É possível também usar o seguinte `endpoint`:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@NomeFoto.jpg;type=image/jpg'
```

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
