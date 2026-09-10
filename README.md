# Tech Challenge — Fase 3

Este projeto foi desenvolvido para o Tech Challenge da Fase 3 da Pós-Tech em Machine Learning Engineering da FIAP.

A proposta é criar um sistema capaz de receber um laudo médico em formato de texto e classificar o nível de urgência como:

- normal;
- atenção;
- urgente.

Além do modelo de classificação, o projeto envolve API, containerização, automação de testes, pipeline de treinamento, monitoramento e otimização da latência de inferência.

> Este é um projeto acadêmico e não deve ser utilizado para diagnóstico ou para decisão real.

## Objetivo

O objetivo é disponibilizar um modelo de classificação de textos médicos por meio de uma API REST, considerando não apenas a qualidade das previsões, mas também aspectos importantes para o uso de modelos em produção, como:

- tempo de resposta;
- testes automatizados;
- padronização do ambiente;
- monitoramento da aplicação;
- automação do treinamento;
- otimização do modelo.

## Tecnologias utilizadas

- Python 3.12
- uv
- Scikit-Learn
- FastAPI
- Docker
- Docker Compose
- GitHub Actions
- Apache Airflow
- Prometheus
- Grafana
- ONNX Runtime
- Pytest
- Ruff
- Pre-commit

## Dataset

Utilizamos o [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
(Schopf et al., NLPIR 2022): 14.438 resumos de laudos/artigos médicos (11.550
treino / 2.888 teste), cada um rotulado com uma de 5 categorias de condição
clínica:

| label | condição |
|---|---|
| 1 | neoplasms |
| 2 | digestive system diseases |
| 3 | nervous system diseases |
| 4 | cardiovascular diseases |
| 5 | general pathological conditions |

**Nota sobre o alvo de classificação:** o dataset não traz um rótulo de
urgência (normal/atenção/urgente) — ele classifica o *tipo* de condição
clínica descrita no texto. Para este projeto acadêmico, usamos a categoria
clínica (`condition_label`) como alvo de classificação da API, servindo como
stand-in didático para o cenário de triagem descrito no desafio: o pipeline
de ponta a ponta (API, CI/CD, monitoramento, otimização de latência) é
idêntico independentemente do significado de negócio das classes. Os
arquivos brutos ficam em `data/raw/` e as versões limpas usadas no treino em
`data/processed/`.

Para baixar novamente os dados brutos:

```bash
curl -o data/raw/medical_tc_train.csv https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv
curl -o data/raw/medical_tc_test.csv https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv
curl -o data/raw/medical_tc_labels.csv https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_labels.csv
```

## Modelo baseline

Pipeline `TF-IDF (uni+bigram, 20k features) + Logistic Regression`, treinado
em `src/medical_triage_mlops/ml/train.py`:

```bash
uv run python -m medical_triage_mlops.ml.train
```

Isso lê os CSVs de `data/raw/`, salva as versões processadas em
`data/processed/`, treina o modelo, imprime o relatório de classificação e
salva:

- `models/model.joblib` — pipeline treinado (vetorizador + classificador);
- `reports/latency/train_metrics.json` — métricas de treino/avaliação.

Resultado atual no conjunto de teste: **acurácia ≈ 0.53**, **macro F1 ≈ 0.53**
(baseline razoável para 5 classes; classes com sintomas mais específicos,
como *neoplasms* e *cardiovascular diseases*, têm F1 mais alto que a classe
genérica *general pathological conditions*).

## Estrutura do projeto

```text
tech-challenge-fase03/
├── .github/
│   └── workflows/
├── airflow/
│   ├── dags/
│   └── logs/
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   └── architecture/
├── models/
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── provisioning/
│   └── prometheus/
├── reports/
│   └── latency/
├── scripts/
├── src/
│   └── medical_triage_mlops/
│       ├── api/
│       ├── core/
│       ├── ml/
│       └── monitoring/
├── tests/
│   ├── integration/
│   └── unit/
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```

## Configuração do ambiente

### Pré-requisitos

Para executar o projeto localmente, é necessário ter instalado:

- Git;
- uv;
- Docker;
- Docker Compose.

O Python 3.12 pode ser instalado e gerenciado pelo próprio `uv`.

### Clonar o repositório

```bash
git clone git@github.com:SkiereszDiego/tech-challenge-fase03.git
cd tech-challenge-fase03
```

### Instalar as dependências

```bash
uv sync
```

Para conferir a versão do Python utilizada pelo projeto:

```bash
uv run python --version
```

O resultado deve ser semelhante a:

```text
Python 3.12.x
```

## Variáveis de ambiente

O arquivo `.env.example` contém as configurações necessárias para executar a aplicação.

Crie uma cópia chamada `.env`:

```bash
cp .env.example .env
```

Configurações disponíveis:

```env
APP_NAME=Medical Triage API
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

MODEL_PATH=models/model.joblib
ONNX_MODEL_PATH=models/model.onnx

LOG_LEVEL=INFO
```

O arquivo `.env` não deve ser enviado ao repositório.

## Executando a API

Para iniciar a aplicação em modo de desenvolvimento:

```bash
uv run fastapi dev src/medical_triage_mlops/api/main.py
```

Depois que o servidor iniciar, os principais endereços estarão disponíveis em:

- API: `http://127.0.0.1:8000`
- Health check: `http://127.0.0.1:8000/health`
- Swagger: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Health check

O endpoint `/health` pode ser utilizado para verificar se a API está funcionando.

```http
GET /health
```

Resposta esperada:

```json
{
  "status": "ok"
}
```

## Qualidade de código

O projeto utiliza Ruff para lint e formatação, além do Pytest para os testes automatizados.

### Verificar o código

```bash
uv run ruff check .
```

### Verificar a formatação

```bash
uv run ruff format --check .
```

### Formatar os arquivos

```bash
uv run ruff format .
```

### Executar os testes

```bash
uv run pytest
```

### Executar os testes com cobertura

```bash
uv run pytest --cov=medical_triage_mlops --cov-report=term-missing
```

## Pre-commit

O pre-commit executa automaticamente as principais verificações antes de cada commit.

Para instalar os hooks:

```bash
uv run pre-commit install
```

Para executar todas as verificações manualmente:

```bash
uv run pre-commit run --all-files
```

Atualmente, os hooks executam:

1. Ruff lint;
2. verificação de formatação;
3. testes com Pytest.

## Organização das branches

O projeto utiliza as seguintes branches:

- `main`: versão estável do projeto;
- `develop`: integração das funcionalidades;
- `feat/*`: novas funcionalidades;
- `fix/*`: correções;
- `chore/*`: configurações e tarefas de manutenção;
- `docs/*`: alterações de documentação;
- `test/*`: criação ou ajuste de testes;
- `refactor/*`: refatorações sem mudança de comportamento.

## Fluxo planejado

O fluxo principal do projeto:

```text
Dataset
   ↓
Validação e preparação dos dados
   ↓
Treinamento do modelo
   ↓
Avaliação
   ↓
Salvamento do modelo
   ↓
Conversão para ONNX
   ↓
Comparação de latência
   ↓
Disponibilização pela API
   ↓
Monitoramento com Prometheus e Grafana
```

## Arquitetura

A aplicação será preparada para inferência em tempo real, já que o cenário de triagem exige uma resposta rápida após o envio do laudo.

A decisão de arquitetura em nuvem será detalhada na documentação do projeto, considerando:

- disponibilidade da API;
- escalabilidade;
- segurança dos dados;
- custo de infraestrutura;
- monitoramento;
- implantação dos containers;
- armazenamento dos modelos.

## Andamento do projeto

### Concluído

- [x] Criação do repositório
- [x] Criação das branches iniciais
- [x] Configuração do Python 3.12
- [x] Configuração do ambiente com uv
- [x] Organização inicial das pastas
- [x] Configuração do Ruff
- [x] Configuração do Pytest
- [x] Configuração do pre-commit
- [x] Criação do endpoint `/health`
- [x] Criação do primeiro teste automatizado
- [x] Escolha e documentação do dataset (Medical Abstracts TC Corpus)
- [x] Pipeline de carregamento e limpeza dos dados (`ml/dataset.py`)
- [x] Treino do primeiro modelo (TF-IDF + Logistic Regression)
- [x] Avaliação dos resultados (`reports/latency/train_metrics.json`)
- [x] Endpoint de classificação (`POST /classify`) + testes

### Próximos passos

- [ ] Criar o Dockerfile da API
- [ ] Configurar o GitHub Actions
- [ ] Criar a DAG de treinamento no Airflow
- [ ] Adicionar métricas do Prometheus
- [ ] Configurar o dashboard do Grafana
- [ ] Converter o modelo para ONNX
- [ ] Comparar a latência dos modelos
- [ ] Finalizar a documentação da arquitetura
- [ ] Gravar o vídeo de apresentação STAR