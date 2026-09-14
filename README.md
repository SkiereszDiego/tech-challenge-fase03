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
│   ├── prepare_dataset.py
│   └── train_model.py
├── src/
│   └── medical_triage_mlops/
│       ├── api/
│       │   ├── main.py
│       │   └── schemas.py
│       ├── core/
│       │   ├── config.py
│       │   └── labels.py
│       ├── ml/
│       │   ├── data.py
│       │   ├── train.py
│       │   └── inference.py
│       └── monitoring/
├── tests/
│   └── unit/
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── Dockerfile
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

## Dataset e mapeamento de urgência

O modelo é treinado com o [Medical Abstracts TC Corpus](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus) (Kaggle), baixado automaticamente via [`kagglehub`](https://pypi.org/project/kagglehub/) — não é necessário criar conta ou configurar credenciais da API do Kaggle para esse dataset. Ele contém 14.438 resumos médicos (`medical_abstract`) rotulados em 5 categorias de doença (`condition_label`).

O dataset **não possui um rótulo de urgência nativo**. Como o desafio permite usar "qualquer dataset tabular contendo uma coluna de texto e uma coluna de target de classificação/urgência", mapeamos as 5 categorias originais para os 3 níveis de urgência pedidos, por meio de uma heurística clínica simplificada (ver `src/medical_triage_mlops/core/labels.py`):

| Categoria original | Nível de urgência | Amostras |
|---|---|---|
| Cardiovascular diseases | **urgente** | 3.051 |
| Neoplasms + Nervous system diseases | **atenção** | 5.088 |
| Digestive system diseases + General pathological conditions | **normal** | 6.299 |

> **Importante:** esse mapeamento é uma simplificação pedagógica para fins do Tech Challenge, não uma classificação clínica validada. Em um cenário real, os rótulos de urgência viriam de triagem feita por profissionais de saúde.

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

### Classificação de laudos

```http
POST /classify
Content-Type: application/json

{
  "text": "Paciente com dor torácica súbita, sudorese e falta de ar."
}
```

Resposta esperada:

```json
{
  "label": "urgente",
  "confidence": 0.82,
  "scores": {
    "normal": 0.05,
    "atenção": 0.13,
    "urgente": 0.82
  }
}
```

Se nenhum modelo tiver sido treinado ainda (`models/model.joblib` ausente), o endpoint responde `503`.

## Pipeline de treino do modelo

O pipeline de dados/treino vive em `src/medical_triage_mlops/ml/` e é exposto por scripts finos em `scripts/`:

```bash
# 1. Baixa o dataset (via kagglehub) e gera data/processed/dataset.csv com o mapeamento de urgência
uv run python scripts/prepare_dataset.py

# 2. Treina o pipeline TF-IDF + LogisticRegression e salva models/model.joblib
uv run python scripts/train_model.py
```

`models/` e `data/` não são versionados no Git (apenas os `.gitkeep`) — é necessário rodar os passos acima localmente antes de subir a API.

## Empacotamento em Docker

A API é empacotada em um `Dockerfile` multi-stage (build com `uv`, imagem final enxuta). Para buildar e medir uma latência local de baseline:

```bash
docker build -t medical-triage-api .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" medical-triage-api
```

```bash
time curl -s -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient with acute chest pain and shortness of breath."}'
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

### Real-time vs. batch

O cenário de triagem exige resposta imediata após o envio do laudo (o médico/enfermeiro está aguardando a classificação para decidir a prioridade de atendimento), então a **inferência é real-time**: a API FastAPI recebe o texto e devolve a classificação síncrona em poucos milissegundos. Já o **retreino do modelo é batch** — não precisa de resposta imediata e pode ser orquestrado periodicamente conforme novos dados chegam.

### Decisão de provedor de nuvem (documentação apenas — nada é provisionado de fato)

Esta seção documenta a decisão arquitetural pedida pelo desafio; a execução do projeto neste repositório é 100% local via Docker Compose.

| Critério | AWS | Azure | GCP |
|---|---|---|---|
| API real-time em container | ECS Fargate / App Runner | Container Apps / AKS | Cloud Run |
| Orquestração de retreino (batch) | MWAA (Airflow gerenciado) ou EventBridge + Batch | Data Factory / AKS CronJob | Cloud Composer (Airflow) |
| Monitoramento | CloudWatch + Managed Prometheus/Grafana (AMP/AMG) | Azure Monitor + Managed Grafana | Cloud Monitoring + self-hosted Prometheus/Grafana |
| Registro de modelos | S3 + (opcional) SageMaker Model Registry | Blob Storage + Azure ML Registry | GCS + Vertex AI Model Registry |
| Maturidade de Airflow gerenciado | Alta (MWAA) | Média (via AKS) | Alta (Cloud Composer) |

**Escolha: AWS.**

- **API de inferência:** ECS Fargate rodando o mesmo container do `Dockerfile` deste repositório, atrás de um Application Load Balancer — sem servidores para gerenciar, escala horizontalmente conforme o número de requisições de triagem, e mantém o comportamento real-time exigido.
- **Retreino (batch):** Amazon MWAA (Managed Workflows for Apache Airflow), disparado por agendamento ou por chegada de novos laudos rotulados em S3.
- **Armazenamento de modelos e dados:** S3 (versionado), com o ECS Fargate lendo o artefato mais recente na inicialização do container.
- **Segurança dos dados:** laudos médicos são dados sensíveis (LGPD/HIPAA-like) — tráfego apenas via HTTPS (ALB + ACM), buckets S3 privados com criptografia em repouso (SSE-KMS), e sem persistência do texto do laudo além do necessário para a resposta da API.
- **Custo:** Fargate e MWAA cobram por uso/tempo ativo, evitando custo fixo de servidores ociosos — adequado para uma carga de triagem que não é constante ao longo do dia.

Essa escolha prioriza serviços gerenciados (menos operação); as próximas etapas do projeto (monitoramento e otimização) serão incorporadas a essa mesma arquitetura sem exigir reescrever a aplicação.

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
- [x] Escolha e documentação do dataset (Medical Abstracts TC Corpus + mapeamento de urgência)
- [x] Pipeline de carregamento e pré-processamento dos dados (`ml/data.py`)
- [x] Treino do modelo (TF-IDF + LogisticRegression, `ml/train.py`)
- [x] Endpoint de classificação (`POST /classify`)
- [x] Dockerfile da API + medição de latência baseline
- [x] Documentação da decisão de arquitetura em nuvem (real-time vs. batch, AWS)

### Próximos passos

- [ ] Configurar o GitHub Actions
- [ ] Criar a DAG de treinamento no Airflow
- [ ] Adicionar métricas do Prometheus
- [ ] Configurar o dashboard do Grafana
- [ ] Converter o modelo para ONNX
- [ ] Comparar a latência dos modelos
- [ ] Gravar o vídeo de apresentação STAR