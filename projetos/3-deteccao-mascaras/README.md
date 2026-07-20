# Projeto 3 — Detecção de Máscaras Faciais (YOLO)

## 💻 O Desafio Técnico

Desenvolva um modelo de **detecção de objetos** capaz de identificar, em uma
imagem com rostos, se cada pessoa está **usando máscara corretamente**, **sem
máscara**, ou **usando a máscara de forma incorreta** — localizando cada rosto
com uma bounding box.

Diferente dos Projetos 1 e 2 (onde você constrói uma CNN do zero), aqui o
objetivo é **adaptar e otimizar um framework de detecção real para Edge AI** —
uma competência bastante prática no dia a dia de Visão Computacional Embarcada,
já que a imensa maioria das aplicações de detecção em produção parte de um
modelo pré-treinado, não de uma arquitetura construída do zero.

> ⚠️ **Exceção importante:** ao contrário dos Projetos 1 e 2, aqui o uso de
> **pesos pré-treinados é permitido e esperado** (fine-tuning). Isso é
> intencional — este projeto avalia uma competência diferente: adaptar,
> treinar e exportar um framework de detecção real para o seu dataset.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**fine-tuning → validação → exportação → otimização para edge**

## 🎯 Conjunto de Dados

Este projeto já vem com um dataset **pronto para uso**, na pasta [`dataset/`](dataset/):
o **Face Mask Detection Dataset** ([Kaggle, andrewmvd](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection),
licença **CC0 1.0** — domínio público), já convertido do formato original (Pascal VOC)
para o formato esperado pelo Ultralytics YOLO.

- **853 imagens** de rostos, com bounding boxes anotadas
- **3 classes:** `with_mask`, `without_mask`, `mask_weared_incorrect`
- Já dividido em treino (~80%) e validação (~20%)
- ⚠️ O dataset é **desbalanceado** — a classe `mask_weared_incorrect` tem
  significativamente menos exemplos que as outras duas. Isso é uma
  característica real de datasets de detecção e não é um bug — comente esse
  ponto no seu relatório se perceber o modelo com dificuldade nessa classe.

Você **não precisa** baixar nada do Kaggle nem escrever código de conversão de
anotações — isso já está pronto em `dataset/`. Seu trabalho começa direto no
fine-tuning do modelo.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Fine-tuning do Modelo (`train_model.py`)

Implemente, usando a biblioteca **Ultralytics** (YOLO):

- Carregamento do modelo pré-treinado **YOLO11n** (`YOLO("yolo11n.pt")`) —
  esta é a única exceção à regra de "sem modelos pré-treinados" do processo
  seletivo, válida especificamente para este projeto
- Fine-tuning no dataset fornecido (`dataset/data.yaml`), em **CPU**, com um
  número de épocas modesto (ex: 15-30 — YOLO converge relativamente rápido
  em fine-tuning, mesmo em CPU)
- Ao final do treino, copie os pesos resultantes (`runs/detect/train/weights/best.pt`)
  para a raiz desta pasta, com o nome **`model.pt`**

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.pt` treinado
- Exportação para **TensorFlow Lite** via `model.export(format="tflite")`
  (a Ultralytics gera automaticamente um arquivo `model.tflite` na mesma pasta)

> 💡 Na primeira execução, a Ultralytics pode instalar automaticamente
> dependências extras necessárias para a exportação (isso é esperado e pode
> levar alguns minutos).

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.pt`) usando `YOLO("model.tflite", task="detect")`
- Execução de inferência em pelo menos **5 imagens** de `dataset/images/val/`,
  **uma de cada vez** — o `model.tflite` exportado aceita apenas 1 imagem por
  chamada (batch=1), que é aliás o cenário real de uso em edge
- Exibição no terminal, para cada imagem, do número de detecções encontradas

> 💡 O Ultralytics salva automaticamente as imagens anotadas com as caixas
> preditas em `runs/detect/...` (pasta já ignorada pelo `.gitignore` — não
> precisa, nem deve, ser commitada). Abra essas imagens localmente pra conferir
> visualmente as predições antes de escrever o relatório.
>
> 💡 Essa etapa existe porque uma métrica agregada (mAP) pode esconder
> problemas que só aparecem olhando exemplos individuais — especialmente dado
> o desbalanceamento de classes deste dataset.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos nem a estrutura de `dataset/`.

```
projetos/3-deteccao-mascaras/
├── train_model.py         # ✏️ Fine-tuning do modelo
├── optimize_model.py      # ✏️ Exportação e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.pt               # 🤖 Gerado por você — deve ser commitado
├── model.tflite            # ⚡ Gerado por você — deve ser commitado
├── README.md               # 📝 Este arquivo (também usado como relatório)
└── dataset/                # 📦 Dataset já pronto (não modificar)
    ├── data.yaml
    ├── images/{train,val}/
    └── labels/{train,val}/
```

## ⚠️ Restrições e Considerações de Engenharia

- Modelo base: **YOLO11n** (variante *nano*, indicada para CPU/edge) — não use
  variantes maiores (s/m/l/x)
- Treinamento apenas em CPU
- Fine-tuning é permitido e esperado (única exceção às regras gerais do processo seletivo)
- **Não é esperada detecção perfeita**, especialmente na classe minoritária
  (`mask_weared_incorrect`) — o objetivo é demonstrar que o pipeline completo
  (fine-tuning → validação → exportação) funciona corretamente
- O tempo de treinamento e exportação deste projeto tende a ser **maior** que
  o dos Projetos 1 e 2 — reserve tempo extra para rodar localmente antes de enviar

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração de `model.pt` e `model.tflite`
- **Qualidade do modelo** — mAP50 no conjunto de validação acima do mínimo esperado
- **Edge AI** — exportação correta para `.tflite`
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Kelvin Oliveira Fernandes

### 1️⃣ Resumo da Abordagem

- **Épocas: 20:** (dentro da faixa 15-30 sugerida para CPU)
- **Tamanho de imagem:** 640x640 (padrão do YOLO11n, mesmo tamanho do pré-treino)
- **Batch size:** 8 (adequado para CPU com 8+ GB de RAM) 
- **Device:** CPU (conforme requisito do projeto)
- **Desbalanceamento:** O dataset é desbalanceado — a classe `mask_weared_incorrect` representa apenas ~2.6% das instâncias de validação (19 de 726), enquanto `with_mask` domina com ~82% (593 instâncias). Para mitigar esse problema, foi utilizado o parâmetro `cls_pw=0.5`, que aplica pesos inversamente proporcionais à frequência de cada classe na loss de classificação:
  - Cálculo: `peso_classe = (1 / n_instâncias) ^ 0.5`
  - Efeito: a classe minoritária recebeu peso ~5.6× maior que `with_mask`, forçando o modelo a prestar mais atenção nela durante o treino
- Com `cls_pw=0.5`, o mAP50 da classe mask_weared_incorrect subiu de **0.549 → 0.611 (+11.3%)** sem degradar as classes majoritárias (`with_mask` manteve 0.964), e o mAP50 geral subiu de 0.773 para 0.793.

### 2️⃣ Bibliotecas Utilizadas

| Biblioteca | Versão | Finalidade |
| --- | --- | --- |
| **ultralytics** | 8.4.99 | Framework YOLO — treino, exportação, inferência |
| **torch** | 2.13.0 | Backend de deep learning (PyTorch) |
| **torchvision** | 0.28.0 | Operações de visão computacional |
| **litert-torch** | 0.9.1 | Exportação para o formato LiteRT/TFLite |
| **ai-edge-litert** | 2.1.5 | Runtime LiteRT para inferência em edge |
| **opencv-python** | 5.0.0.93 | Manipulação de imagens |
| **numpy** | 2.4.6 | Operações numéricas |

### 3️⃣ Técnica de Otimização do Modelo

O modelo foi exportado para LiteRT (formato `.tflite`) com **quantização INT8 estática** via `quantize=8`. O processo:
1. Converte o grafo PyTorch → ONNX → LiteRT
2. Executa calibração no dataset de validação (`data="dataset/data.yaml"`) para determinar thresholds de quantização por camada
3. Reduz os pesos de FP32 (4 bytes) para INT8 (1 byte), resultando em um modelo ~4× menor

A técnica utilizada é **post-training quantization (PTQ) estática com calibração**, que oferece boa relação compressão × acurácia sem necessidade de re-treino.

### 4️⃣ Resultados Obtidos

#### mAP por classe (validação):

| Classe | Imagens | Instâncias | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| **Todas** | 170 | 726 | 0.793 | 0.549 |
| **with_mask** | 149 | 593 | 0.964 | 0.674 |
| **without_mask** | 57 | 114 | 0.803 | 0.529 |
| **mask_weared_incorrect** | 15 | 19 | 0.611 | 0.445 |

#### Impacto do cls_pw=0.5:

| Classe | Sem cls_pw | Com cls_pw | Ganho |
| --- | --- | --- | --- |
| **Todas** | 0.773 | 0.793 | +2.6% |
| **mask_weared_incorrect** | 0.549 | 0.611 | +11.3% |

#### Tamanho dos artefatos:

| Arquivo | Precisão | Tamanho |
| --- | --- | --- |
| **model.pt** | FP32 | 5.3 MB |
| **model.tflite** | INT8 | 3.0 MB |

### 5️⃣ Comentários Adicionais (Opcional)

**Dificuldades encontradas:**
- **Auto-update travado:** durante a exportação para TFLite, a instalação automática de `litert-torch` e `ai-edge-litert` ficou parada por ~15 minutos. Foi necessário cancelar e instalar as dependências manualmente.
- **Incompatibilidade torch/torchvision:** o `run_inference.py` quebrou com `RuntimeError: operator torchvision::nms does not exist.` O `requirements.txt` só especificava `ultralytics>=8.4`, e o pip resolveu versões incompatíveis entre PyTorch 2.13 e torchvision 0.28. Resolvido com `pip install --upgrade torch torchvision`.
- **Incompatibilidade litert-torch/torch na CI:** o workflow falhava na re-execução do `optimize_model.py` porque o auto-update do Ultralytics fazia downgrade do `torch 2.13.0 → 2.12.1`, quebrando a exportação LiteRT. Resolvido adicionando `litert-torch>=0.9.0` e `ai-edge-litert>=2.1.4` ao `requirements.txt` para que todas as dependências fossem instaladas juntas, evitando o auto-update com downgrade.

Decisões técnicas:
- Uso de `cls_pw=0.5` para mitigar o desbalanceamento — a classe `mask_weared_incorrect` melhorou de 0.549 → 0.611 (+11.3%) sem penalizar as demais.
- `batch=8` e `epochs=20` como equilíbrio entre tempo de treino em CPU e qualidade do modelo.

Limitações do modelo:
- A classe `mask_weared_incorrect` (apenas 19 instâncias na validação) permanece com desempenho inferior (mAP50 0.611 vs 0.964 das máscaras corretas) — limitação inerente ao tamanho reduzido da amostra.
- A exportação INT8 com LiteRT gera o arquivo como `model_int8.tflite`, exigindo renomeação manual ou `shutil.copy` para o nome esperado (`model.tflite`).

### 6️⃣ Exemplo de Inferência
```
============================================================
Projeto 3 — Inferência com model.tflite (Edge AI)
============================================================

Rodando inferência em 5 amostras usando model.tflite:

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
Loading /home/kelvao/processoseletivoIA/projetos/3-deteccao-mascaras/model.tflite for LiteRT inference...
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Results saved to /home/kelvao/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss105.jpg                         10  [10x with_mask]
Results saved to /home/kelvao/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss107.jpg                          1  [1x with_mask]
Results saved to /home/kelvao/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss11.jpg                          35  [32x with_mask, 2x mask_weared_incorrect, 1x without_mask]
Results saved to /home/kelvao/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss113.jpg                          7  [7x with_mask]
Results saved to /home/kelvao/processoseletivoIA/projetos/3-deteccao-mascaras/runs/detect/inferencia_exemplos/predicoes
maksssksksss12.jpg                          21  [17x with_mask, 4x without_mask]
----------------------------------------------------------------------
TOTAL                                       74

✅ Imagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/
   (Abra essa pasta para verificar visualmente as bounding boxes preditas)
```
**Observações:** 
- Em algumas imagens, as labels ficaram sobrepostas aos rostos, mas isso é apenas visual e não impacta as detecções.
- Na imagem maksssksksss11.jpg, o modelo detectou 35 rostos — 32 com máscara correta, 2 com máscara incorreta e 1 sem máscara — demonstrando que a classe minoritária está sendo capturada mesmo em cenas densas.

---

## 📄 Créditos do Dataset

Face Mask Detection Dataset — [Kaggle: andrewmvd/face-mask-detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), licença CC0 1.0 (domínio público).
