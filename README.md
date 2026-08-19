# Operações com Imagens: Python, OpenCV e Visão Computacional

Material didático completo, progressivo e executável para aprender processamento de imagens e visão computacional. O conteúdo parte da representação matricial de uma imagem e chega a CNNs, YOLO, estimativa de pose, reconhecimento facial, OCR, visão estéreo e GANs.

> **Para leitura com fonte ampliada:** a documentação foi preparada com fonte Atkinson Hyperlegible, tamanho mínimo de 20 px, alto contraste, tema claro/escuro e navegação por teclado. Execute `mkdocs serve` e abra `http://127.0.0.1:8000`.

## O que você encontra aqui

- 18 capítulos em português, seguindo a sequência do livro **Operações com Imagens**;
- explicações que conectam intuição, analogia, matemática e implementação;
- códigos comentados que geram as próprias imagens de teste;
- resultados visuais reproduzíveis, sem depender de arquivos secretos;
- exercícios de previsão, implementação e reflexão;
- seção de erros comuns e estratégias de depuração;
- testes automatizados e documentação publicável no GitHub Pages.

## Rota de aprendizagem

| Bloco | Capítulos | Pergunta central |
|---|---:|---|
| Pixels e geometria | 1–2 | Como uma imagem é representada e movimentada? |
| Processamento clássico | 3–6 | Como limpar, separar, medir e selecionar regiões? |
| Detecção e movimento | 7–9 | Como localizar padrões e acompanhar deslocamentos? |
| Inferência com redes | 10–15 | Como consumir modelos e interpretar suas saídas? |
| Construção e geração | 16–18 | Como CNNs aprendem e como detectores modernos operam? |

## Começo rápido

```bash
git clone https://github.com/vmotta/opera-es-com-imagens.git
cd opera-es-com-imagens

python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python exemplos/cap01_fundamentos.py
mkdocs serve
```

As imagens produzidas pelos programas são gravadas em `outputs/capXX/`. Os exemplos não chamam `cv2.imshow()`, portanto também funcionam em servidores, Codespaces e integração contínua.

## Execução por nível

```bash
# Núcleo: capítulos 1 a 15 (simulações incluídas)
python -m pip install -e .

# OCR com Tesseract instalado no sistema
python -m pip install -e ".[ocr]"

# Construção de CNNs e GANs
python -m pip install -e ".[deep]"

# Inferência com Ultralytics YOLO
python -m pip install -e ".[yolo]"
```

Pesos de redes neurais não são versionados: podem ocupar centenas de megabytes e têm licenças próprias. Cada capítulo avançado explica o arquivo esperado, onde colocá-lo e como executar uma simulação sem o modelo.

## Organização

```text
docs/                 capítulos, diagramas e resultados visuais
exemplos/             programas Python executáveis
scripts/              geração de imagens e validações auxiliares
tests/                 testes dos exemplos autocontidos
.github/workflows/    integração contínua e documentação
```

## Validação

```bash
ruff check .
pytest
mkdocs build --strict
```

## Autoria e uso educacional

Material organizado por **Prof. Vinícius da Rocha Motta**, Ifes. As referências acadêmicas e técnicas utilizadas no livro e nesta expansão estão listadas na documentação. Modelos, datasets e bibliotecas externas permanecem sujeitos às respectivas licenças.
