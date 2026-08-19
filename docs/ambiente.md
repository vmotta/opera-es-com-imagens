# Preparação do ambiente

## 1. Instale o Python

Use Python 3.10 ou superior. Confirme:

```bash
python --version
```

No Windows, o comando pode ser `py --version`.

## 2. Crie um ambiente isolado

Um ambiente virtual é como uma caixa exclusiva para os materiais desta disciplina: versões instaladas aqui não bagunçam outros projetos.

```bash
python -m venv .venv
```

Ativação no Linux/macOS:

```bash
source .venv/bin/activate
```

Ativação no Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Instale as dependências

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 4. Faça o teste mínimo

```bash
python -c "import cv2, numpy; print(cv2.__version__)"
python exemplos/cap01_fundamentos.py
```

Devem surgir arquivos PNG em `outputs/cap01/`.

## Problemas comuns

**`ModuleNotFoundError: No module named 'cv2'`:** o ambiente não está ativado ou as dependências foram instaladas em outro Python. Compare `python -m pip --version` com `python --version`.

**A imagem não abre:** caminhos relativos partem da pasta em que o comando foi executado. Os exemplos geram entradas sintéticas justamente para evitar esse bloqueio inicial.

**`cv2.imshow` falha em servidor:** este repositório usa `cv2.imwrite`; abra o PNG salvo. A variante `opencv-python-headless` não inclui interface gráfica.

**Pesos ausentes:** capítulos avançados fornecem simulação ou encerram com uma mensagem indicando o caminho esperado. Não renomeie um modelo incompatível para “fazê-lo funcionar”: arquitetura e pesos precisam corresponder.
