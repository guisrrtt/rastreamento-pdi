# Rastreamento de Objetos em Vídeo — Parte Prática

Script em Python + OpenCV que rastreia objetos em vídeo (arquivo ou webcam) e mostra na tela:
**bounding boxes, ID de cada objeto, classe/confiança, trilha (trajetória) e FPS**.

Há dois modos, para comparar as abordagens vistas na apresentação:

| Modo | Como funciona | Quando usar |
|------|---------------|-------------|
| `yolo` (padrão) | **Tracking-by-Detection**: o YOLO detecta os objetos em cada frame e o **ByteTrack** (ou BoT-SORT) associa as detecções entre frames, mantendo o mesmo ID | Vários objetos, detecção automática (pessoas, carros, bola…) |
| `classico` | Trackers clássicos do OpenCV (**CSRT**, **KCF**, **MIL**): você marca o objeto no 1º frame e o algoritmo segue a aparência dele | Um objeto qualquer (não precisa ser uma classe conhecida), máquina sem GPU |

---

## 1. Pré-requisitos

- **Python 3.9 a 3.12** (verifique com `python --version`)
- Webcam **ou** um vídeo `.mp4` (sugestão: um vídeo curto de rua/trânsito ou de pessoas andando)
- Internet na primeira execução do modo `yolo` (o modelo `yolov8n.pt`, ~6 MB, é baixado automaticamente)

## 2. Instalação (passo a passo)

```bash
# 1) Entre na pasta do projeto
cd rastreamento

# 2) Crie e ative um ambiente virtual (recomendado)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# 3) Atualize o pip
python -m pip install --upgrade pip

# 4) Instale as dependências
pip install -r requirements.txt
```

Ou, sem o `requirements.txt`:

```bash
pip install opencv-contrib-python numpy ultralytics lap
```

> **Importante:** use `opencv-contrib-python` (e não só `opencv-python`), pois os trackers CSRT e KCF ficam nos módulos extras.
> Se você já tiver `opencv-python` instalado e der conflito: `pip uninstall opencv-python opencv-python-headless` e reinstale o contrib.

## 3. Como executar

### Modo YOLO + ByteTrack (automático)

```bash
# Webcam
python rastreador.py --modo yolo --fonte 0

# Arquivo de vídeo
python rastreador.py --modo yolo --fonte video.mp4

# Só pessoas (classe 0) e carros (classe 2)
python rastreador.py --modo yolo --fonte video.mp4 --classes 0 2

# Trocar o algoritmo de associação para BoT-SORT
python rastreador.py --modo yolo --fonte video.mp4 --rastreador-yolo botsort.yaml

# Salvar o resultado em vídeo (bom para levar pronto para a apresentação)
python rastreador.py --modo yolo --fonte video.mp4 --salvar resultado.mp4
```

### Modo clássico (CSRT / KCF / MIL)

```bash
python rastreador.py --modo classico --tracker CSRT --fonte video.mp4
```

1. O primeiro frame abre parado.
2. Desenhe um retângulo com o mouse em volta do objeto e aperte **ENTER** (repita para mais objetos).
3. Aperte **ESC** para iniciar o rastreamento.

Troque `CSRT` por `KCF` para ver a diferença: KCF é bem mais rápido (FPS maior), CSRT é mais preciso.

### Teclas durante a execução

| Tecla | Ação |
|-------|------|
| `q` ou `ESC` | Sair |
| `ESPAÇO` | Pausar / continuar |
| `s` | (modo clássico) Selecionar novos objetos no frame atual |

### Todas as opções

```bash
python rastreador.py --help
```

| Opção | Padrão | Descrição |
|-------|--------|-----------|
| `--modo` | `yolo` | `yolo` ou `classico` |
| `--fonte` | `0` | Caminho do vídeo ou índice da webcam |
| `--modelo` | `yolov8n.pt` | Pesos do YOLO (`yolov8s.pt`, `yolo11n.pt`…) — maior = mais preciso e mais lento |
| `--rastreador-yolo` | `bytetrack.yaml` | `bytetrack.yaml` ou `botsort.yaml` |
| `--conf` | `0.35` | Confiança mínima das detecções |
| `--classes` | todas | IDs COCO: 0 pessoa, 1 bicicleta, 2 carro, 3 moto, 5 ônibus, 7 caminhão, 32 bola |
| `--tracker` | `CSRT` | `CSRT`, `KCF` ou `MIL` (modo clássico) |
| `--roi X Y W H` | — | Caixa inicial sem seleção manual (modo clássico; pode repetir) |
| `--largura` | `960` | Redimensiona o vídeo (0 = tamanho original). Menor = mais FPS |
| `--salvar` | — | Grava o vídeo processado |
| `--sem-janela` | — | Não abre janela (só gera o arquivo salvo) |

## 4. Como interpretar a tela

- **Caixa colorida**: posição do objeto (bounding box). Cada ID tem sempre a mesma cor.
- **`ID 3 | person 0.84`**: identificador persistente do objeto, classe detectada e confiança do YOLO (0 a 1).
- **Linha atrás do objeto**: trajetória dos últimos 30 frames — mostra que o sistema "lembra" do objeto.
- **FPS**: quantos frames por segundo o computador consegue processar. Abaixo de ~15 o vídeo parece travado.
- **Objetos**: quantos objetos estão sendo rastreados naquele frame.
- No modo clássico, **"ID n: objeto perdido"** aparece quando o tracker falha (oclusão, saída de cena).

## 5. Problemas comuns

| Problema | Solução |
|----------|---------|
| `AttributeError: ... TrackerCSRT_create` | Instale `opencv-contrib-python` (veja a nota da seção 2) |
| `No module named 'lap'` | `pip install lap` |
| `Não foi possível abrir a fonte de vídeo` | Confira o caminho do arquivo, ou tente `--fonte 1` para outra webcam |
| FPS muito baixo | Use `--largura 640`, o modelo `yolov8n.pt` e `--classes` para filtrar; ou o modo clássico com `KCF` |
| Janela não abre (Linux sem interface) | Use `--sem-janela --salvar saida.mp4` e assista ao arquivo |
| IDs trocando quando pessoas se cruzam | Comportamento esperado (é o desafio de oclusão); teste `--rastreador-yolo botsort.yaml` |

> Dica para a apresentação: gere antes um `resultado.mp4` com `--salvar`. Se a demo ao vivo falhar (sem internet, PC lento), você ainda tem o vídeo pronto.
