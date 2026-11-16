#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rastreamento de Objetos em Vídeo — Parte Prática
=================================================

Dois modos de rastreamento, para comparar as duas abordagens vistas na apresentação:

  1) Modo "yolo"     -> Tracking-by-Detection (YOLO detecta + ByteTrack/BoT-SORT associa IDs)
                        Detecta e rastreia VÁRIOS objetos automaticamente, com IDs persistentes.

  2) Modo "classico" -> Trackers clássicos do OpenCV (CSRT, KCF, MIL)
                        O usuário seleciona o(s) objeto(s) no 1º frame e o algoritmo
                        acompanha a aparência dele, sem rede neural de detecção.

Na tela são exibidos: bounding boxes, ID de cada objeto, classe/confiança (YOLO),
trilha (trajetória recente) e FPS de processamento.

Exemplos:
    python rastreador.py --modo yolo --fonte 0                      # webcam
    python rastreador.py --modo yolo --fonte video.mp4 --classes 0  # só pessoas
    python rastreador.py --modo classico --tracker CSRT --fonte video.mp4
    python rastreador.py --modo yolo --fonte video.mp4 --salvar saida.mp4

Teclas durante a execução:
    q / ESC  -> sair
    ESPAÇO   -> pausar / continuar
    s        -> (modo clássico) selecionar novos objetos no frame atual
"""

import argparse
import sys
import time
from collections import defaultdict, deque

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Utilidades de desenho
# --------------------------------------------------------------------------- #
def cor_por_id(track_id: int) -> tuple:
    """Gera uma cor BGR fixa e bem distinta para cada ID (mesmo ID = mesma cor)."""
    rng = np.random.default_rng(int(track_id) * 7919)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def desenhar_caixa(frame, x1, y1, x2, y2, track_id, texto_extra=""):
    """Desenha a bounding box com um rótulo 'ID n | extra' acima dela."""
    cor = cor_por_id(track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

    rotulo = f"ID {track_id}" + (f" | {texto_extra}" if texto_extra else "")
    (tw, th), base = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    y_txt = max(y1, th + base + 4)
    x_txt = max(0, min(x1, frame.shape[1] - tw - 6))   # mantém o rótulo dentro da imagem
    cv2.rectangle(frame, (x_txt, y_txt - th - base - 4), (x_txt + tw + 6, y_txt), cor, -1)
    cv2.putText(frame, rotulo, (x_txt + 3, y_txt - base - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def desenhar_trilha(frame, pontos, track_id):
    """Desenha a trajetória recente (centro da caixa) de um objeto."""
    if len(pontos) < 2:
        return
    cor = cor_por_id(track_id)
    pts = np.array(pontos, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=False, color=cor, thickness=2)


def desenhar_painel(frame, fps, modo, n_objetos):
    """Painel no canto superior esquerdo com FPS, modo e nº de objetos."""
    linhas = [f"FPS: {fps:5.1f}", f"Modo: {modo}", f"Objetos: {n_objetos}"]
    cv2.rectangle(frame, (8, 8), (250, 18 + 24 * len(linhas)), (0, 0, 0), -1)
    for i, txt in enumerate(linhas):
        cor = (0, 255, 0) if i == 0 else (255, 255, 255)
        cv2.putText(frame, txt, (16, 32 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2, cv2.LINE_AA)


class MedidorFPS:
    """FPS suavizado por média móvel exponencial (evita número 'pulando')."""

    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.fps = 0.0
        self.t_anterior = time.perf_counter()

    def atualizar(self):
        agora = time.perf_counter()
        dt = agora - self.t_anterior
        self.t_anterior = agora
        if dt > 0:
            inst = 1.0 / dt
            self.fps = inst if self.fps == 0 else self.alpha * self.fps + (1 - self.alpha) * inst
        return self.fps


# --------------------------------------------------------------------------- #
# Modo 1: YOLO + ByteTrack (Tracking-by-Detection)
# --------------------------------------------------------------------------- #
class RastreadorYOLO:
    def __init__(self, modelo="yolov8n.pt", tracker="bytetrack.yaml",
                 conf=0.35, classes=None, tamanho_trilha=30):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("[ERRO] Pacote 'ultralytics' não instalado. Rode: pip install ultralytics\n"
                     "       Ou use o modo clássico: --modo classico")
        print(f"[INFO] Carregando modelo {modelo} (baixa automaticamente na 1ª vez)...")
        self.modelo = YOLO(modelo)
        self.tracker = tracker
        self.conf = conf
        self.classes = classes
        self.trilhas = defaultdict(lambda: deque(maxlen=tamanho_trilha))

    def processar(self, frame):
        # persist=True mantém o estado do tracker entre frames -> IDs persistentes
        resultado = self.modelo.track(frame, persist=True, tracker=self.tracker,
                                      conf=self.conf, classes=self.classes,
                                      verbose=False)[0]
        caixas = resultado.boxes
        if caixas is None or caixas.id is None:   # nenhum objeto confirmado neste frame
            return 0

        xyxy = caixas.xyxy.cpu().numpy().astype(int)
        ids = caixas.id.cpu().numpy().astype(int)
        cls = caixas.cls.cpu().numpy().astype(int)
        confs = caixas.conf.cpu().numpy()

        for (x1, y1, x2, y2), tid, c, cf in zip(xyxy, ids, cls, confs):
            centro = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.trilhas[tid].append(centro)
            desenhar_trilha(frame, self.trilhas[tid], tid)
            desenhar_caixa(frame, x1, y1, x2, y2, tid,
                           f"{resultado.names[c]} {cf:.2f}")
        return len(ids)


# --------------------------------------------------------------------------- #
# Modo 2: Trackers clássicos do OpenCV (CSRT / KCF / MIL)
# --------------------------------------------------------------------------- #
def criar_tracker_classico(nome: str):
    """Cria o tracker compatível com várias versões do OpenCV (API nova e legacy)."""
    nome = nome.upper()
    fabrica = f"Tracker{nome}_create"
    for modulo in (cv2, getattr(cv2, "legacy", None)):
        if modulo is not None and hasattr(modulo, fabrica):
            return getattr(modulo, fabrica)()
    sys.exit(f"[ERRO] Tracker {nome} indisponível nesta instalação do OpenCV.\n"
             "       Instale a versão com módulos extras: pip install opencv-contrib-python")


class RastreadorClassico:
    def __init__(self, tipo="CSRT", tamanho_trilha=30):
        self.tipo = tipo.upper()
        self.objetos = []          # lista de [id, tracker, ativo]
        self.proximo_id = 1
        self.trilhas = defaultdict(lambda: deque(maxlen=tamanho_trilha))

    def adicionar(self, frame, rois):
        """rois: lista de (x, y, w, h) selecionados pelo usuário."""
        for roi in rois:
            x, y, w, h = [int(v) for v in roi]
            if w <= 0 or h <= 0:
                continue
            trk = criar_tracker_classico(self.tipo)
            trk.init(frame, (x, y, w, h))
            self.objetos.append([self.proximo_id, trk, True])
            self.proximo_id += 1

    def processar(self, frame):
        ativos = 0
        for obj in self.objetos:
            tid, trk, _ = obj
            ok, caixa = trk.update(frame)
            obj[2] = ok
            if ok:
                x, y, w, h = [int(v) for v in caixa]
                self.trilhas[tid].append((x + w // 2, y + h // 2))
                desenhar_trilha(frame, self.trilhas[tid], tid)
                desenhar_caixa(frame, x, y, x + w, y + h, tid, self.tipo)
                ativos += 1
            else:
                # Falha típica: oclusão, saída de cena ou mudança brusca de aparência
                cv2.putText(frame, f"ID {tid}: objeto perdido", (10, frame.shape[0] - 15 - 25 * (tid % 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return ativos


def selecionar_rois(janela, frame):
    print("[INFO] Desenhe uma caixa em volta de cada objeto e pressione ENTER/ESPAÇO.\n"
          "       Depois de selecionar todos, pressione ESC para começar.")
    rois = cv2.selectROIs(janela, frame, showCrosshair=True, fromCenter=False)
    return [tuple(r) for r in rois] if len(rois) else []


# --------------------------------------------------------------------------- #
# Programa principal
# --------------------------------------------------------------------------- #
def abrir_fonte(fonte: str):
    alvo = int(fonte) if fonte.isdigit() else fonte   # "0" -> webcam
    cap = cv2.VideoCapture(alvo)
    if not cap.isOpened():
        sys.exit(f"[ERRO] Não foi possível abrir a fonte de vídeo: {fonte}")
    return cap


def main():
    ap = argparse.ArgumentParser(description="Rastreamento de objetos em vídeo (YOLO+ByteTrack ou CSRT/KCF).")
    ap.add_argument("--modo", choices=["yolo", "classico"], default="yolo",
                    help="yolo = detecção+rastreamento automático; classico = CSRT/KCF/MIL com seleção manual")
    ap.add_argument("--fonte", default="0", help="Caminho do vídeo ou índice da webcam (padrão: 0)")
    ap.add_argument("--modelo", default="yolov8n.pt", help="Pesos do YOLO (ex.: yolov8n.pt, yolo11n.pt)")
    ap.add_argument("--rastreador-yolo", default="bytetrack.yaml",
                    choices=["bytetrack.yaml", "botsort.yaml"], help="Algoritmo de associação no modo yolo")
    ap.add_argument("--conf", type=float, default=0.35, help="Confiança mínima das detecções (modo yolo)")
    ap.add_argument("--classes", type=int, nargs="*", default=None,
                    help="IDs de classes COCO a rastrear (0=pessoa, 2=carro, 32=bola...). Padrão: todas")
    ap.add_argument("--tracker", default="CSRT", choices=["CSRT", "KCF", "MIL"],
                    help="Tracker clássico (modo classico)")
    ap.add_argument("--roi", type=int, nargs=4, action="append", metavar=("X", "Y", "W", "H"),
                    help="(modo classico) caixa inicial sem seleção manual; pode repetir")
    ap.add_argument("--largura", type=int, default=960, help="Redimensiona o frame para esta largura (0 = original)")
    ap.add_argument("--salvar", default=None, help="Salva o vídeo processado (ex.: saida.mp4)")
    ap.add_argument("--sem-janela", action="store_true", help="Não abre janela (útil só para gerar o vídeo salvo)")
    args = ap.parse_args()

    cap = abrir_fonte(args.fonte)
    janela = "Rastreamento de Objetos - PDI"
    mostrar = not args.sem_janela
    if mostrar:
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    def ler_frame():
        ok, f = cap.read()
        if ok and args.largura and f.shape[1] != args.largura:
            escala = args.largura / f.shape[1]
            f = cv2.resize(f, (args.largura, int(f.shape[0] * escala)))
        return ok, f

    ok, frame = ler_frame()
    if not ok:
        sys.exit("[ERRO] Não foi possível ler o primeiro frame.")

    # ---- Inicialização do rastreador escolhido ----
    if args.modo == "yolo":
        rastreador = RastreadorYOLO(args.modelo, args.rastreador_yolo, args.conf, args.classes)
        nome_modo = f"YOLO + {args.rastreador_yolo.split('.')[0]}"
    else:
        rastreador = RastreadorClassico(args.tracker)
        nome_modo = f"Classico ({args.tracker})"
        rois = args.roi if args.roi else (selecionar_rois(janela, frame) if mostrar else [])
        if not rois:
            sys.exit("[ERRO] Nenhum objeto selecionado.")
        rastreador.adicionar(frame, rois)

    # ---- Gravação opcional ----
    gravador = None
    if args.salvar:
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        h, w = frame.shape[:2]
        gravador = cv2.VideoWriter(args.salvar, cv2.VideoWriter_fourcc(*"mp4v"), fps_video, (w, h))

    medidor = MedidorFPS()
    pausado = False
    print("[INFO] Rodando. Teclas: q/ESC = sair | ESPAÇO = pausar | s = nova seleção (clássico)")

    while True:
        if not pausado:
            n = rastreador.processar(frame)            # <- detecção/associação ou update do tracker
            desenhar_painel(frame, medidor.atualizar(), nome_modo, n)
            if gravador is not None:
                gravador.write(frame)
            if mostrar:
                cv2.imshow(janela, frame)

        tecla = cv2.waitKey(1) & 0xFF if mostrar else -1
        if tecla in (ord("q"), 27):
            break
        if tecla == ord(" "):
            pausado = not pausado
        if tecla == ord("s") and args.modo == "classico":
            rastreador.adicionar(frame, selecionar_rois(janela, frame))

        if not pausado:
            ok, frame = ler_frame()
            if not ok:
                print("[INFO] Fim do vídeo.")
                break

    cap.release()
    if gravador is not None:
        gravador.release()
        print(f"[INFO] Vídeo salvo em: {args.salvar}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
