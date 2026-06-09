import argparse
import random
import sys
import threading
import time
from pathlib import Path

import os

import numpy as np
import pygame
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torchvision import models


# ============== constants =====================

GESTURES      = ['rock', 'paper', 'scissor']
LABEL_TO_GEST = {0: 'rock', 1: 'paper', 2: 'scissor'}
BEATS         = {'rock': 'scissor', 'scissor': 'paper', 'paper': 'rock'}

W, H = 1100, 680

C_BG     = ( 12,  12,  22)
C_WHITE  = (240, 240, 240)
C_DIM    = (110, 110, 120)
C_YELLOW = (255, 210,   0)
C_GREEN  = ( 60, 210,  90)
C_RED    = (215,  55,  55)
C_BLUE   = ( 70, 150, 225)
C_ORANGE = (255, 140,   0)

GEST_COLOR = {'rock': C_BLUE, 'paper': C_GREEN, 'scissor': C_RED}
GEST_TEXT  = {'rock': 'ROCK', 'paper': 'PAPER', 'scissor': 'SCISSORS'}

COUNTDOWN_WORDS  = ['Rock', 'Paper', 'Scissors', 'GO!']
COUNTDOWN_COLORS = [C_BLUE, C_GREEN, C_RED, C_YELLOW]
COUNTDOWN_STEP   = 900    # ms per word

REPLAY_FRAMES   = 16
REPLAY_FRAME_MS = 100
REPLAY_SURF_W   = 768
REPLAY_SURF_H   = 432

# Buffer hard cap — guards against timestamp-spike-driven chunk accumulation
CHUNK_CAP  = 4000
CHUNK_KEEP = 3000


# ============== model =================================================

def build_model(repr_name: str) -> nn.Module:
    channels = {'histogram': 2, 'voxel': 5, 'timesurface': 1}[repr_name]

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = models.resnet18(weights=None)
            self.resnet.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc    = nn.Linear(self.resnet.fc.in_features, 3)
        def forward(self, x):
            return self.resnet(x)

    return _Model()


# ============== event representations =============================================
# Match training pipeline exactly (640×360, downsampled from 1280×720).

def to_histogram(evs, out_h=360, out_w=640, src_h=720, src_w=1280):
    arr = np.zeros((2, out_h, out_w), dtype=np.float32)
    if evs.size == 0:
        return arr
    x = (evs['x'].astype(np.int32) * out_w // src_w)
    y = (evs['y'].astype(np.int32) * out_h // src_h)
    p = evs['p']
    valid = (x >= 0) & (x < out_w) & (y >= 0) & (y < out_h)
    x, y, p = x[valid], y[valid], p[valid]
    on_mask = p == 1
    np.add.at(arr[0], (y[on_mask],  x[on_mask]),  1)
    np.add.at(arr[1], (y[~on_mask], x[~on_mask]), 1)
    return arr


def to_voxel(evs, out_h=360, out_w=640, src_h=720, src_w=1280, bins=5):
    arr = np.zeros((bins, out_h, out_w), dtype=np.float32)
    if evs.size == 0:
        return arr
    x = (evs['x'].astype(np.int32) * out_w // src_w)
    y = (evs['y'].astype(np.int32) * out_h // src_h)
    valid = (x >= 0) & (x < out_w) & (y >= 0) & (y < out_h)
    x, y = x[valid], y[valid]
    t = evs['t'][valid].astype(np.float64)
    p = evs['p'][valid]
    if len(t) == 0:
        return arr
    t_min, t_max = t.min(), t.max()
    if t_max == t_min:
        bin_idx = np.zeros(len(t), dtype=np.int32)
    else:
        t_norm  = (t - t_min) / (t_max - t_min)
        bin_idx = np.clip((t_norm * bins).astype(np.int32), 0, bins - 1)
    weights = np.where(p == 1, 1.0, -1.0).astype(np.float32)
    np.add.at(arr, (bin_idx, y, x), weights)
    return arr


def to_timesurface(evs, out_h=360, out_w=640, src_h=720, src_w=1280):
    arr = np.zeros((1, out_h, out_w), dtype=np.float32)
    if evs.size == 0:
        return arr
    x = (evs['x'].astype(np.int32) * out_w // src_w)
    y = (evs['y'].astype(np.int32) * out_h // src_h)
    valid = (x >= 0) & (x < out_w) & (y >= 0) & (y < out_h)
    x, y = x[valid], y[valid]
    t = evs['t'][valid].astype(np.float64)
    if len(t) == 0:
        return arr
    t_min, t_max = t.min(), t.max()
    surface = np.zeros((out_h, out_w), dtype=np.float64)
    np.maximum.at(surface, (y, x), t)
    mask = surface > 0
    if t_max > t_min:
        arr[0][mask] = ((surface[mask] - t_min) / (t_max - t_min)).astype(np.float32)
    else:
        arr[0][mask] = 1.0
    return arr


REPR_FN = {
    'histogram':   to_histogram,
    'voxel':       to_voxel,
    'timesurface': to_timesurface,
}


# ============== event replay rendering ================================================

def build_replay_frame(evs: np.ndarray, t_end: int) -> pygame.Surface:
    surf = pygame.Surface((REPLAY_SURF_W, REPLAY_SURF_H))
    surf.fill(C_BG)
    if evs.size == 0:
        return surf
    mask      = evs['t'] < t_end
    frame_evs = evs[mask]
    if frame_evs.size == 0:
        return surf
    x  = (frame_evs['x'].astype(np.int32) * REPLAY_SURF_W // 1280).clip(0, REPLAY_SURF_W - 1)
    y  = (frame_evs['y'].astype(np.int32) * REPLAY_SURF_H // 720 ).clip(0, REPLAY_SURF_H - 1)
    p  = frame_evs['p']
    px = pygame.surfarray.pixels3d(surf)
    on = p == 1
    px[x[on],  y[on]]  = C_GREEN
    px[x[~on], y[~on]] = C_RED
    del px
    return surf


def precompute_replay_frames(evs: np.ndarray) -> list:
    """Called from a background thread — never blocks the main loop."""
    if evs.size == 0:
        return [pygame.Surface((REPLAY_SURF_W, REPLAY_SURF_H)) for _ in range(REPLAY_FRAMES)]
    t_min  = int(evs['t'].min())
    t_max  = int(evs['t'].max())
    t_span = max(t_max - t_min, 1)
    return [
        build_replay_frame(evs, t_min + int(t_span * i / REPLAY_FRAMES))
        for i in range(1, REPLAY_FRAMES + 1)
    ]


# ============== thread-safe event buffer ===============================================================

class EventBuffer:
    """
    Keeps the last KEEP_US µs of events.

    _latest_t is strictly monotonic: HAL timestamp spikes (NonMonotonicTimeHigh)
    never decrease it, so time-based pruning always fires correctly.
    A hard chunk cap is a second line of defence against runaway accumulation.
    """
    KEEP_US = 3_000_000  # 3 s

    def __init__(self):
        self._lock     = threading.Lock()
        self._chunks   = []
        self._latest_t = 0

    def push(self, evs: np.ndarray):
        if evs.size == 0:
            return
        with self._lock:
            self._chunks.append(evs.copy())
            new_t = int(evs['t'][-1])
            if new_t > self._latest_t:          # monotonic — never go backward
                self._latest_t = new_t
            cutoff = self._latest_t - self.KEEP_US
            self._chunks = [c for c in self._chunks if int(c['t'][-1]) >= cutoff]
            if len(self._chunks) > CHUNK_CAP:   # hard cap against spike-driven growth
                self._chunks = self._chunks[-CHUNK_KEEP:]

    def latest_t(self) -> int:
        with self._lock:
            return self._latest_t

    def get_window(self, t_start: int, duration_us: int) -> np.ndarray:
        t_end = t_start + duration_us
        with self._lock:
            chunks = list(self._chunks)         # shallow copy outside lock
        chunks = [c for c in chunks
                  if int(c['t'][-1]) >= t_start and int(c['t'][0]) < t_end]
        if not chunks:
            return np.zeros(0, dtype=[('x', '<u2'), ('y', '<u2'), ('p', 'i1'), ('t', '<i8')])
        all_evs = np.concatenate(chunks)
        return all_evs[(all_evs['t'] >= t_start) & (all_evs['t'] < t_end)]


# ============== capture thread ===================================

def capture_worker(buf: EventBuffer, stop: threading.Event):
    """
    Reads live events and pushes into buf.
    Sets bias_diff_on/off = 32 to match the recording session.
    """
    from metavision_core.event_io import EventsIterator

    try:
        mv_iter = EventsIterator(input_path="", delta_t=1_000)
    except TypeError:
        mv_iter = EventsIterator(delta_t=1_000)

    try:
        device = mv_iter.reader.device
        biases = device.get_i_ll_biases()
        biases.set("bias_diff_on",  32)
        biases.set("bias_diff_off", 32)
        print("[capture] biases set: diff_on=32  diff_off=32")
    except Exception as e:
        print(f"[capture] bias config failed (continuing anyway): {e}")

    for evs in mv_iter:
        if stop.is_set():
            break
        buf.push(evs)


# ============== game helpers ============================

def winner_of(player: str, robot: str) -> str:
    if player == robot:
        return 'draw'
    return 'player' if BEATS[player] == robot else 'robot'


def classify(model, buf: EventBuffer, go_t: int,
             offset_us: int, window_us: int,
             repr_name: str, device) -> tuple[str, int, np.ndarray]:
    """
    Returns (gesture_name, event_count, raw_evs).

    Looks backward from buf.latest_t() rather than forward from go_t so that
    HAL timestamp spikes (NonMonotonicTimeHigh) don't land us in an empty window.
    classify() is called after waiting offset_ms + window_ms from GO, so
    the most recent window_us covers exactly the gesture.
    """
    t_now = buf.latest_t()
    evs   = buf.get_window(t_now - window_us, window_us)

    t0 = time.perf_counter()
    arr = REPR_FN[repr_name](evs)
    if repr_name == 'voxel':
        max_val = np.abs(arr).max()
    else:
        max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val
    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred   = int(logits.argmax(1).item())
    t1 = time.perf_counter()

    print(f"[classify] events={evs.size}  inference={1000*(t1-t0):.0f}ms  "
          f"logits={logits.cpu().numpy()}  pred={pred}")
    return LABEL_TO_GEST[pred], evs.size, evs


# ============== pygame UI ============================

class UI:
    def __init__(self, screen):
        self.screen = screen
        self.f_huge  = pygame.font.SysFont('dejavusans', 110, bold=True)
        self.f_big   = pygame.font.SysFont('dejavusans',  68, bold=True)
        self.f_med   = pygame.font.SysFont('dejavusans',  44)
        self.f_small = pygame.font.SysFont('dejavusans',  30)

    def text(self, msg, font, color, cx, cy):
        s = font.render(str(msg), True, color)
        self.screen.blit(s, s.get_rect(center=(cx, cy)))

    def center(self, msg, font, color, cy):
        self.text(msg, font, color, W // 2, cy)

    def bar(self, pct, x, y, bw, bh, fg):
        pygame.draw.rect(self.screen, C_DIM, (x, y, bw, bh), border_radius=bh // 2)
        if pct > 0:
            pygame.draw.rect(self.screen, fg,
                             (x, y, max(int(bw * pct), 4), bh),
                             border_radius=bh // 2)

    def divider(self):
        pygame.draw.line(self.screen, C_DIM, (W // 2, 100), (W // 2, 440), 2)

    def gesture_card(self, label, gesture, cx, cy):
        color = GEST_COLOR.get(gesture, C_WHITE)
        self.text(label,              self.f_med, C_DIM,  cx, cy - 60)
        self.text(GEST_TEXT[gesture], self.f_big, color,  cx, cy + 10)

    def spike_screen(self):
        panel = pygame.Surface((W, H), pygame.SRCALPHA)
        panel.fill((80, 0, 0, 220))
        self.screen.blit(panel, (0, 0))
        self.center('⚠',                                  self.f_huge, C_ORANGE, H // 2 - 180)
        self.center('TIMESTAMP SPIKE',                     self.f_big,  C_ORANGE, H // 2 -  80)
        self.center('0 events captured — result invalid',  self.f_med,  C_WHITE,  H // 2 -  10)
        self.center('Camera HAL emitted NonMonotonicTimeHigh',
                    self.f_small, C_DIM, H // 2 + 55)
        pygame.draw.rect(self.screen, C_ORANGE, (0, 0, W, H), 6)
        self.center('SPACE to try again', self.f_small, C_YELLOW, H - 40)


# ============== main ========================================================

def run(args):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Rock Paper Scissors — Event Camera Demo')
    ui    = UI(screen)
    clock = pygame.time.Clock()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = build_model(args.repr).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model  : {args.model_path}")
    print(f"Repr   : {args.repr}   window={args.window_ms}ms   offset={args.offset_ms}ms")
    print(f"Device : {device}")

    buf  = EventBuffer()
    stop = threading.Event()
    threading.Thread(target=capture_worker, args=(buf, stop), daemon=True).start()

    # Game state
    state          = 'IDLE'
    countdown_t0   = 0
    go_t_event     = 0
    capture_t0     = 0
    robot_choice   = ''
    player_gesture = ''
    winner         = ''
    event_count    = 0
    last_evs       = None
    total_wait_ms  = args.offset_ms + args.window_ms

    # Replay (background thread so main loop never blocks)
    replay_frames      = []
    replay_frame_idx   = 0
    replay_last_t      = 0
    replay_build_done  = threading.Event()

    running = True
    while running:
        clock.tick(60)
        now = pygame.time.get_ticks()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    if state == 'IDLE':
                        state        = 'COUNTDOWN'
                        countdown_t0 = now
                    elif state in ('RESULT', 'SPIKE', 'REPLAY'):
                        state = 'IDLE'
                elif ev.key == pygame.K_r and state == 'RESULT':
                    if last_evs is not None and last_evs.size > 0:
                        _snap = last_evs.copy()
                        replay_frames = []
                        replay_build_done.clear()

                        def _build(e=_snap):
                            replay_frames.extend(precompute_replay_frames(e))
                            replay_build_done.set()

                        threading.Thread(target=_build, daemon=True).start()
                        state = 'BUILDING_REPLAY'

        # ============== state transitions =========================================
        if state == 'COUNTDOWN':
            if (now - countdown_t0) // COUNTDOWN_STEP >= len(COUNTDOWN_WORDS):
                state        = 'CAPTURE'
                go_t_event   = buf.latest_t()
                robot_choice = random.choice(GESTURES)
                capture_t0   = now
                print(f"GO — robot={robot_choice}   t_event={go_t_event}µs")

        elif state == 'CAPTURE':
            if (now - capture_t0) >= total_wait_ms:
                # Render once so "Classifying..." is visible before blocking
                screen.fill(C_BG)
                ui.center('Classifying...', ui.f_big, C_BLUE, H // 2)
                pygame.display.flip()

                try:
                    player_gesture, event_count, last_evs = classify(
                        model, buf, go_t_event,
                        args.offset_ms * 1000,
                        args.window_ms * 1000,
                        args.repr, device,
                    )
                except Exception as e:
                    print(f"[classify error] {type(e).__name__}: {e}")
                    player_gesture, event_count, last_evs = 'rock', 0, None

                winner = winner_of(player_gesture, robot_choice)
                state  = 'SPIKE' if event_count == 0 else 'RESULT'
                print(f"Player={player_gesture}   Robot={robot_choice}   "
                      f"Winner={winner}   events={event_count}"
                      + ("  *** SPIKE ***" if event_count == 0 else ""))

        elif state == 'BUILDING_REPLAY':
            if replay_build_done.is_set():
                replay_frame_idx = 0
                replay_last_t    = now
                state            = 'REPLAY'

        elif state == 'REPLAY':
            if now - replay_last_t >= REPLAY_FRAME_MS:
                replay_frame_idx = (replay_frame_idx + 1) % REPLAY_FRAMES
                replay_last_t    = now

        # ============== draw ======================================================
        screen.fill(C_BG)

        if state == 'IDLE':
            cam_ok = buf.latest_t() > 0
            ui.center('Rock  Paper  Scissors',  ui.f_big,   C_WHITE,  H // 2 - 100)
            ui.center('Event Camera Demo',       ui.f_med,   C_DIM,    H // 2 -  20)
            ui.center('Press SPACE to play',     ui.f_small, C_YELLOW, H // 2 +  65)
            ui.center(
                'Camera live  (ready)' if cam_ok else 'No camera signal — check connection',
                ui.f_small,
                C_GREEN if cam_ok else C_RED,
                H - 38,
            )

        elif state == 'COUNTDOWN':
            idx = min((now - countdown_t0) // COUNTDOWN_STEP, len(COUNTDOWN_WORDS) - 1)
            ui.center(COUNTDOWN_WORDS[idx], ui.f_huge, COUNTDOWN_COLORS[idx], H // 2)

        elif state == 'CAPTURE':
            elapsed = now - capture_t0
            pct     = min(elapsed / total_wait_ms, 1.0)
            ui.center('GO!', ui.f_huge, C_YELLOW, H // 2 - 80)
            ui.bar(pct, W // 2 - 280, H // 2 + 30, 560, 22, C_GREEN)
            ui.center(
                f'Capturing  {args.offset_ms}ms offset + {args.window_ms}ms window',
                ui.f_small, C_DIM, H // 2 + 80,
            )

        elif state == 'SPIKE':
            ui.spike_screen()

        elif state == 'RESULT':
            ui.center('Result', ui.f_med, C_DIM, 52)
            ui.divider()
            ui.gesture_card('YOU',   player_gesture, W // 4,     H // 2 - 30)
            ui.gesture_card('ROBOT', robot_choice,   3 * W // 4, H // 2 - 30)
            ui.center('VS', ui.f_big, C_DIM, H // 2 - 30)

            if winner == 'player':
                msg, col = 'YOU WIN!',     C_GREEN
            elif winner == 'robot':
                msg, col = 'ROBOT WINS!', C_RED
            else:
                msg, col = 'DRAW!',        C_YELLOW
            ui.center(msg, ui.f_huge, col, H - 180)

            ui.center(f'{event_count} events  |  {args.repr}  {args.window_ms}ms',
                      ui.f_small, C_DIM, H - 100)
            ui.center('SPACE to play again   R to replay events',
                      ui.f_small, C_DIM, H - 40)

        elif state == 'BUILDING_REPLAY':
            ui.center('Building replay...', ui.f_big, C_BLUE, H // 2 - 30)
            ui.center(f'{event_count} events', ui.f_small, C_DIM, H // 2 + 40)

        elif state == 'REPLAY':
            if replay_frames:
                surf = replay_frames[replay_frame_idx]
                rx   = (W - REPLAY_SURF_W) // 2
                ry   = 60
                screen.blit(surf, (rx, ry))
                pct = (replay_frame_idx + 1) / REPLAY_FRAMES
                ui.bar(pct, rx, ry + REPLAY_SURF_H + 12, REPLAY_SURF_W, 8, C_BLUE)
                on_lbl  = ui.f_small.render('● ON',  True, C_GREEN)
                off_lbl = ui.f_small.render('● OFF', True, C_RED)
                screen.blit(on_lbl,  (W - 160, ry + REPLAY_SURF_H + 30))
                screen.blit(off_lbl, (W - 80,  ry + REPLAY_SURF_H + 30))

            ui.center(f'Event replay  —  {args.window_ms}ms window  ({event_count} events)',
                      ui.f_small, C_DIM, 30)
            ui.center(f'{player_gesture.upper()}  vs  robot {robot_choice.upper()}',
                      ui.f_small, C_WHITE, H - 60)
            ui.center('SPACE to play again', ui.f_small, C_DIM, H - 25)

        pygame.display.flip()

    stop.set()
    pygame.quit()
    sys.exit(0)


# ============== entry point ============================

if __name__ == '__main__':
    load_dotenv(Path(__file__).parent / '.env')

    ap = argparse.ArgumentParser(description='RPS event camera demo')
    ap.add_argument('--window_ms',  type=int, default=50)
    ap.add_argument('--repr',       default='histogram',
                    choices=['histogram', 'voxel', 'timesurface'])
    ap.add_argument('--offset_ms',  type=int, default=200,
                    help='ms after GO before capture window (default 200 = human reaction time)')
    args = ap.parse_args()

    sliding_root = os.getenv("SLIDING_ROOT")
    if not sliding_root:
        print("SLIDING_ROOT not set in .env")
        sys.exit(1)

    args.model_path = (
        Path(sliding_root) / args.repr
        / f"{args.window_ms}ms" / "merged"
        / f"model_{args.repr}_{args.window_ms}ms_best.pth"
    )

    if not args.model_path.exists():
        print(f"Model not found: {args.model_path}")
        sys.exit(1)

    run(args)