#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maze_runner.py
端末で動く迷路ジェネレーター＆ソルバー。
- 標準ライブラリのみ
- 再現性のための --seed
- 解答のアニメーション表示 (--animate, --delay)
- カラーON/OFF (--no-color)

使い方:
    python maze_runner.py --width 31 --height 17 --animate --delay 0.01 --seed 42
    python maze_runner.py -w 21 -H 11 --no-color --no-solve

サイズ指定は「セル数」です。表示は 2*h+1 x 2*w+1 の文字グリッドになります。
"""

from __future__ import annotations
import argparse
import os
import random
import sys
import time
from collections import deque
from typing import List, Tuple, Optional, Iterable, Dict

# === ANSI helpers ===
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
FG_GREEN = "\x1b[32m"
FG_CYAN = "\x1b[36m"
FG_YELLOW = "\x1b[33m"
FG_MAGENTA = "\x1b[35m"
FG_RED = "\x1b[31m"


def supports_color(stream) -> bool:
    if not stream.isatty():
        return False
    if os.name == "nt":
        return True  # 最近のWindowsならOK
    return True


# === Maze core ===
Cell = Tuple[int, int]  # (x, y) in cell coords


def carve_maze(width: int, height: int, rng: random.Random,
               on_visit: Optional[callable] = None) -> List[List[int]]:
    """Recursive backtracker で迷路を掘る。
    セルごとのビットフラグで通路を保持。
    ビット: 1=N, 2=E, 4=S, 8=W （NESW）
    """
    N, E, S, W = 1, 2, 4, 8
    DX = {E: 1, W: -1, N: 0, S: 0}
    DY = {E: 0, W: 0, N: -1, S: 1}
    OPP = {E: W, W: E, N: S, S: N}

    grid = [[0 for _ in range(width)] for _ in range(height)]
    stack: List[Cell] = [(0, 0)]
    visited = [[False] * width for _ in range(height)]
    visited[0][0] = True

    while stack:
        x, y = stack[-1]
        dirs = [N, E, S, W]
        rng.shuffle(dirs)
        carved = False
        for d in dirs:
            nx, ny = x + DX[d], y + DY[d]
            if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                grid[y][x] |= d
                grid[ny][nx] |= OPP[d]
                visited[ny][nx] = True
                stack.append((nx, ny))
                carved = True
                if on_visit:
                    on_visit((nx, ny))
                break
        if not carved:
            stack.pop()
    return grid


def maze_to_chars(grid: List[List[int]]) -> List[List[str]]:
    """セルベースの迷路を文字グリッド(壁=#, 通路=空白)に変換。"""
    h = len(grid)
    w = len(grid[0]) if h else 0
    H, W = 2 * h + 1, 2 * w + 1
    chars = [["#" for _ in range(W)] for _ in range(H)]

    # 開口部: 左上を入口、右下を出口
    chars[1][0] = " "
    chars[H - 2][W - 1] = " "

    # セル中心を空白にし、通路に応じて壁を壊す
    for y in range(h):
        for x in range(w):
            cx, cy = 2 * x + 1, 2 * y + 1
            chars[cy][cx] = " "
            val = grid[y][x]
            if val & 1:  # N
                chars[cy - 1][cx] = " "
            if val & 2:  # E
                chars[cy][cx + 1] = " "
            if val & 4:  # S
                chars[cy + 1][cx] = " "
            if val & 8:  # W
                chars[cy][cx - 1] = " "
    return chars


def solve_maze(grid: List[List[int]]) -> List[Cell]:
    """幅優先で (0,0) -> (w-1,h-1) の最短経路（セル座標列）を返す。"""
    h = len(grid)
    w = len(grid[0]) if h else 0
    start: Cell = (0, 0)
    goal: Cell = (w - 1, h - 1)
    N, E, S, W = 1, 2, 4, 8
    DX = {E: 1, W: -1, N: 0, S: 0}
    DY = {E: 0, W: 0, N: -1, S: 1}

    q = deque([start])
    prev: Dict[Cell, Optional[Cell]] = {start: None}

    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            break
        for d in (N, E, S, W):
            if grid[y][x] & d:
                nx, ny = x + DX[d], y + DY[d]
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in prev:
                    prev[(nx, ny)] = (x, y)
                    q.append((nx, ny))

    path: List[Cell] = []
    cur: Optional[Cell] = goal if goal in prev else None
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


# === Rendering ===

def clear():
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def draw(chars: List[List[str]], *, color: bool = True):
    lines = ["".join(row) for row in chars]
    if color:
        # 壁を薄め、入口/出口を少し色付け
        colored = []
        for r, line in enumerate(lines):
            line2 = line.replace("#", DIM + "#" + RESET)
            if r == 1:
                line2 = line2[:0] + FG_GREEN + line2[0] + RESET + line2[1:]
            colored.append(line2)
        # 出口の列末の1文字を色付け
        if colored:
            r = len(colored) - 2
            line = colored[r]
            colored[r] = line[:-1] + FG_RED + line[-1] + RESET
        print("\n".join(colored))
    else:
        print("\n".join(lines))


def overlay_path(chars: List[List[str]], path_cells: Iterable[Cell], *,
                 color: bool = True,
                 step: Optional[float] = None):
    """path を文字グリッド上に重ね書き。step を与えるとアニメーション。"""
    def cell_to_xy(c: Cell) -> Tuple[int, int]:
        x, y = c
        return 2 * x + 1, 2 * y + 1

    prev_xy: Optional[Tuple[int, int]] = None
    for idx, cell in enumerate(path_cells):
        x, y = cell_to_xy(cell)
        if color:
            chars[y][x] = FG_CYAN + "•" + RESET
        else:
            chars[y][x] = "."
        if prev_xy is not None:
            px, py = prev_xy
            mx, my = (x + px) // 2, (y + py) // 2
            chars[my][mx] = " " if not color else FG_CYAN + "·" + RESET
        prev_xy = (x, y)
        if step is not None and step > 0:
            clear()
            draw(chars, color=color)
            time.sleep(step)


# === CLI ===

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Terminal maze generator & solver")
    p.add_argument("--width", "-w", type=int, default=31, help="cells in X (odd recommended)")
    p.add_argument("--height", "-H", type=int, default=17, help="cells in Y (odd recommended)")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    p.add_argument("--no-solve", action="store_true", help="do not solve; just show the maze")
    p.add_argument("--animate", action="store_true", help="animate the solving path")
    p.add_argument("--delay", type=float, default=0.005, help="animation delay seconds per step")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return p.parse_args(argv)


def main(argv: List[str] = None) -> int:
    ns = parse_args(argv or sys.argv[1:])

    if ns.width < 2 or ns.height < 2:
        print("width/height は 2 以上にしてください。", file=sys.stderr)
        return 2

    rng = random.Random(ns.seed)

    # 生成
    chars: List[List[str]]

    def _on_visit(_cell: Cell):
        # 生成中アニメーションにしてもよいが、デフォルトは高速化のため抑制
        pass

    grid = carve_maze(ns.width, ns.height, rng, on_visit=_on_visit)
    chars = maze_to_chars(grid)

    # ラベル（S/G）
    sy, sx = 1, 0
    gy, gx = len(chars) - 2, len(chars[0]) - 1
    chars[sy][sx] = "S"
    chars[gy][gx] = "G"

    color = supports_color(sys.stdout) and not ns.no_color

    clear()
    draw(chars, color=color)

    if not ns.no_solve:
        path = solve_maze(grid)
        if len(path) <= 1:
            return 0
        # S/G を避けて重ねる
        inner_path = path[1:-1]
        overlay_path(chars, inner_path, color=color,
                     step=(ns.delay if ns.animate else None))
        if not ns.animate:
            clear()
            draw(chars, color=color)

        # 経路長を表示
        length = len(path) - 1
        if color:
            print(f"\n{BOLD}{FG_YELLOW}Shortest path length: {length}{RESET}")
        else:
            print(f"\nShortest path length: {length}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
