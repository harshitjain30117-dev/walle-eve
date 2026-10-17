"""Grid utilities: 4-connected BFS shortest paths on a wall mask."""
from collections import deque

import numpy as np

INF = 10**6


def bfs(walls, src):
    """BFS from `src` (x, y). Returns (dist[H, W] int array, parent dict)."""
    H, W = walls.shape
    dist = np.full((H, W), INF, dtype=np.int64)
    parent = {}
    sx, sy = int(src[0]), int(src[1])
    dist[sy, sx] = 0
    q = deque([(sx, sy)])
    while q:
        x, y = q.popleft()
        d = dist[y, x] + 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not walls[ny, nx] and dist[ny, nx] == INF:
                dist[ny, nx] = d
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return dist, parent


def path_from_parent(parent, src, dst):
    """Cells from src to dst inclusive (raises KeyError if unreachable)."""
    src, dst = tuple(src), tuple(dst)
    path = [dst]
    while path[-1] != src:
        path.append(parent[path[-1]])
    path.reverse()
    return path
