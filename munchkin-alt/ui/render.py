"""Draw the city map from recorded simulator frames (with smooth interpolation between ticks)."""
from __future__ import annotations

import pygame

from environment.constants import GRID_SIZE, DriverStatus, OrderStatus
from ui.widgets import ACCENT, BAD, GOOD, MUTED, PANEL, VIOLET, WARN, text

DRIVER_COLOR = {int(DriverStatus.AVAILABLE): GOOD, int(DriverStatus.TO_RESTAURANT): ACCENT,
                int(DriverStatus.AT_RESTAURANT): WARN, int(DriverStatus.TO_CUSTOMER): VIOLET}
DRIVER_NAME = {0: "Available", 1: "To restaurant", 2: "Waiting at restaurant", 3: "To customer"}
ORDER_COLOR = {int(OrderStatus.PREPARING): WARN, int(OrderStatus.READY): GOOD,
               int(OrderStatus.ASSIGNED): ACCENT, int(OrderStatus.PICKED_UP): VIOLET}
RESTAURANT_COLOR = (255, 140, 66)


class MapView:
    def __init__(self, rect: pygame.Rect, restaurants):
        self.rect = rect
        self.restaurants = restaurants        # list of (x, y)
        self.scale = rect.w / (GRID_SIZE + 0.0)

    def to_px(self, x, y):
        return (int(self.rect.x + x * self.scale), int(self.rect.y + (GRID_SIZE - y) * self.scale))

    def draw(self, surf, frame, next_frame=None, alpha=0.0):
        pygame.draw.rect(surf, (22, 28, 42), self.rect, border_radius=10)
        for i in range(int(GRID_SIZE) + 1):
            c = (36, 44, 64)
            pygame.draw.line(surf, c, self.to_px(i, 0), self.to_px(i, GRID_SIZE))
            pygame.draw.line(surf, c, self.to_px(0, i), self.to_px(GRID_SIZE, i))

        # restaurants
        counts = {}
        for o in frame["orders"]:
            if o["status"] < int(OrderStatus.PICKED_UP):
                counts[o["rx"], o["ry"]] = counts.get((o["rx"], o["ry"]), 0) + 1
        for i, (x, y) in enumerate(self.restaurants):
            px = self.to_px(x, y)
            r = pygame.Rect(0, 0, 26, 26)
            r.center = px
            pygame.draw.rect(surf, RESTAURANT_COLOR, r, border_radius=5)
            text(surf, f"R{i}", px, 13, (20, 20, 20), True, "center")
            n = counts.get((x, y), 0)
            if n:
                text(surf, f"{n}", (px[0] + 16, px[1] - 24), 15, RESTAURANT_COLOR, True)

        drivers_now = {d["id"]: d for d in frame["drivers"]}
        drivers_next = {d["id"]: d for d in (next_frame["drivers"] if next_frame else frame["drivers"])}

        # routes (L-shaped: x first, then y)
        for d in frame["drivers"]:
            pts = [(d["x"], d["y"])] + [(w[0], w[1]) for w in d["route"]]
            col = DRIVER_COLOR[d["status"]]
            for a, b in zip(pts, pts[1:]):
                corner = (b[0], a[1])
                pygame.draw.lines(surf, col, False, [self.to_px(*a), self.to_px(*corner), self.to_px(*b)], 2)

        # customers / orders
        for o in frame["orders"]:
            px = self.to_px(o["cx"], o["cy"])
            if o["status"] == int(OrderStatus.DELIVERED):
                col = BAD if o["late"] else (90, 160, 120)
                pygame.draw.circle(surf, col, px, 5)
                text(surf, "v", px, 11, (15, 18, 26), True, "center")
            else:
                col = ORDER_COLOR[o["status"]]
                pygame.draw.polygon(surf, col, [(px[0], px[1] + 9), (px[0] - 6, px[1] - 3), (px[0] + 6, px[1] - 3)])
                pygame.draw.circle(surf, col, (px[0], px[1] - 5), 6)
                if o["late"]:
                    pygame.draw.circle(surf, BAD, (px[0], px[1] - 5), 9, 2)
                text(surf, o["id"], (px[0], px[1] - 5), 10, (15, 18, 26), True, "center")

        # drivers (interpolated)
        for did, d in drivers_now.items():
            n = drivers_next.get(did, d)
            x = d["x"] + (n["x"] - d["x"]) * alpha
            y = d["y"] + (n["y"] - d["y"]) * alpha
            px = self.to_px(x, y)
            pygame.draw.circle(surf, DRIVER_COLOR[d["status"]], px, 11)
            pygame.draw.circle(surf, (15, 18, 26), px, 11, 2)
            text(surf, did, px, 12, (15, 18, 26), True, "center")
            if d["load"] > 1:
                text(surf, f"x{d['load']}", (px[0] + 13, px[1] - 18), 12, WARN, True)

    def draw_legend(self, surf, pos):
        x, y = pos
        items = [("Driver: available", GOOD), ("to restaurant", ACCENT), ("waiting for food", WARN),
                 ("to customer", VIOLET)]
        for label, col in items:
            pygame.draw.circle(surf, col, (x + 6, y + 9), 6)
            text(surf, label, (x + 18, y), 14, MUTED)
            x += 28 + len(label) * 7
