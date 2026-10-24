"""Tiny pygame widgets: Button, Slider, text helper."""
from __future__ import annotations

import pygame

BG = (18, 22, 33)
PANEL = (28, 34, 50)
PANEL_LIGHT = (40, 48, 70)
TEXT = (232, 236, 245)
MUTED = (140, 150, 172)
ACCENT = (64, 156, 255)
GOOD = (84, 214, 140)
WARN = (255, 196, 61)
BAD = (255, 99, 99)
VIOLET = (170, 120, 255)

_fonts: dict = {}


def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _fonts:
        _fonts[key] = pygame.font.SysFont("segoeui,arial,helvetica", size, bold=bold)
    return _fonts[key]


def text(surf, s, pos, size=18, color=TEXT, bold=False, anchor="topleft"):
    img = font(size, bold).render(str(s), True, color)
    r = img.get_rect(**{anchor: pos})
    surf.blit(img, r)
    return r


class Button:
    def __init__(self, rect, label, on_click=None, enabled=True, toggle=False, active=False, size=17,
                 tooltip=""):
        self.rect = pygame.Rect(rect)
        self.label, self.on_click, self.enabled = label, on_click, enabled
        self.toggle, self.active, self.size, self.tooltip = toggle, active, size, tooltip

    def handle(self, event) -> bool:
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.enabled
                and self.rect.collidepoint(event.pos)):
            if self.on_click:
                self.on_click()
            return True
        return False

    def draw(self, surf):
        hover = self.rect.collidepoint(pygame.mouse.get_pos()) and self.enabled
        if not self.enabled:
            bg, fg = (34, 38, 52), (95, 102, 122)
        elif self.active:
            bg, fg = ACCENT, (10, 14, 24)
        else:
            bg, fg = (PANEL_LIGHT if not hover else (56, 66, 94)), TEXT
        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        text(surf, self.label, self.rect.center, self.size, fg, anchor="center")


class Slider:
    def __init__(self, rect, label, vmin, vmax, value, step=1, fmt="{:.0f}"):
        self.rect = pygame.Rect(rect)
        self.label, self.vmin, self.vmax, self.value, self.step, self.fmt = label, vmin, vmax, value, step, fmt
        self.drag = False

    def _set_from_x(self, x):
        f = min(1.0, max(0.0, (x - self.rect.x) / self.rect.w))
        v = self.vmin + f * (self.vmax - self.vmin)
        self.value = round(v / self.step) * self.step
        self.value = min(self.vmax, max(self.vmin, self.value))

    def handle(self, event) -> bool:
        grab = pygame.Rect(self.rect.x - 10, self.rect.y - 14, self.rect.w + 20, self.rect.h + 28)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and grab.collidepoint(event.pos):
            self.drag = True
            self._set_from_x(event.pos[0])
            return True
        if event.type == pygame.MOUSEMOTION and self.drag:
            self._set_from_x(event.pos[0])
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.drag:
            self.drag = False
            return True
        return False

    def draw(self, surf):
        text(surf, self.label, (self.rect.x, self.rect.y - 30), 17, MUTED)
        text(surf, self.fmt.format(self.value), (self.rect.right, self.rect.y - 30), 18, TEXT, True, "topright")
        pygame.draw.rect(surf, PANEL_LIGHT, self.rect, border_radius=4)
        f = (self.value - self.vmin) / (self.vmax - self.vmin)
        fill = pygame.Rect(self.rect.x, self.rect.y, int(self.rect.w * f), self.rect.h)
        pygame.draw.rect(surf, ACCENT, fill, border_radius=4)
        pygame.draw.circle(surf, TEXT, (self.rect.x + int(self.rect.w * f), self.rect.centery), 10)
