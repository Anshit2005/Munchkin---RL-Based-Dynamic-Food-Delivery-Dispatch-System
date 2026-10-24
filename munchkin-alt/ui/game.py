"""Game-like interface: configuration -> animated simulation -> leaderboard.

    python -m ui.game

Episode duration is fixed at 3 simulated minutes and is not a configurable parameter.
"""
from __future__ import annotations

import sys

import pygame

from ui.screens.config_screen import ConfigScreen

SIZE = (1120, 730)


class Game:
    def __init__(self, size=SIZE):
        pygame.init()
        pygame.display.set_caption("RL Food Delivery Dispatch")
        self.screen = pygame.display.set_mode(size)
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = ConfigScreen(self)

    def goto(self, state):
        self.state = state

    def tick(self, events, dt):
        for e in events:
            if e.type == pygame.QUIT:
                self.running = False
            else:
                self.state.handle(e)
        self.state.update(dt)
        self.state.draw(self.screen)

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.tick(pygame.event.get(), dt)
            pygame.display.flip()
        pygame.quit()


def main():
    Game().run()
    sys.exit(0)


if __name__ == "__main__":
    main()
