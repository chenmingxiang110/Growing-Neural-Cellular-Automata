import pygame
import numpy as np

class displayer:

    def __init__(self, _map_shape, pix_size, has_gap=False):
        """
        _map_size: tuple
        color_map: a list indicates the color to each index.
                   0 : empty block, should always white
                   1+: varies building types
        """
        pygame.init()
        clock = pygame.time.Clock()
        clock.tick(60)

        self.has_gap = has_gap
        self.pix_size = pix_size
        self.screen = pygame.display.set_mode((_map_shape[1]*self.pix_size,
                                               _map_shape[0]*self.pix_size))

    def update(self, _map):
        self.screen.fill((255,255,255))

        for i in range(_map.shape[0]):
            for j in range(_map.shape[1]):
                x = j * self.pix_size + int(self.pix_size/2)
                y = i * self.pix_size + int(self.pix_size/2)
                if self.has_gap:
                    size = min(int(self.pix_size * 0.75), self.pix_size-2)
                else:
                    size = self.pix_size
                s = pygame.Surface((size,size))
                c = (_map[i,j]*256).astype(int)[:3]
                s.fill(c)
                self.screen.blit(s, (x-int(size/2), y-int(size/2)))
        pygame.display.update()
