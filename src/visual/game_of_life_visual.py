"""
Conway's Game of Life - Visual Implementation with Pygame

Controls:
    SPACE       - Pause/Resume simulation
    R           - Reset with random grid
    C           - Clear grid
    G           - Add glider at mouse position
    U           - Add glider gun at mouse position
    LEFT CLICK  - Draw cells
    RIGHT CLICK - Erase cells
    UP/DOWN     - Increase/Decrease speed
    +/-         - Zoom in/out
    ESC         - Quit

Usage: python game_of_life_visual.py [width] [height] [cell_size]
"""
import pygame
import numpy as np
import sys

BLACK = (0, 0, 0)          # background
GRAY = (40, 40, 40)        # grid lines
GREEN = (0, 255, 100)      # alive cells
YELLOW = (255, 255, 0)     # paused UI
WHITE = (255, 255, 255)    # text


class GameOfLife:
    def __init__(self, width: int = 120, height: int = 80, cell_size: int = 10):
        pygame.init()  # init pygame modules

        self.grid_width = width          # grid width in cells
        self.grid_height = height        # grid height in cells
        self.cell_size = cell_size       # cell size in pixels

        self.window_width = 1200         # fixed window width
        self.window_height = 800         # fixed window height

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))  # main window
        pygame.display.set_caption("Conway's Game of Life")  # window title

        self.grid_surface = pygame.Surface((width * cell_size, height * cell_size))  # grid canvas

        self.offset_x = 0                # pan offset x (unused for now)
        self.offset_y = 0                # pan offset y (unused for now)

        self.grid = np.zeros((height, width), dtype=np.uint8)  # 0=dead, 1=alive
        self.init_random()               # random start state

        self.running = False             # paused by default
        self.generation = 0              # generation counter
        self.speed = 10                  # sim steps per second

        self.clock = pygame.time.Clock()                 # frame timing
        self.font = pygame.font.Font(None, 28)           # main UI font
        self.small_font = pygame.font.Font(None, 22)     # help UI font

        self.mouse_down = False          # dragging state
        self.mouse_button = None         # 1=left, 3=right

    def init_random(self, density: float = 0.3):
        self.grid = (np.random.random((self.grid_height, self.grid_width)) < density).astype(np.uint8)  # Bernoulli field
        self.generation = 0  # reset counter

    def clear_grid(self):
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)  # all dead
        self.generation = 0  # reset counter

    def add_glider(self, x: int, y: int):
        if x + 2 < self.grid_width and y + 2 < self.grid_height and x >= 0 and y >= 0:  # bounds check
            self.grid[y, x + 1] = 1              # row 0
            self.grid[y + 1, x + 2] = 1          # row 1
            self.grid[y + 2, x:x + 3] = 1        # row 2

    def add_glider_gun(self, x: int, y: int):
        if x + 36 >= self.grid_width or y + 9 >= self.grid_height or x < 0 or y < 4:  # needs space
            return

        self.grid[y:y + 2, x:x + 2] = 1                                  # left block
        self.grid[y:y + 3, x + 10] = 1                                   # spine
        self.grid[y - 1, x + 11] = self.grid[y + 3, x + 11] = 1          # tips
        self.grid[y - 2, x + 12:x + 14] = self.grid[y + 4, x + 12:x + 14] = 1
        self.grid[y + 1, x + 14] = 1
        self.grid[y - 1, x + 15] = self.grid[y + 3, x + 15] = 1
        self.grid[y:y + 3, x + 16] = 1
        self.grid[y + 1, x + 17] = 1
        self.grid[y - 2:y + 1, x + 20:x + 22] = 1                        # center block
        self.grid[y - 3, x + 22] = self.grid[y + 1, x + 22] = 1
        self.grid[y - 4, x + 24] = self.grid[y - 3, x + 24] = 1
        self.grid[y + 1, x + 24] = self.grid[y + 2, x + 24] = 1
        self.grid[y - 2:y, x + 34:x + 36] = 1                            # right block

    def step(self):
        neighbors = np.zeros_like(self.grid)  # neighbor accumulator
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue  # skip self
                neighbors += np.roll(np.roll(self.grid, dy, axis=0), dx, axis=1)  # wrap-around shifts

        birth = (self.grid == 0) & (neighbors == 3)  # B3
        survive = (self.grid == 1) & ((neighbors == 2) | (neighbors == 3))  # S23
        self.grid = (birth | survive).astype(np.uint8)  # next state

        self.generation += 1  # increment gen

    def screen_to_grid(self, sx: int, sy: int) -> tuple[int, int]:
        return (sx + self.offset_x) // self.cell_size, (sy + self.offset_y) // self.cell_size  # pixel -> cell

    def set_cell(self, sx: int, sy: int, value: int):
        gx, gy = self.screen_to_grid(sx, sy)  # convert coords
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:  # clip to grid
            self.grid[gy, gx] = value  # write cell

    def draw(self):
        self.grid_surface.fill(BLACK)  # clear canvas

        if self.cell_size >= 4:  # avoid clutter when zoomed out
            for x in range(0, self.grid_width * self.cell_size, self.cell_size):
                pygame.draw.line(self.grid_surface, GRAY, (x, 0), (x, self.grid_height * self.cell_size))  # vertical
            for y in range(0, self.grid_height * self.cell_size, self.cell_size):
                pygame.draw.line(self.grid_surface, GRAY, (0, y), (self.grid_width * self.cell_size, y))  # horizontal

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y, x]:  # draw only alive cells
                    rect = pygame.Rect(
                        x * self.cell_size + 1,
                        y * self.cell_size + 1,
                        self.cell_size - 1,
                        self.cell_size - 1
                    )
                    pygame.draw.rect(self.grid_surface, GREEN, rect)  # filled cell

    def draw_ui(self):
        pygame.draw.rect(self.screen, (30, 30, 30), (0, self.window_height - 35, self.window_width, 35))  # status bar

        status = "RUNNING" if self.running else "PAUSED"  # state label
        color = GREEN if self.running else YELLOW         # state color
        text = self.font.render(
            f"[{status}]  Gen: {self.generation}  Cells: {np.sum(self.grid)}  Speed: {self.speed} fps",
            True,
            color
        )
        self.screen.blit(text, (10, self.window_height - 28))  # status text

        if not self.running:  # show help only when paused
            help_bg = pygame.Surface((self.window_width, 25))  # top strip
            help_bg.set_alpha(200)                             # translucent
            help_bg.fill((30, 30, 30))                         # dark bg
            self.screen.blit(help_bg, (0, 0))                  # draw bg

            text = self.small_font.render(
                "SPACE: Play/Pause | R: Random | C: Clear | G: Glider | U: Gun | Click: Draw | UP/DOWN: Speed",
                True,
                WHITE
            )
            self.screen.blit(text, (10, 5))  # help text

    def handle_events(self) -> bool:
        for event in pygame.event.get():  # poll events
            if event.type == pygame.QUIT:
                return False  # close window

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False  # quit

                elif event.key == pygame.K_SPACE:
                    self.running = not self.running  # toggle run/pause

                elif event.key == pygame.K_r:
                    self.init_random()  # random reset

                elif event.key == pygame.K_c:
                    self.clear_grid()  # clear all

                elif event.key == pygame.K_g:
                    gx, gy = self.screen_to_grid(*pygame.mouse.get_pos())  # mouse cell
                    self.add_glider(gx, gy)  # stamp glider

                elif event.key == pygame.K_u:
                    gx, gy = self.screen_to_grid(*pygame.mouse.get_pos())  # mouse cell
                    self.add_glider_gun(gx, gy + 4)  # gun needs headroom

                elif event.key == pygame.K_UP:
                    self.speed = min(self.speed + 5, 60)  # speed up

                elif event.key == pygame.K_DOWN:
                    self.speed = max(self.speed - 5, 1)  # slow down

                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.cell_size = min(self.cell_size + 2, 50)  # zoom in
                    self.grid_surface = pygame.Surface((self.grid_width * self.cell_size, self.grid_height * self.cell_size))  # resize

                elif event.key == pygame.K_MINUS:
                    self.cell_size = max(self.cell_size - 2, 2)  # zoom out
                    self.grid_surface = pygame.Surface((self.grid_width * self.cell_size, self.grid_height * self.cell_size))  # resize

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_down = True  # start drag
                self.mouse_button = event.button  # store button
                self.set_cell(event.pos[0], event.pos[1], 1 if event.button == 1 else 0)  # paint once

            elif event.type == pygame.MOUSEBUTTONUP:
                self.mouse_down = False  # stop drag
                self.mouse_button = None  # clear state

            elif event.type == pygame.MOUSEMOTION and self.mouse_down:
                self.set_cell(event.pos[0], event.pos[1], 1 if self.mouse_button == 1 else 0)  # paint while dragging

        return True  # keep running

    def run(self):
        while self.handle_events():  # main loop
            if self.running:
                self.step()  # advance simulation

            self.screen.fill(BLACK)  # clear window
            self.draw()  # draw grid surface
            self.screen.blit(self.grid_surface, (-self.offset_x, -self.offset_y))  # blit grid
            self.draw_ui()  # draw overlays

            pygame.display.flip()  # swap buffers

            self.clock.tick(self.speed if self.running else 60)  # sim rate vs UI rate

        pygame.quit()  # clean exit


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 120      # CLI width
    height = int(sys.argv[2]) if len(sys.argv) > 2 else 80      # CLI height
    cell_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10   # CLI zoom

    GameOfLife(width, height, cell_size).run()  # start app


if __name__ == "__main__":
    main()  # entry point