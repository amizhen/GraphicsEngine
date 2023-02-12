import numpy as np
import PIL


class Screen:
    def __init__(self, rows, cols):
        d_color = (0, 255, 0)
        self.screen = np.array([[d_color for _ in range(cols)] for _ in range(rows)], dtype='B')

    def color_pixel(self, x, y, r, g, b):
        self.screen[y][x] = [r, g, b]

    def write_screen(self, file, string=False):
        rows, cols, _ = np.shape(self.screen)
        if string:
            with open(file, 'w') as file:
                file.write(f'P3\n{rows} {cols}\n255\n')
                for row in range(rows):
                    for col in range(cols):
                        file.write(f'{" ".join([str(i) for i in self.screen[row, col]])}\n')
        else:
            with open(file, 'wb') as file:
                file.write(f'P6\n {rows} {cols} {255} + \n'.encode())
                file.write(self.screen)  # This works numpy bytes array can be writte


s = Screen(50, 50)
s.write_screen("test.ppm")
