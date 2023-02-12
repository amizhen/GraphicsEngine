import numpy as np
from PIL import Image


class Screen:
    def __init__(self, rows, cols, d_color=(0, 0, 0)):
        self.screen = np.array([[d_color for _ in range(cols)] for _ in range(rows)], dtype='B')
        self.img = Image.new('RGB', (self.screen.shape[0], self.screen.shape[1]))

    # Drawing

    def pixel(self, x, y, d_color=(0, 0, 0)):
        self.screen[y][x] = np.array(d_color)

    def color_screen(self, d_color=(0, 0, 0)):
        for row in range(np.shape(self.screen)[0]):
            for col in range(np.shape(self.screen)[1]):
                self.screen[row][col] = np.array(d_color)

    def line(self, x1, y1, x2, y2, color=(0, 0, 0)):

        # Casework for negative / negative slope
        if x1 > x2:
            if y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = x2, x1

        x, y = x1, y1
        while x < x2:
            self.pixel(x, y, d_color=color)
            if (y2 - y1) * (x - x1) > (x2 - x1) * (y - y1):  # If under the line
                y += 1
            x += 1
        self.pixel(x1, y1, d_color=(255, 0, 0))
        self.pixel(x2, y2, d_color=(255, 0, 0))

    # Utility Functions

    def write_ppm(self, file, string=False):
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
                file.write(self.screen)  # This works numpy bytes array can be written

    def save_extension(self, imgfile, extension='PNG'):
        pixels = [tuple(pixel) for row in self.screen for pixel in row]
        self.img.putdata(pixels)
        self.img.save(imgfile, extension)

    def display(self):
        pixels = [tuple(pixel) for row in self.screen for pixel in row]
        self.img.putdata(pixels)
        self.img.show()


s = Screen(500, 500)
# s.line(100, 10, 270, 100, color=(0, 0, 255))
s.line(270, 100, 100, 10, color=(0, 0, 255))
# s.color_screen((0,255,0))
s.display()
# s.write_ppm("test.ppm")
# s.save_extension("test.png")
