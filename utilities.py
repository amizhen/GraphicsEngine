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
        if x1 > x2:  # Make x2 > x1
            y1, y2 = y2, y1
            x1, x2 = x2, x1

        delta = 1
        if y2 < y1:  # If Y is decreasing
            delta = -1
        cost = 0
        if delta == 1:
            print(1)
            if x2 - x1 >= abs(y2 - y1):  # |m| <= 1
                x, y = x1, y1
                while x <= x2:  # Inclusive endpoints
                    self.pixel(x, y, d_color=color)
                    x += 1
                    cost += (2 * y2 - 2 * y1)*delta  # Adjust for x
                    if cost + (- x2 + x1) * delta > 0:  # If under the line
                        y += delta
                        cost += (-2 * x2 + 2 * x1)*delta

            else:
                # if delta == -1:
                #     y1, y2 = y2, y1
                #     x1, x2 = x2, x1
                x, y = x1, y1
                # print(2, x1, y1, x2, y2)
                while y <= y2:  # Inclusive endpoints
                    self.pixel(x, y, d_color=color)
                    y += 1
                    cost += (-2 * x2 + 2 * x1)*delta  # Adjust for y
                    if cost + (y2 - y1)*delta < 0:  # If over the line
                        x += delta
                        cost += (2 * y2 - 2 * y1)*delta
        else:
            print(2)
            if x2 - x1 >= abs(y2 - y1):  # |m| <= 1
                x, y = x1, y1
                while x <= x2:  # Inclusive endpoints
                    self.pixel(x, y, d_color=color)
                    x += 1
                    cost += 2 * y2 - 2 * y1  # Adjust for x
                    if cost + (- x2 + x1) * delta < 0:  # If under the line
                        y += delta
                        cost += (-2 * x2 + 2 * x1) * delta

            else:
                if delta == -1:
                    y1, y2 = y2, y1
                    x1, x2 = x2, x1
                x, y = x1, y1
                # print(2, x1, y1, x2, y2)
                while y <= y2:  # Inclusive endpoints
                    self.pixel(x, y, d_color=color)
                    y += 1
                    cost += -2 * x2 + 2 * x1  # Adjust for y
                    if cost + (y2 - y1) * delta > 0:  # If over the line
                        x += delta
                        cost += (2 * y2 - 2 * y1) * delta
        # 249, 249, 0, 499
        # 0 499 249 249
        # self.pixel(x1, y1, d_color=(255,0,00))
        # self.pixel(x2, y2, d_color=(255,0,00))

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

if __name__ =='main':
    s = Screen(500, 500)
    s.line(249, 249, 499, 249, color=(255, 0, 0))  # m = 0
    s.line(249, 249, 499, 499, color=(0, 255, 0))  # m = 1
    s.line(249, 249, 249, 499, color=(0, 0, 255))  # m = infinity
    s.line(249, 249, 0, 499, color=(255, 255, 0))  # m = -1
    # s.line(249, 249, 0, 249, color=(255, 0, 255))  # m = -0
    # s.line(249, 249, 0, 0, color=(0, 255, 255))  # m = 1
    # # s.line(249, 249, 349, 0, color=(0, 128, 128))  # m = -infinity
    # # s.pixel(349,0,(255,255,255))
    # s.line(249, 249, 499, 0, color=(128, 128, 0))  # m = -1


    s.pixel(249, 249, d_color=(255, 255, 255))

    s.display()
    # s.write_ppm("test.ppm")
    # s.save_extension("test.png")
