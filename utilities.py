import math
import numpy as np
from PIL import Image


class Screen:
    """
    A class used to represent a computer image

    ...

    Attributes
    ----------
    screen : numpy.ndarray
        3 Dimensional Array made of arrays of 3 ints representing a pixel's color, within columns within rows
    img : PIL.Image.Image
        PIL Image object used for exporting

    Methods
    -------
    pixel(x, y, d_color=(0,0,0))
        Colors a single pixel
    color_screen(d_color=(0, 0, 0))
        Colors the entire screen
    line(self, x1, y1, x2, y2, color=(0, 0, 0))
        Draws a line
    write_ppm(self, file, string=False)
        Saves image as a ppm file
    save_extension(self, imgfile, extension='PNG')
        Saves image with any extension using PIL library
    display()
        Displays image
    """

    def __init__(self, rows, cols, d_color=(0, 0, 0)):
        """Creates a Screen object

        :param rows: Height of the screen
        :type rows: int
        :param cols: Width of the screen
        :type cols: int
        :param d_color: Base screen color
                (default is (0,0,0)
        :type d_color: tuple
        :return: None
        """
        self.screen = np.array([[d_color for _ in range(cols)] for _ in range(rows)], dtype='B')
        self.img = Image.new('RGB', (self.screen.shape[1], self.screen.shape[0]))

    # Drawing

    def pixel(self, x, y, d_color=(0, 0, 0)):
        """Colors one pixel, with origin at the bottom left

        :param x: X Cord
        :type x: int
        :param y: Y Cord
        :type y: int
        :param d_color: Pixel color
            (default is (0,0,0))
        :type d_color: tuple
        :return None
        """
        self.screen[len(self.screen) - y - 1][x] = np.array(d_color)

    def color_screen(self, d_color=(0, 0, 0)):
        """Colors the entire screen

        :param d_color: Screen color
            (default is (0,0,0)
        :type d_color: tuple
        :return: None
        """
        for row in range(np.shape(self.screen)[0]):
            for col in range(np.shape(self.screen)[1]):
                self.screen[row][col] = np.array(d_color)

    def line(self, x1, y1, x2, y2, d_color=(0, 0, 0)):
        """Draws a line

        :param x1: X cord 1st endpoint
        :type x1: int
        :param y1: Y cord 1st endpoint
        :type y1: int
        :param x2: X cord 2nd endpoint
        :type x2: int
        :param y2: Y cord 2nd endpoint
        :type y2: int
        :param d_color: Line color
            (default is (0,0,0))
        :type d_color: tuple
        :return: None
        """
        if x1 > x2:  # Make x2 > x1
            y1, y2 = y2, y1
            x1, x2 = x2, x1

        delta = 1
        if y2 < y1:  # If Y is decreasing
            delta = -1
        cost = 0
        if x2 - x1 >= abs(y2 - y1):  # |m| <= 1
            x, y = x1, y1
            while x <= x2:  # Inclusive endpoints
                self.pixel(x, y, d_color=d_color)
                x += 1
                cost += 2 * y2 - 2 * y1  # Adjust for x
                if (cost + (- x2 + x1) * delta) * delta > 0:  # If under the line
                    y += delta
                    cost += (-2 * x2 + 2 * x1) * delta
        else:
            if delta == -1:
                y1, y2 = y2, y1
                x1, x2 = x2, x1
            x, y = x1, y1
            while y <= y2:  # Inclusive endpoints
                self.pixel(x, y, d_color=d_color)
                y += 1
                cost += -2 * x2 + 2 * x1  # Adjust for y
                if (cost + (y2 - y1) * delta) * delta < 0:  # If over the line
                    x += delta
                    cost += (2 * y2 - 2 * y1) * delta

    # Utility Functions

    def write_ppm(self, file, string=False):
        """Saves image as a p6 (or p3) ppm file

        :param file: Name of the file
        :type file: str
        :param string: If True saves as in ASCII with p3 format
            (default = False)
        :type string: bool
        :return: None
        """
        rows, cols, _ = np.shape(self.screen)
        if string:
            with open(file, 'w') as file:
                file.write(f'P3\n{cols} {rows}\n255\n')
                for row in range(rows):
                    for col in range(cols):
                        file.write(f'{" ".join([str(i) for i in self.screen[row, col]])}\n')
        else:
            with open(file, 'wb') as file:
                # print(cols, rows)
                file.write(f'P6\n {cols} {rows} {255} + \n'.encode())
                file.write(self.screen)  # This works numpy bytes array can be written

    def save_extension(self, imgfile, extension='PNG'):
        """Saves image as any file type

        :param imgfile: Name of the file
        :type imgfile: str
        :param extension: File type
            (default 'PNG')
        :type extension: str
        :return: None
        """
        pixels = [tuple(pixel) for row in self.screen for pixel in row]
        self.img.putdata(pixels)
        self.img.save(imgfile, extension)

    def display(self):
        """ Displays the image

        :return: None
        """
        pixels = [tuple(pixel) for row in self.screen for pixel in row]
        self.img.putdata(pixels)
        # pixels = [tuple(self.screen[row, col]) for col in range(np.shape(self.screen)[1]) for row in range(np.shape(self.screen)[0])]
        # self.img.putdata(pixels)
        # self.img = Image.fromarray(self.screen)
        self.img.show()


class Matrix:
    def __init__(self):
        self.matrix = []
        self.colors = []

    def __str__(self):
        out = ''
        for col in range(len(self.matrix[0])):
            for row in range(len(self.matrix)):
                out += str(self.matrix[row][col])
                out += ' '
            out += '\n'
        return out

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, item):
        return self.matrix[item]

    def add_point(self, x, y, z):
        self.matrix.append([x, y, z, 1])

    def add_edge(self, x1, y1, z1, x2, y2, z2, color=(0, 0, 0)):
        self.add_point(x1, y1, z1)
        self.add_point(x2, y2, z2)
        self.colors.append(color)

    def eye(self, n):
        self.matrix = [[0 if k != i else 1 for k in range(n)] for i in range(n)]

    def mult(self, m):
        self.matrix = [[sum(self.matrix[r][i] * m[i][c] for i in range(len(m))) for c in range(len(m[0]))] for r in
                       range(len(self.matrix))]

    # Transformations
    # def rot(self, theta, axis='z'):
    #     axi = {'z':2, 'y':1, 'x':0}
    #     axis = axi[axis]
    #     core = [[math.cos(theta), math.sin(theta), 0], [-1*math.sin(theta), math.cos(theta), 0]]
    #     core[0].insert(axis, 0)
    #     core[1].insert(axis, 0)
    #     core.insert(axis, [0 if i != axis else 1 for i in range(4)])
    #     core.insert(3, [0,0,0,0])
    #     self.mult(core)

    def draw(self, screen):
        for point in range(0, len(self.matrix), 2):
            screen.line(self.matrix[point][0], self.matrix[point][1], self.matrix[point + 1][0],
                        self.matrix[point + 1][1], d_color=self.colors[point // 2])


if __name__ == '__main__':
    m1 = Matrix()
    m2 = Matrix()

    print('\nTesting add_edge. Adding (1, 2, 3), (4, 5, 6) m2 =')
    m2.add_edge(1, 2, 3, 4, 5, 6)
    print(m2)

    print('Testing ident. m1 = ')
    m1.eye(4)
    print(m1)

    print('\nTesting matrix_mult. m1 * m2 =')
    m2.mult(m1)
    print(m2)

    m1 = Matrix()
    m1.add_edge(1, 2, 3, 4, 5, 6)
    m1.add_edge(7, 8, 9, 10, 11, 12)
    print("\nTesting Matrix mult. m1 =")
    print(m1)
    print("\nTesting Matrix mult. m1 * m2 =")
    m1.mult(m2)
    print(m1)

    edges = Matrix()
    edges.add_edge(50, 450, 0, 100, 450, 0)
    edges.add_edge(50, 450, 0, 50, 400, 0)
    edges.add_edge(100, 450, 0, 100, 400, 0)
    edges.add_edge(100, 400, 0, 50, 400, 0)

    edges.add_edge(200, 450, 0, 250, 450, 0)
    edges.add_edge(200, 450, 0, 200, 400, 0)
    edges.add_edge(250, 450, 0, 250, 400, 0)
    edges.add_edge(250, 400, 0, 200, 400, 0)

    edges.add_edge(150, 400, 0, 130, 360, 0)
    edges.add_edge(150, 400, 0, 170, 360, 0)
    edges.add_edge(130, 360, 0, 170, 360, 0)

    edges.add_edge(100, 340, 0, 200, 340, 0)
    edges.add_edge(100, 320, 0, 200, 320, 0)
    edges.add_edge(100, 340, 0, 100, 320, 0)
    edges.add_edge(200, 340, 0, 200, 320, 0)

    screen = Screen(500, 500, d_color=(255, 255, 255))
    color = [0, 0, 0]
    edges.draw(screen)
    screen.save_extension("bob.png")
    screen.display()

