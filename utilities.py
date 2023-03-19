import numpy as np
from PIL import Image
import math


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
        self.screen[self.screen.shape[0] - y - 1, x] = np.array(d_color)

    def color_screen(self, d_color=(0, 0, 0)):
        """Colors the entire screen

        :param d_color: Screen color
            (default is (0,0,0)
        :type d_color: tuple
        :return: None
        """
        for row in range(np.shape(self.screen)[0]):
            for col in range(np.shape(self.screen)[1]):
                self.screen[row, col] = np.array(d_color)

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
        self.img.show()


class EdgeList:
    def __init__(self):
        self.matrix = []
        self.colors = []

    def __str__(self):
        out = ''
        for col in range(len(self.matrix[0])-1):
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
        """Adds one endpoint to the EdgeList

        :param x: X cord
        :type x: int
        :param y: Y cord
        :type: y: int
        :param z: Z cord
        :type z: int
        :return: None
        """
        self.matrix.append(np.array([x, y, z, 1]))

    def add_edge(self, x1, y1, z1, x2, y2, z2, d_color=(0, 0, 0)):
        """ Adds the two endpoints of an edge to the EdgeList

        :param x1: X cord 1st endpoint
        :type x1: int
        :param y1: Y cord 1st endpoint
        :type y1: int
        :param z1: Z cord 1st endpoint
        :type z1: int
        :param x2: X cord 2nd endpoint
        :type x2: int
        :param y2: Y cord 2nd endpoint
        :type y2: int
        :param z2: Z cord 2nd endpoint
        :type z2: int
        :param d_color: Line color
            (default is (0,0,0))
        :return: None
        """
        self.add_point(x1, y1, z1)
        self.add_point(x2, y2, z2)
        self.colors.append(d_color)

    # Shapes

    def circle(self, x, y, z, r, steps=100, d_color=(0, 0, 0)):
        x1 = x + r
        y1 = y
        for i in range(steps + 1):
            x2 = math.cos(2 * math.pi * i / steps) * r + x
            y2 = math.sin(2 * math.pi * i / steps) * r + y
            self.add_edge(x1, y1, z, x2, y2, z, d_color=d_color)
            x1, y1 = x2, y2

    def hermite(self, x1, y1, x2, y2, rx1, ry1, rx2, ry2, z, steps=100, d_color=(0, 0, 0)):
        g = np.array([[x1, y1], [x2, y2, ], [rx1, ry1], [rx2, ry2]])
        h_inv = np.array([[2, -2, 1, 1], [-3, 3, -2, -1], [0, 0, 1, 0], [1, 0, 0, 0]])
        coef = h_inv @ g
        x, y = x1, y1
        for i in range(1, steps):
            xt, yt = x, y
            x, y = np.array([(i / steps) ** 3, (i / steps) ** 2, (i / steps), 1]) @ coef
            self.add_edge(xt, yt, z, x, y, z, d_color=d_color)

    def bezier(self, x1, y1, cx1, cy1, cx2, cy2, x2, y2, z, steps=100, d_color=(0, 0, 0)):
        b = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
        g = np.array([[x1, y1], [cx1, cy1], [cx2, cy2], [x2, y2]])
        coef = b @ g
        x, y = x1, y1
        for i in range(0, steps):
            xt, yt = x, y
            x, y = np.array([(i / steps) ** 3, (i / steps) ** 2, (i / steps), 1]) @ coef
            self.add_edge(xt, yt, z, x, y, z, d_color=d_color)


    def draw(self, scrn):
        """Draw edges to the given Screen

        :param scrn: Screen to be drawn to
        :return: None
        """
        for point in range(0, len(self.matrix), 2):
            scrn.line(round(self.matrix[point][0]), round(self.matrix[point][1]), round(self.matrix[point + 1][0]),
                      round(self.matrix[point + 1][1]), d_color=self.colors[point // 2])

    def transform(self, m):
        """Multiplies EdgeList by given transformation matrix

         :param m: Matrix to be multiplied by
         :return: None
         """
        # This is left multiplication (I think)
        # self.matrix = [[sum(self.matrix[r][i] * m.matrix[i, c] for i in range(np.shape(m.matrix)[0])) for c in range(np.shape(m.matrix)[1])] for r in
        #                range(len(self.matrix))]
        self.matrix = [m.matrix @ v for v in self.matrix]



class Transformation:
    def __init__(self):
        self.matrix = np.eye(4)

    def __str__(self):
        return str(self.matrix)
    # Transformations

    def clear(self):
        self.matrix = np.eye(4)

    def rot(self, theta, axis='z'):
        theta = theta / 180.0 * math.pi # If theta in degrees
        axi = {'z': 2, 'y': 1, 'x': 0}
        axis = axi[axis]
        core = [[math.cos(theta), (-1)**(axis+1) * math.sin(theta), 0], [(-1)**axis*math.sin(theta), math.cos(theta), 0]]
        core[0].insert(axis, 0)
        core[1].insert(axis, 0)
        core.insert(axis, [0 if i != axis else 1 for i in range(4)])
        core.insert(3, [0, 0, 0, 1])

        self.matrix = np.array(core) @ self.matrix

    def translate(self, x, y, z):
        m = np.eye(4)
        m[0:3, 3] = np.array([x, y, z])
        self.matrix = m @ self.matrix

    def scale(self, x, y, z):
        m = np.eye(4)
        m[0, 0] = x
        m[1, 1] = y
        m[2, 2] = z
        self.matrix = m @ self.matrix


def parse(filename, screen):
    with open(filename, 'r') as file:
        t = Transformation()
        edges = EdgeList()
        current_cmd = None
        for l in file:
            l = l.strip()
            if not current_cmd:
                current_cmd = l
                # 0 arg commands
                if current_cmd[0] == '#':
                    current_cmd = None
                else:
                    match current_cmd:
                        case 'apply':
                            edges.transform(t)
                            # t = Transformation()
                            current_cmd = None
                        case 'ident':
                            t = Transformation()
                            current_cmd = None
                        case 'draw':
                            edges.draw(screen)
                            current_cmd = None
                        case 'display':
                            screen.color_screen()
                            edges.draw(screen)
                            screen.display()
                            current_cmd = None
                        case 'quit':
                            return None  # Or break
            else:
                # Commands with args
                match current_cmd:
                    case 'line':
                        edges.add_edge(*[int(i) for i in l.split(' ')], d_color=(0, 255, 0))
                    case 'save':
                        screen.color_screen()
                        edges.draw(screen)
                        screen.save_extension(l)
                    case 'scale':
                        t.scale(*[float(i) for i in l.split(' ')])
                    case 'translate':
                        t.translate(*[float(i) for i in l.split(' ')])
                    case 'rotate':
                        t.rot(float(l.split(' ')[1]), axis=l.split(' ')[0])
                    case 'hermite':
                        edges.hermite(*[float(i) for i in l.split(' ')], 0, d_color=(0, 255,0 ))
                    case 'bezier':
                        edges.bezier(*[float(i) for i in l.split(' ')], 0, d_color=(0,255,0))
                    case 'circle':
                        edges.circle(*[float(i) for i in l.split(' ')], d_color=(0,255,0))

                current_cmd = None



if __name__ == '__main__':
    s = Screen(500, 500)
    parse('script', s)


