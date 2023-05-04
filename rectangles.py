from utilities import *


screen = Screen(500,500, d_color=(0,0,0))

m = EdgeList()


for i in range(0, 250, 5):
    screen.line(i, i, 499 - i, i, d_color=(int(255 * (250 - i) / 250), 0, int(i / 250 * 255)))
    screen.line(i, i, i, 499 - i, d_color=(int(255 * (250 - i) / 250), 0, int(i / 250 * 255)))
    screen.line(i, 499 - i, 499 - i, 499 - i, d_color=(int(255 * (250 - i) / 250), 0, int(i / 250 * 255)))
    screen.line(499 - i, i, 499 - i, 499 - i, d_color=(int(255 * (250 - i) / 250), 0, int(i / 250 * 255)))

screen.display()