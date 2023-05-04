from utilities import *

width = 501
height = 601
s = Screen(height, width)

for x in range(0, width, 5):
    x0, y0, x1, y1 = 0, 0, 0, 0
    if x < 226:
        y1 = 600
        x0 = 0
        y0 = ((-2 * x + 500) * (x0-x) - x*x + 500*x)//125
        x1 = (y1*125-500*x+x*x)//(-2*x+500)+x
        # print(x, x0, y0, x1, y1)
    elif x < 274:
        x0 = 0
        x1 = 500
        y0 = ((-2 * x + 500) * (x0-x) - x*x + 500*x)//125
        y1 = ((-2 * x + 500) * (x1-x) - x*x + 500*x)//125
    else:
        y0 = 600
        x1 = 500
        y1 = ((-2 * x + 500) * (x1-x) - x*x + 500*x)//125
        x0 = (y0*125-500*x+x*x)//(-2*x+500)+x

    s.line(x0, y0, x1, y1, d_color=(0, 255, 0))
# s.line(0,0,50,100, color=(0, 255, 0))
s.display()
# s.write_ppm("test.ppm")
