

def rectangle(bl,width,height,theta):

    """
    return ---> Rx,Ry, which are corner points to be plotted
    """
    import numpy as np

    br = [bl[0]+width-1,bl[1]]
    tr = [br[0],br[1]+height-1]
    tl = [tr[0]-width+1,tr[1]]

    rect = zip(bl,br,tr,tl)
    L = list(rect)

    Rx = list(L[0])
    Ry = list(L[1])

    # Rx1 , Rx2 are x and y coordinates of corners.
    Rx1 = Rx.copy()
    Ry1 = Ry.copy()

    # To complete rectangle first corner(bl) included.
    Rx.append(Rx[0])
    Ry.append(Ry[0])

    # generating rotation matrix.
    theta = np.deg2rad(theta)
    R = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])

    # XY are array of corners to be rotated.
    XY = list(zip(Rx1,Ry1))
    XY = [np.array(l) for l in XY]

    # centre of rotation is translated to bottom left.
    translated_rect = [m-XY[0] for m in XY]

    # Rotation performed.
    translated_rect_p = [np.matmul(R,m) for m in translated_rect]

    # origin translated back to actual origin and XpYp are transformed corners of the rectangles.
    XpYp = [m+XY[0] for m in translated_rect_p]

    # restructuring to give primed version of the corners
    Rxp = [round(l[0]) for l in XpYp]
    Ryp = [round(l[1]) for l in XpYp]
    Rxp.append(Rxp[0])
    Ryp.append(Ryp[0])


    # print(f"XpYp = {XpYp}")

    return Rx,Ry,Rxp,Ryp


def rotation_of_coordinate(coordinate,theta):
    """
    coordinate is either tuple or list of two numbers.
    theta in degrees.
    return ---> rotated coordinate.

    e.g.

    m = rotation_of_coordinate((8,0),45)
    print(m)
    return [6, -6]
    """
    import numpy as np
    a = np.array(coordinate)

    theta = np.deg2rad(theta)
    R = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    ap = np.matmul(R,a)
    ap = [round(l) for l in ap]
    return ap



def create_coordinates(width,height):
    """
    return list of coordinates starting from [0,0] of width and height.

    e.g.
    l = create_coordinates(2,3)
    print(l)
    return [[0 0]
            [1 0]
            [0 1]
            [1 1]
            [0 2]
            [1 2]]
    """
    import numpy as np
    m = int(width)
    n = int(height)

    x = np.arange(m)
    y = np.arange(n)

    xx,yy = np.meshgrid(x,y)

    l = list(zip(xx,yy))
    M = [list(zip(k[0],k[1])) for k in l]
    M = np.array(M)
    coordinates = M.reshape((m*n,2))
    return coordinates


def ROI(image,width,height,bottom_left,theta,show_plot=False):
    import numpy as np
    import matplotlib.pyplot as plt
    bl = bottom_left
    l = create_coordinates(width,height)
    roi = np.zeros((height,width))
    for j in l:
        k = rotation_of_coordinate(j,theta=theta)
        roi[j[1]][j[0]] = image[k[1]+bl[1]][k[0]+bl[0]]

    if show_plot:
        Rx,Ry,Rxp,Ryp = rectangle(bl=bl,width=width,height=height,theta=theta)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1,)
        ax1.imshow(img,origin = "lower")
        # ax1.plot(Rx,Ry,color='red')
        ax1.plot(Rxp,Ryp,color='red')

        ax2 = fig.add_subplot(1,2,2,)
        ax2.imshow(roi,origin = "lower")

    return roi

if __name__=="__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    bl = [10,25]
    width = 5
    height = 7
    theta = 45      # in degrees
    # img = np.linspace(1,7000,7000).reshape((100,70))
    img = np.random.randint(0,50,(100,70))

    x = ROI(image=img,width=width,height=height,bottom_left=bl,theta=12,show_plot=True)
    print(x)
    plt.show()


