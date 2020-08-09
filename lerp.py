import numpy as np 
import cv2
from line_profiler import LineProfiler

profile = LineProfiler()


def lerp1d( a,  b,  w):
    if b>a:
        return a + w*(b-a)
    else:
        return b + w*(a-b)
    """
    a + w*(b-a)

    Returns the linear interpolation of a and b based on weight w.

    a and b are either both scalars or both vectors of the same length. 
    The weight w may be a scalar or a vector of the same length as a and b. 
    w can be any value (so is not restricted to be between zero and one); 
    if w has values outside the [0,1] range, it actually extrapolates.

    lerp returns a when w is zero and returns b when w is one.
    """


@profile
# def lerp2d(grid, centroid:np.ndarray):
#     """ Linear Interpolation
#     grid is a 2by2 matrix 
#     centroid is the centroid of the 2x2 matrix, (row-y,col-x), range:[0,1]
#      -----r0-- ---------
#     |0,0   |  |0,1      |
#     |      |  |         |
#     | -px- x -+ - qx - -|
#      ------+--+---------
#     |1,0   |  |1,1      |
#     |     qy  |         |
#     |      |  |         |
#      -----r1-- ---------
#     """

#     p = (1 - np.round(centroid)+centroid)/2

#     r0 = lerp1d(grid[0,0],grid[0,1],p[1])
#     r1 = lerp1d(grid[1,0],grid[1,1],p[1])
#     r = lerp1d(r0,r1,p[0]) +0.0001 # +0.0001 for np.round, sometimes 3.5 round down to 3. since computer science basis..
#     # if (grid<np.round(r)).all():
#     #     print(f'grid: {grid[0,0]},{grid[0,1]},{grid[1,0]},{grid[1,1]} | r: {r,np.round(r)}| p: {np.round(p,4)} | centroid: {centroid}')
#     print(f'grid: {grid[0,0]},{grid[0,1]},{grid[1,0]},{grid[1,1]} | r: {r,np.round(r)}| p: {np.round(p,4)} | centroid: {centroid}')
#     return np.round(r)

def lerp2d(f00,f01,f10,f11, centroid_h, centroid_w):
    """ Linear Interpolation
    grid is a 2by2 matrix 
    centroid is the centroid of the 2x2 matrix, (row-y,col-x), range:[0,1]
     -----r0-- ---------
    |0,0   |  |0,1      |
    |      |  |         |
    | -px- x -+ - qx - -|
     ------+--+---------
    |1,0   |  |1,1      |
    |     qy  |         |
    |      |  |         |
     -----r1-- ---------

    centroid to weight
    diff + 1block / 2blocks

    diff = round(x) - x [-0.4999, 0.4999]
    p = [1 block + (round(x)- x)] 
        -------------------------
                2 blocks

    """

    weight_h = (1 + np.round(centroid_h)-centroid_h)/2  
    weight_w = (1 + np.round(centroid_w)-centroid_w)/2

    r0 = lerp1d(f00,f01,weight_w)
    r1 = lerp1d(f10,f11,weight_w)
    r = lerp1d(r0,r1,weight_h) +0.0001 # +0.0001 for np.round, sometimes 3.5 round down to 3. since computer science basis..
    # if (grid<np.round(r)).all():
    #     print(f'grid: {grid[0,0]},{grid[0,1]},{grid[1,0]},{grid[1,1]} | r: {r,np.round(r)}| p: {np.round(p,4)} | centroid: {centroid}')
    # print(f'mid: {f11}, grid: {f00},{f01},{f10},{f11} | r0: {round(r0,2)}, r1: {round(r1,2)},r: {r, np.round(r)}| p h: {round(weight_h,4)}, w: {round(weight_h,4)} | centroid h: {round(centroid_h,4)}, w: {round(centroid_w,4)}')
    return np.round(r)


@profile
def downsample(inp, out):
    """"
    centroid is the centroid of the 2x2 matrix, (row-y,col-x), range:[0,1]

    ** only consider downsample resize, 

    When s < 0.5 grid only have 1 block, this would cause numpy error. (dimension)

    2 solutions can solve, 
    1) padding + conv mean and 
    2) condition catch if s < 0.5 , dst[i,j]= src[0,0]
    
    """
    src_h, src_w = inp.shape
    dst_h, dst_w = out.shape

    stride_h = src_h / dst_h
    stride_w = src_w / dst_w
    # centroid = np.zeros((2,), dtype=np.float32)
    for h in range(out.shape[0]): # i, h 
        for w in range(out.shape[1]): # j, w
            centroid_h = stride_h * (h + 0.5) # row / y 
            centroid_w = stride_w * (w + 0.5) # col / x 
            if centroid_h % 2 == 0.5: centroid_h+=0.00001 # python even rounding
            if centroid_w % 2 == 0.5: centroid_w+=0.00001 # python even rounding

            grid = inp[int(round(centroid_h - 1 )) : int(round(centroid_h + 1)),
                       int(round(centroid_w - 1 )) : int(round(centroid_w + 1))]

            f00 = grid[0,0]
            f01 = grid[0,1]
            f10 = grid[1,0]
            f11 = grid[1,1]

            # print(int(round(centroid[0] - 1 )) , int(round(centroid[0] + 1)), int(round(centroid[1] - 1 )), int(round(centroid[1] + 1)))
            # print(grid, np.round(centroid,2))
            assert grid.size == 4 
            out[h,w] = lerp2d(f00,f01,f10,f11, centroid_h, centroid_w)
def main():
    """
    # inp_image = cv2.resize(cv2.imread("rgba.png"),(1920,1080))
    inp_image = cv2.imread("trump.jpg")
    inp_image = cv2.resize(inp_image,(1080,1920))
    out_image = np.zeros((608,608,3),dtype = np.uint8)

    for i in range(3):
        downsample(inp=inp_image[:,:,i], out=out_image[:,:,i])
    print(out_image.shape)
    cv2.imwrite("output.jpg",out_image)


    cv2.imwrite("trump_nn.jpg",cv2.resize(inp_image,(608,608),interpolation = cv2.INTER_NEAREST))
    cv2.imwrite("trump_lerp.jpg",cv2.resize(inp_image,(608,608)))
    exit()
    """
    inp_image = (np.arange((1920*1080*3),dtype = np.uint8)+1).reshape(1080,1920,3)
    # inp_image = (np.arange((9*9*3),dtype = np.uint8)+1).reshape(9,9,3)
    # inp_image = (np.arange((10*18),dtype = np.uint8)+1).reshape(10,18)
    out_image = np.zeros((608,608,3),dtype = np.uint8)
    # out_image = np.zeros((3,3,3),dtype = np.uint8)

    # print(np.array(inp_image.shape) / np.array(out_image.shape))
    for i in range(3):
        downsample(inp=inp_image[:,:,i], out=out_image[:,:,i])
        print("-----------------Before-----------------")
        print(inp_image[0:10,0:10,i])
        print("-----------------After-----------------")
        print(out_image[0:10,0:10,i])
        print("=======================================")

    
    # out_image2 = cv2.resize(inp_image,(3,3))
    # print(out_image2)


if __name__ == "__main__":
    main()
    profile.print_stats()


# int x=i*m/a
# int x=(i+0.5)*m/a-0.5

# int y=j*n/b
# int y=(j+0.5)*n/b-0.5