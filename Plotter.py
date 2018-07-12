import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

class PlotGraphs:
    
    def plot_graph(self, values): 

        one_5d = values[0]
        one_4d = one_5d[int(len(one_5d)/2)]
        one_3d = one_4d[int(len(one_4d)/2)]
        one_2d = one_3d[int(len(one_3d)/2)]
        one_1d = one_2d[int(len(one_2d)/2)]

        two_5d = values[1]
        two_4d = two_5d[int(len(two_5d)/2)]
        two_3d = two_4d[int(len(two_4d)/2)]
        two_2d = two_3d[int(len(two_3d)/2)]
        two_1d = two_2d[int(len(two_2d)/2)]

        fig, ax = plt.subplots()
        ax.plot(one_1d, label="initial")
        ax.plot(two_1d, label="final")
        ax.legend()

        ax.grid(True, linestyle='-.')
        ax.tick_params(labelcolor='r', labelsize='medium', width=3)

        plt.title("Z vs Y")
        plt.show()

        # XY Plane
        fig,axis = plt.subplots(ncols=2)
        plt.suptitle("XY Planes")
        axis[0].imshow(values[0][0,0,:,:,0])
        axis[1].imshow(values[1][0,0,:,:,0])
        plt.show()

        # YZ Plane
        fig,axis = plt.subplots(ncols=2)
        plt.suptitle("YZ Planes")
        axis[0].imshow(values[0][0,0,0,:,:])
        axis[1].imshow(values[1][0,0,0,:,:])
        plt.show()

        # XZ Plane
        fig,axis = plt.subplots(ncols=2)
        plt.suptitle("XZ Planes")
        axis[0].imshow(values[0][0,0,:,0,:])
        axis[1].imshow(values[1][0,0,:,0,:])
        plt.show()
        

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # X = values[0][0][0][0,:,:]
        # print(X)
        # print()
        # Y = values[0][0][0][:,0,:]
        # Z = values[0][0][0][:,:,0]
        # print()
        # print(Y)
        # print()
        # print(Z)
        # ax.plot_surface(X, Y, Z)
        # # cset = ax.contour(X, Y, Z, zdir='z', cmap=cm.coolwarm)
        # # cset = ax.contour(X, Y, Z, zdir='x', cmap=cm.coolwarm)
        # # cset = ax.contour(X, Y, Z, zdir='y', cmap=cm.coolwarm)

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # plt.show()

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # X = values[1][0][0][0,:,:]
        # print(X)
        # print()
        # Y = values[1][0][0][:,0,:]
        # Z = values[1][0][0][:,:,0]
        # print()
        # print(Y)
        # print()
        # print(Z)
        # ax.plot_surface(X, Y, Z)
        # # cset = ax.contour(X, Y, Z, zdir='z', cmap=cm.coolwarm)
        # # cset = ax.contour(X, Y, Z, zdir='x', cmap=cm.coolwarm)
        # # cset = ax.contour(X, Y, Z, zdir='y', cmap=cm.coolwarm)

        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')

        # plt.show()

        # xs = values[1][0][0][0,:,:]
        # ys = values[1][0][0][:,0,:]
        # zs = values[1][0][0][:,:,0]

        # fig = plt.figure()
        # ax = axes3d.Axes3D(fig)
        # ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='hot')
        # plt.show()

        pass