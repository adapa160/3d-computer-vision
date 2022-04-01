import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
from mpl_toolkits.mplot3d import Axes3D

def matchFeatures(data):
    ########################## Compute the fundamental matrix F and print it to the console. ###########################
    # intrinsic of camera 1
    k1 = data['K_1']  # shape=[3,3]
    k1_inv_trans = np.transpose(np.linalg.inv(k1))
    # translation of camera 1
    t1 = data['t_1']
    tx =t1[0,0]
    ty = t1[0,1]
    tz = t1[0,2]
    t1_cross = np.array([0,-tz,ty,tz,0,-tx,-ty,tx,0]).reshape(3,3)
    #print(t1_cross)
    # rotation of camera 1
    r1 = data['R_1']
    # intrinsic of camera 0
    k0 = data['K_0']

    # Fundamental matrix
    #F = k1_inv_trans.dot(t1_cross).dot(r1).dot(k0)
    F = np.matmul(k1_inv_trans,np.matmul(t1_cross,np.matmul(r1,np.linalg.inv(k0))))
    F_copy = copy.deepcopy(F)
    print('Fundamental matrix-',F)


    # For each chessboard feature in camera image 0 (Camera00.jpg), compute the corresponding epipolar
    # line l 0 = (a, b, c) in the image of camera 1 (Camera01.jpg).
    cornersCam0 = data['cornersCam0']
    cam0_homo = np.ones([len(cornersCam0), 3])
    cam0_homo[:,0:2] = cornersCam0
    #print('cam0_homo', cam0_homo.shape)

    epilines_1 = []
    for i in range(len(cam0_homo)):
        #print(cam0_homo[i])
        l = np.dot(F,cam0_homo[i])
        epilines_1.append(l)
    epilines_1 = np.asarray(epilines_1)



    ################# Draw each epipolar line computed above. ######################
    list_im = ['Camera00.jpg', 'Camera01.jpg']
    imgs = [Image.open(i) for i in list_im]
    # im1 = np.array(Image.open('Camera01.jpg'), dtype=np.uint8)
    # im0 = np.array(Image.open('Camera00.jpg'), dtype=np.uint8)
    fig1 = plt.figure()
    plt.imshow(imgs[1])
    plt.scatter(epilines_1[:,0], epilines_1[:, 1], s=20, c='g', edgecolors='g', alpha=0.5)
    fig1.savefig('epilines.jpg')


####################### Compute the matching feature as the one in cornersCam1 with minimal algebraic distance to l 0 .########################
    cornersCam1 = data['cornersCam1']
    cam1_homo = np.ones([len(cornersCam1), 3])
    cam1_homo[:, 0:2] = cornersCam1
    matched = np.zeros(cam1_homo.shape)

    for i in range(len(cam1_homo)):
        #print('cam1_homo',cam1_homo[i])

        minidx = 0
        for k in range(len(epilines_1)):
            dist = np.abs(np.matmul(cam1_homo[i],epilines_1[k]))

            #dist = np.linalg.norm(np.abs(cornersCam1[i] - X_Image[k, :-1]))
            if(k==0):
                mindist = dist

            if (dist < mindist):
                mindist = dist
                minidx = k
                # print(minidx)
                # print('dist', dist)
                # print('dist', epilines_1[k,:-1])

        matched[minidx] = cam1_homo[i]
        # print('out',minidx)
        # #print('epilines_1',X_Image)
        # plt.imshow(im0)
        # plt.scatter(cam0_homo[minidx, 0], cam0_homo[minidx, 1], s=20, c='g', edgecolors='g', alpha=0.5)
        # plt.show()
        # plt.imshow(im1)
        # plt.scatter(cam1_homo[i, 0], cam1_homo[i, 1], s=20, c='g', edgecolors='g', alpha=0.5)
        # plt.show()

######################### Connect corresponding points between the images with lines. ###############################
    imgs_comb = np.vstack((np.asarray(i) for i in imgs))
    cornersCam0 = data['cornersCam0']
    cornersCam1 = data['cornersCam1']
    matched[:, 1] = matched[:, 1] + 3168
    #print(imgs_comb.shape)
    fig2 = plt.figure()
    plt.imshow(imgs_comb)
    plt.scatter(cornersCam0[:, 0], cornersCam0[:, 1], s=20, c='g', edgecolors='g', alpha=0.5)
    plt.scatter(matched[:, 0], matched[:, 1] , s=20, c='g', edgecolors='g', alpha=0.5)
    for i in range(len(cornersCam0)):
        plt.plot([cornersCam0[i,0],matched[i,0]], [cornersCam0[i,1],matched[i,1]],color=np.random.rand(3,) )
    # plt.show()
    fig2.savefig('matches.jpg')

################## 3D Structure Reconstruction ############################
    p1 = np.identity(3)
    temp = (np.zeros(3)).reshape(3, 1)
    P0 = np.append(p1, temp, axis=1)

    P1 = np.append(r1, np.transpose(t1), axis=1)
    #print(P1,P2)

    # take one corresponding poin on each camera view
    #print(cam0_homo.shape,P1.shape)
    A1 = np.asarray([np.cross(point0,P0,axis=0)[0:2] for point0 in cam0_homo])
    A2 = np.asarray([np.cross(point1, P1, axis=0)[0:2] for point1 in cam1_homo])
    # print('A1',A1.shape)
    # print('A2', A2.shape)

    X_world = []
    for a,b in zip(A1,A2):
        # Stack equations for all camera views:
        A = np.append(a, b, axis=0)

        #Solution for X is eigenvector of A T A corresponding to smallest eigenvalue
        A_sq = np.dot(np.transpose(A),A)

        #print('A', A_sq.shape)
        eigValues, eigVectors = np.linalg.eig(A_sq)
        minidx = np.argmin(eigValues,axis=0)
        X_3D = eigVectors[minidx]
        X_3D = X_3D/X_3D[3]
        X_world.append(X_3D[0:3])

    # plot the 3D points
    X_world = np.asarray(X_world)
    fig3 = plt.figure()
    ax = fig3.add_subplot((111), projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # for b in bones_3d:
    #     ax.plot(joints[b, 0], joints[b, 1], joints[b, 2], linewidth=1.0, c='b')

    ax.scatter(X_world[:, 0], X_world[:, 1], X_world[:, 2], c="b", edgecolors='b')
    #ax.scatter(0,0,0, c="r", edgecolors='r')

    ax.scatter(t1[0,0], t1[0,1], t1[0,2], c="r", edgecolors='r')
    ax.scatter(0, 0, 0, c="r", edgecolors='r')
    plt.show()


if __name__ == '__main__':
    base_folder = ''
    # load the data


    data = io.loadmat(base_folder+'data.mat')
    # for d in data:
    #     print( d)
    #     print(data[d])


    matchFeatures(data)
