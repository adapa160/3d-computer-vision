import os
import numpy as np

import cv2 as cv
import scipy.io as io
import sklearn
import math
from sklearn.feature_extraction import image
from PIL import Image

# Select the dataset
dataset = 'medieval_port'
# dataset = 'kitti'

experiment = 'medieval_port_exp_one'

os.makedirs(f'./{experiment}', exist_ok=True)

# While experimenting it is better to work with a lower resolution version of the image
# Since the dataset is of high resolution we will work with resized version.
# You can choose the reduction factor using the scale_factor variable.
scale_factor = 2

# Choose similarity metric
# similarity_metric = 'ncc'
similarity_metric = 'ssd'

# Outlier Filtering Threshold. You can test other values, too.
# This is a parameter which you have to select carefully for each dataset
outlier_threshold = 4

# Patch Size
# Experiment with other values like 3, 5, 7,9,11,15,13,17 and observe the result

patch_width = 7

if dataset == 'kitti':
    # Minimum and maximum disparies
    min_disparity = 0//scale_factor
    max_disparity = 150//scale_factor
    # Focal length
    calib = io.loadmat('./data/kitti/pose_and_K.mat')
    kmat = calib['K']
    #cam_pose = calib['Pose']
    baseline= calib['Baseline']
    kmat[0:2,0:2] /= scale_factor
    focal_length = kmat[0,0]
    left_img_path = './data/kitti/left.png'
    right_img_path = './data/kitti/right.png'

elif dataset == 'medieval_port':
    # Minimum and maximum disparies
    min_disparity = 0//scale_factor
    max_disparity = 80//scale_factor

    # Focal length
    kmat = np.array([[700.0000,   0.0000, 320.0000],
            [0.0000, 933.3333, 240.0000],
            [0.0000,   0.0000,   1.0000]], dtype=np.float32)
    kmat[:2, :] = kmat[:2, :]/scale_factor
    focal_length = kmat[0,0]
    baseline= 0.5
    left_img_path = './data/medieval_port/left.jpg'
    right_img_path = './data/medieval_port/right.jpg'
else:
    assert False, 'Dataset Error'

# Read Images
l_im = cv.imread(left_img_path, 1)
h, w, c = l_im.shape
resized_l_img = cv.resize(l_im, (w//scale_factor, h//scale_factor))
r_im = cv.imread(right_img_path, 1)
resized_r_img = cv.resize(r_im, (w//scale_factor, h//scale_factor))

#**************************************** Utilitiy Functions ************************************#
def ply_creator(input_3d, rgb_data=None, filename='dummy'):
    ''' Creates colored point cloud that you can visualise using meshlab.
    Inputs:
        input_3d: it sould have shape=[Nx3], each row is 3D coordinate of each point
        rgb_data: it sould have shape=[Nx3], each row is rgb color value of each point
        filename: file name for the .ply file to be created
    Note: All 3D points whose Z value is set 0 are ignored.
    '''
    assert (input_3d.ndim==2),"Pass 3d points as NumPointsX3 array "
    pre_text1 = """ply
    format ascii 1.0"""
    pre_text2 = "element vertex "
    pre_text3 = """property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header"""
    valid_points = input_3d.shape[0]- np.sum(input_3d[:, 2]==0)
    pre_text22 = pre_text2 + str(valid_points)
    pre_text11 = pre_text1
    pre_text33 = pre_text3
    fid = open(filename + '.ply', 'w')
    fid.write(pre_text11)
    fid.write('\n')
    fid.write(pre_text22)
    fid.write('\n')
    fid.write(pre_text33)
    fid.write('\n')
    for i in range(input_3d.shape[0]):
        # Check if the depth is not set to zero
        if input_3d[i,2]!=0:
            for c in range(3):
                fid.write(str(input_3d[i,c]) + ' ')
            if not rgb_data is None:
                for c in range(3):
                    fid.write(str(rgb_data[i,c]) + ' ')
            #fid.write(str(input_3d[i,2]))
            if i!=input_3d.shape[0]:
                fid.write('\n')
    fid.close()
    return True

def disparity_to_depth(disparity, baseline):
    """
    Converts disparity to depth.
    """
    inv_depth = (disparity+10e-5)/(baseline*focal_length)
    return 1/inv_depth

def write_depth_to_file(depth, f_name):
    """
    This function writes depth map as an image
    You can modify it, if you think of a better way to visualise depth/disparity
    You can also use it to save disparities
    """
    assert (depth.ndim==2),"Depth map should be a 2D array "

    depth = depth + 0.0001
    depth_norm = 255*((depth-np.min(depth))/np.max(depth)*0.9)
    cv.imwrite(f_name, depth_norm)

def copy_make_border(img, patch_width):
    """
    This function applies cv.copyMakeBorder to extend the image by patch_width/2
    in top, bottom, left and right part of the image
    Patches/windows centered at the border of the image need additional padding of size patch_width/2
    """
    offset = np.int(patch_width/2.0)
    return cv.copyMakeBorder(img,
                             top=offset, bottom=offset,
                             left=offset, right=offset,
                             borderType=cv.BORDER_REFLECT)

def extract_pathches(img, patch_width):
    '''
    Input:
        image: size[h,w,3]
    Return:
        patches: size[h, w, patch_width, patch_width, c]
    '''
    if img.ndim==3:
        h, w, c = img.shape
    else:
        h, w = img.shape
        c = 1
    img_padded = copy_make_border(img, patch_width)
    patches = image.extract_patches_2d(img_padded, (patch_width, patch_width))
    patches = patches.reshape(h, w, patch_width, patch_width, c)
    return patches

#**************************************** Utilitiy Functions ************************************#

def depth_to_3d(depth_map, kmat):
    """
    Input:
        depth_map: per pixel depth value, shape [h,w]
        kmat= marix of camera intrinsics, shape [3x3]
    Return: 3D coordinates, with shape [h, w, 3]
    1. First back-project the point from homogeneous image space to 3D,
    by multiplying it with inverse of the camera intrinsic matrix, inv(K)
    2. Then scale it by its depth.
    """
    height_depth_map = len(depth_map)
    width_depth_map = len(depth_map[0])
    threeDCoordinate = list()
    k_inv = np.linalg.inv(kmat)
    #coordinateArray = list()
    '''
    for i in range(height_depth_map):
        rowList = list()
        for j in range(width_depth_map):
            templist = list()
            templist.append(i)
            templist.append(j)
            templist.append(1)
            rowList.append(np.array(templist))
        coordinateArray.append(np.array(rowList))
    print("coordinateArray",coordinateArray)'''

    for i in range(height_depth_map):
        for j in range(width_depth_map):
            a = np.transpose(np.array([i,j,1]))
            threeDCoordinate.append(np.matmul(k_inv,np.transpose(np.array([i,j,1])))*depthMap[i][j])
        #print("threeDCoordinate",threeDCoordinate)
    return np.array(threeDCoordinate)

def pixelCostFunction(pixel1,pixel2):
    #print(pixel1," ",pixel2)
    #return math.sqrt((pixel1[0]-pixel2[0])^2+(pixel1[0]-pixel2[0])^2+(pixel1[0]-pixel2[0])^2)
    return np.abs(pixel1[0]-pixel2[0])+np.abs((pixel1[0]-pixel2[0]))+np.abs((pixel1[0]-pixel2[0]))

def pixelBasedDenseMatching(resized_l_img,resized_r_img):
    
    size_img_width = len(resized_l_img[0])
    size_img_height = len(resized_l_img)
    disparityMap = [ [0]*size_img_width for i in range(size_img_height)]
    
    print("disparitymap",disparityMap)
    for j in range(size_img_height):
        for i in range(size_img_width):
            minimum = pixelCostFunction(resized_l_img[j][i],resized_r_img[j][i])

            disparityIndex = [j,i,0]
            for d in range(min_disparity+1,max_disparity):
                
                if i-d > -1:
                    print(j,i)
                    pixelCost = pixelCostFunction(resized_l_img[j][i],resized_r_img[j][i-d])
                    if pixelCost < minimum:
                        minimum = pixelCost
                        disparityIndex  =  [j,i-d,d]


                if i+d < size_img_width:
                    pixelCost = pixelCostFunction(resized_l_img[j][i],resized_r_img[j][i+d])
                    if pixelCost < minimum:
                        minimum = pixelCost
                        disparityIndex  =  [j,i+d,d]
            disparityMap[disparityIndex[0]][disparityIndex[1]] = disparityIndex[2]
            #print("PixelCost",minimum,"disparityIndex",disparityIndex)
    #print("disparitymap",disparityMap)
    return disparityMap

def mask_outliers(similiarity_scores, sim_metric, threshold):
    '''
    Details are given in the exercise sheet.
    '''
    raise NotImplementedError
    
def getAverage(img, u, v, n):
    """img as a square matrix of numbers"""
    s = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += img[u+i][v+j]
    return float(s)/(2*n+1)**2

def getStandardDeviation(img, u, v, n):
    s = 0
    avg = getAverage(img, u, v, n)
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += (img[u+i][v+j] - avg)**2
    return (s**0.5)/(2*n+1)

def ssd(left_img, right_img, kernel, max_offset):
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)    
    w, h = left_img.size  
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
       
    kernel_half = int(kernel / 2)    
    offset_adjust = 255 / max_offset  
      
    for y in range(kernel_half, h - kernel_half):      
        print(".", end="", flush=True)  
        
        for x in range(kernel_half, w - kernel_half):
            best_offset = 0
            prev_ssd = 65534
            
            for offset in range(max_offset):               
                ssd = 0
                ssd_temp = 0                            
                
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        ssd_temp = int(left[y+v, x+u]) - int(right[y+v, (x+u) - offset])  
                        ssd += ssd_temp * ssd_temp              
                
                if ssd < prev_ssd:
                    prev_ssd = ssd
                    best_offset = offset
                            
            depth[y, x] = best_offset * offset_adjust
                              
    Image.fromarray(depth).save('./data/kitti/ssddepth.png')

def ncc(left_img, right_img, kernel, max_offset):
    left_img = Image.open(left_img).convert('L')
    left = np.asarray(left_img)
    right_img = Image.open(right_img).convert('L')
    right = np.asarray(right_img)    
    w, h = left_img.size  
    depth = np.zeros((w, h), np.uint8)
    depth.shape = h, w
    
    depth2 = np.zeros((w, h), np.float)
    depth2.shape = h, w
    
    stdDeviation1 = getStandardDeviation(left, 1, 1, kernel)
    stdDeviation2 = getStandardDeviation(right, 1, 1, kernel)
    avg1 = getAverage(left, 1, 1, kernel)
    avg2 = getAverage(right, 1, 1, kernel)
       
    kernel_half = int(kernel / 2) 
    stdD = stdDeviation1 * stdDeviation2 * (2*kernel+1)**2
    #offset_adjust = 255 / max_offset  
    ncc = 0   
    for y in range(kernel_half, h - kernel_half):      
        print(".", end="", flush=True)  
        
        for x in range(kernel_half, w - kernel_half):
                ncc_temp = 0                            
                        
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        ncc_temp += (left[y+v, x+u] - avg1)*(right[y+v, x+u] - avg2)
                        ncc= float(ncc_temp)/(stdD)
                        depth2[y, x] = ncc

    formatted = (depth2 * 255 / np.max(depth2)).astype('uint8')
    print(formatted)
    #img = Image.fromarray(formatted)
    #img.show()
    Image.fromarray(formatted).save('./data/kitti/depth.png')


def stereo_matching(img_left, img_right, patch_width):
    '''
    This is tha main function for your implementation.
    '''

    # This is the main function for your implementation
    # make sure you do the following tasks
    # 1. estimate disparity for every pixel in the left image
    stereo=cv.StereoBM_create(numDisparities=16, blockSize=13)
    imgL = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)
    disparity=stereo.compute(imgL,imgR)
    print(disparity)
    cv.imwrite("Output_ssd.png",disparity)



    # 2. convert estimated disparity to depth, save it to file
    # 3. convert depth to 3D points and save it as colored point cloud using the ply_creator function
    # 4. visualize the estimted 3D point cloud using meshlab
    raise NotImplementedError

if __name__=='__main__':
    stereo_matching(resized_l_img, resized_r_img, patch_width)
     
    
    pixelDisparityMap = np.array(pixelBasedDenseMatching(resized_l_img,resized_r_img))
    size_img_width = len(resized_l_img[0])
    size_img_height = len(resized_l_img)
    print("pixelDisparityMap",len(pixelDisparityMap), len(pixelDisparityMap[0]))
    print("depthMap",size_img_width,"img height",size_img_height)
    depthMap = np.full((size_img_height,size_img_width),0)

    print("depthMap",len(depthMap))

    write_depth_to_file(pixelDisparityMap,'disparitymapkitti'+str(scale_factor)+'.png')
    for i in range(size_img_height):
        for j in range(size_img_width):
            depthMap[i][j] = disparity_to_depth(pixelDisparityMap[i][j],baseline)
    #print("depthMap",depthMap)
    write_depth_to_file(depthMap,'depthmapkitti'+str(scale_factor)+'.png')
    
    ssd("./data/kitti/left.png", "./data/kitti/right.png", 7, 30)
    ncc("./data/kitti/left.png", "./data/kitti/right.png", 3, 30)
    
    # Practical Tip:
    # Use smaller image resolutions untill you get the solution
    # A naive appraoch would use three levels of for-loop
    #     for x in range(0, width):
    #         for y in range(0, height):
    #             for d in range(0, d_max):
    # Such methods might get prohibitively slow,
    # Therefore try to avoid for loops(as much as you can)
    # instead try to think of solutions that use multi-dimensional array operations,
    # for example you can have only one for loop: going over the disparities
    # First extract patches, as follows
    # left_patches = extract_pathches(img_left, patch_width)
    # right_patches = extract_pathches(img_right, patch_width)
    # for d in range(d_min, d_max):
    #     shifted_right_patches = shift the right_patches matrix by d pixels
    #     shifted_right_patches --> will be of shape [H, W, feat_size]
    #     left_patches --> will be of shape [H, W, feat_size]
    #     compute the ssd and ncc on multi-dimensional arrays of left_patches and shifted_right_patches