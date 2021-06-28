#!/usr/bin/env python3

import numpy as np
from PIL import Image
import imageio
import OpenEXR
import struct
import os

def get_pointcloud(color_image,depth_image,camera_intrinsics):
    """ creates 3D point cloud of rgb images by taking depth information

        input : color image: numpy array[h,w,c], dtype= uint8
                depth image: numpy array[h,w] values of all channels will be same

        output : camera_points, color_points - both of shape(no. of pixels, 3)
    """

    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,image_width-1,image_width),
                                  np.linspace(0,image_height-1,image_height))
    camera_points_x = np.multiply(pixel_x-camera_intrinsics[0,2],depth_image/camera_intrinsics[0,0])
    camera_points_y = np.multiply(pixel_y-camera_intrinsics[1,2],depth_image/camera_intrinsics[1,1])
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x,camera_points_y,camera_points_z]).transpose(1,2,0).reshape(-1,3)

    color_points = color_image.reshape(-1,3)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind,:]
    color_points = color_points[valid_depth_ind,:]
    print(camera_points)
    return camera_points,color_points

def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated

    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))
    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc", xyz_points[i, 0], xyz_points[i, 1], xyz_points[i, 2],
                                        rgb_points[i, 0].tostring(), rgb_points[i, 1].tostring(),
                                        rgb_points[i, 2].tostring())))
    fid.close()




############################################################
#  Main
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='create point cloud from depth and rgb image.')
    parser.add_argument('--rgb_filename', required=False,default='/media/zirsha/New Volume/votenet-master/depth/table_chair_rgb.png',
                        help='path to the rgb image')
    parser.add_argument('--depth_filename', required=False,default='/media/zirsha/New Volume/votenet-master/depth/table_chair.png',
                        help="path to the depth image ")
    parser.add_argument('--output_directory', required=False,default='sky',
                        help="directory to save the point cloud file")
    parser.add_argument('--fx', required=False, type=float,default='1.5',
                        help="focal length along x-axis (longer side) in pixels")
    parser.add_argument('--fy', required=False, type=float,default='2.1',
                        help="focal length along y-axis (shorter side) in pixels")
    parser.add_argument('--cx', required=False, type=float,default='3.2',
                        help="centre of image along x-axis")
    parser.add_argument('--cy', required=False, type=float,default='2.3',
                        help="centre of image along y-axis")

    args = parser.parse_args()

    color_data = imageio.imread(args.rgb_filename)
    # color_data = np.asarray(im_color, dtype = "uint8")

    if os.path.splitext(os.path.basename(args.depth_filename))[1] == '.npy':
        depth_data = np.load(args.depth_filename)
    else:
        im_depth = imageio.imread(args.depth_filename)
        depth_data = im_depth[:,:,0] # values of all channels are equal


    # camera_intrinsics  = [[fx 0 cx],
    #                       [0 fy cy],
    #                       [0 0 1]]
    args.fx=550.0
    args.cx=150.0
    args.fy=401.0
    args.cy=150.0
    camera_intrinsics  = np.asarray([[args.fx, 0, args.cx], [0, args.fy, args.cy], [0, 0, 1]])

    filename = os.path.basename(args.rgb_filename)[:9] + '-pointCloud.ply'
    output_filename = os.path.join(args.output_directory, filename)

    print("Creating the point Cloud file at : ", output_filename )
    camera_points, color_points = get_pointcloud(color_data, depth_data, camera_intrinsics)

    write_pointcloud(output_filename, camera_points)
