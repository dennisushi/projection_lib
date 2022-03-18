import numpy as np
import torch

def convert_depth_frame_to_pointcloud_t(depth_image, K ):
  # This function was taken from the interwebs, modified
  # to work with ROS intrinsic matrix
  """
  Convert the depthmap to a 3D point cloud
  Parameters:
  -----------
  depth_frame 	  : 2D ndarray, The depth_frame containing the depth map
  camera_intrinsics : 3x3 ndarray, The intrinsic values of the imager 
            in whose coordinate system the depth_frame is 
            computed
  Return:
  ----------
  x : array, The x values of the pointcloud in meters
  y : array, The y values of the pointcloud in meters
  z : array, The z values of the pointcloud in meters
  """
	
  [height, width] = depth_image.shape
  fx = K[0,0] 
  fy = K[1,1] 
  ppx = K[0,2]
  ppy = K[1,2]

  ny = torch.linspace(0, height-1, height).float()
  nx = torch.linspace(0,width-1,width).float()

  v,u = torch.meshgrid(nx, ny, indexing='xy')  # input xy = vu

  x_over_d = (v.flatten() - ppx)/fx
  y_over_d = (u.flatten() - ppy)/fy

  z = depth_image.float().flatten() # / 1000
  x = torch.multiply(x_over_d,z)
  y = torch.multiply(y_over_d,z)

  # Filter
  xf = x[torch.nonzero(z)][:,0]
  yf = y[torch.nonzero(z)][:,0]
  zf = z[torch.nonzero(z)][:,0]

  return xf, yf, zf

def convert_depth_frame_to_pointcloud(depth_image, K ):
  # This function was taken from the interwebs, modified
  # to work with ROS intrinsic matrix
  """
  Convert the depthmap to a 3D point cloud
  Parameters:
  -----------
  depth_frame 	  : 2D ndarray, The depth_frame containing the depth map
  camera_intrinsics : 3x3 ndarray, The intrinsic values of the imager 
            in whose coordinate system the depth_frame is 
            computed
  Return:
  ----------
  x : array, The x values of the pointcloud in meters
  y : array, The y values of the pointcloud in meters
  z : array, The z values of the pointcloud in meters
  """
  if torch.is_tensor(depth_image): return convert_depth_frame_to_pointcloud_t(depth_image, K)

  [height, width] = depth_image.shape
  fx = K[0,0] 
  fy = K[1,1] 
  ppx = K[0,2]
  ppy = K[1,2]

  nx = np.linspace(0, width-1, width)
  ny = np.linspace(0, height-1, height)

  v,u = np.meshgrid(nx, ny)
  x = (v.flatten() - ppx)/fx
  y = (u.flatten() - ppy)/fy

  z = depth_image.flatten() # / 1000
  x = np.multiply(x,z)
  y = np.multiply(y,z)

  x = x[np.nonzero(z)]
  y = y[np.nonzero(z)]
  z = z[np.nonzero(z)]

  return x, y, z

def apply_homogenous_transform_to_points_t(tf, pts):
  ''' Given a homogenous tf matrix and a 3xN Torch tensor
  of points, apply the tf to the points to produce
  a new 3xN array of points.
  :param tf: 4x4 torch tensor of matching dtype to pts
  :param pts: 3xN torch tensor of matching dtype to tf
  :return: 3xN torch tensor of matching dtype to tf and pts'''
  return ((tf[:3, :3].mm(pts).T) + tf[:3, 3]).T


def apply_homogenous_transform_to_points(tf, pts):
  ''' Given a homogenous tf matrix and a 3xN NP array
  of points, apply the tf to the points to produce
  a new 3xN array of points.
  :param tf: 4x4 numpy array of matching dtype to pts
  :param pts: 3xN numpy array of matching dtype to tf
  :return: 3xN numpy array of matching dtype to tf and pts'''
  if torch.is_tensor(pts): 
    return apply_homogenous_transform_to_points_t(tf,pts)
  return ((tf[:3, :3].dot(pts).T) + tf[:3, 3]).T
  
def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, K):
	# This function was taken from the interwebs
	"""
	Convert the depth and image point information to metric coordinates
	Parameters:
	-----------
	depth 	 	 		: double, The depth value of the image point
	pixel_x 	  	 	: double, The x value of the image coordinate
	pixel_y 	  	 	: double, The y value of the image coordinate
	camera_intrinsics 	: The intrinsic values of the imager in whose 
						  coordinate system the depth_frame is computed
	Return:
	----------
	X : double, The x value in meters
	Y : double, The y value in meters
	Z : double, The z value in meters
	"""
	#X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx *depth
	#Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy *depth
	X = (pixel_x - K[0,2])/K[0,0] *depth
	Y = (pixel_y - K[1,2])/K[1,1] *depth
	return X, Y, depth


def pixel_cam_to_cam( u, v, depth_uv, world_to_cam1, world_to_cam2, K1, K2 ,
                    return_depth = False):
    # u corresponds to columns
    # v corresponds to rows

    # CAMERA 1 PIXELS to CAMERA 1 FRAME 
    pt3d = convert_depth_pixel_to_metric_coordinate(depth_uv, u, v, K1)
    pt3d = np.array(pt3d)
    assert pt3d.shape[0] == 3
    # CAMERA 1 FRAME points to WORLD FRAME
    pt3d = apply_homogenous_transform_to_points(world_to_cam1, pt3d)
    # WORLD FRAME to CAMERA 2 FRAME
    cam2_to_world = np.linalg.inv(world_to_cam2)
    pt3d = apply_homogenous_transform_to_points(cam2_to_world, pt3d)
    # CAMERA 2 FRAME to CAMERA 2 PIXELS
    u_est2 = (pt3d[0,:] / pt3d[2,:] ) * K2[0,0] + K2[0,2]
    v_est2 = (pt3d[1,:] / pt3d[2,:] ) * K2[1,1] + K2[1,2]
    z_est2 = pt3d[2,:]
    
    def cascading_or(conditions):
        out = conditions[0]
        for c in conditions[1:]:
            out = np.logical_or(out, c)
        return out
        
    invalid = cascading_or( [u_est2<0, u_est2>640, v_est2<0, v_est2>480] )
    
    u_est2 = u_est2 [np.logical_not(invalid)]
    v_est2 = v_est2 [np.logical_not(invalid)]
    
    #uv_cam1_in_cam2 = np.stack([u2,v2]).T
    #uv_cam1_in_cam2 = uv_cam1_in_cam2.astype(np.uint16)
    if return_depth:
      z_est2 = z_est2 [np.logical_not(invalid)]
      return u_est2, v_est2, z_est2, invalid

    return u_est2, v_est2, invalid
