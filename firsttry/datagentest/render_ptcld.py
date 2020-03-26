import trimesh

def read_obj(path):
    """read obj file, normalized to be zero centered and scale to unit cube
    
    Arguments:
        path {str} -- path to the .obj file
    """
    mesh = trimesh.load_mesh(path)
    mesh -= mesh.vertices.mean(axis = 0)
    bbox_max = abs(mesh.vertices).max(axis = 0)
    mesh.vertices /= bbox_max
    return mesh
    
    

def sample_k_points(mesh, n_points=1000):
    """sample K points on a mesh 
    
    Arguments:
        mesh {mesh object} -- the mesh for sampling
    
    Keyword Arguments:
        n_points {int} -- number of points to be sampled (default: {1000})
    """
    points, face_ids = trimesh.sample.sample_surface(mesh, n_points)
    return points, face_ids


def calculate_point_cloud_visibility(point_cloud, rotation_matrix):
    pass


def render_point_trajectory(mesh, traj, output_path='./', time_range=[0, 1], steps=1000):
    pass
