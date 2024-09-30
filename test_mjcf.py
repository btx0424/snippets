import mujoco

urdf_path = "/home/btx0424/isaaclab/HIMLoco/legged_gym/resources/robots/aliengo/urdf/aliengo.urdf"

# connect to mjcf and save
mjcf_model = mujoco.MjModel.from_xml_path(urdf_path)
