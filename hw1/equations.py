import numpy as np


def flops(image_size, batch):
  S = np.asarray(image_size, dtype=np.float64)
  B = np.asarray(batch, dtype=np.float64)
  return B * (17712.0 * (S**2) + 313344.0)


def memory(image_size, batch):
  S = np.asarray(image_size, dtype=np.float64)
  B = np.asarray(batch, dtype=np.float64)
  return 4159872.0 + 44.0 * B * (S**2)


def bytes_moved(image_size, batch):
  S = np.asarray(image_size, dtype=np.float64)
  B = np.asarray(batch, dtype=np.float64)
  return 4159872.0 + 188.0 * B * (S**2)


def latency(image_size, batch, theta):
  theta_0, theta_bw, theta_gflops = theta

  fl = flops(image_size, batch)
  bm = bytes_moved(image_size, batch)

  t_mem = bm / theta_bw
  t_comp = fl / theta_gflops

  return theta_0 + np.maximum(t_mem, t_comp)


def energy(image_size, batch, theta_energy):
  theta_0, theta_bw, theta_gflops, P_idle, e_byte, e_flop = theta_energy

  theta_lat = (theta_0, theta_bw, theta_gflops)
  lat = latency(image_size, batch, theta_lat)

  bm = bytes_moved(image_size, batch)
  fl = flops(image_size, batch)

  return P_idle * lat + e_byte * bm + e_flop * fl
