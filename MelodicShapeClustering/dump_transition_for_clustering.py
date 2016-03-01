import numpy as np
import scipy.signal as sig
from scipy.cluster.vq import vq, kmeans, kmeans2, whiten
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hac
from collections import Counter
import math
eps = np.finfo(np.float).eps
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_PythonNew/batchProcessing'))

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_PythonNew/melodyProcessing'))

import basicOperations as BO
import batchProcessing as BP
import pitchHistogram as PH
import segmentation as seg
import transcription as ts
import copy
import pickle


def batch_proc(rootDir):
  """
  """
  generate_file_list(rootDir)
  generate_segment_id(fileList = 'fileList.txt', output_file = 'id_map.pkl', segExt = '.seg')
  get_transients(fileList = 'fileList.txt', map_file = 'id_map.pkl', segExt = '.seg', pitchExt = '.pitchSilIntrpPP', tonicExt = '.tonicFine')
  cluster_data(featureFile = 'transientShapes.npy', id_file = 'transientIds.npy', map_file = 'id_map.pkl', output_file = 'cluster_map.pkl')
  


def generate_file_list(rootDir):
  """
  """
  fileList = BP.generateFileList(rootDir, 'fileList.txt', ext = '.mp3')
  
  
def generate_segment_id(fileList, output_file, segExt = '.seg'):
  """
  """
  lines = open(fileList,'r').readlines()
  cnt = 1
  
  file_row_to_id_map = {}
  id_to_file_row = {}
  
  for ii, line in enumerate(lines):
    if not file_row_to_id_map.has_key(ii):
      file_row_to_id_map[ii]={}
      
    seg_filename = line.strip() + segExt
    seg_file = np.loadtxt(seg_filename)
    
    for jj in range(seg_file.shape[0]):
      if not file_row_to_id_map[ii].has_key(jj):
	file_row_to_id_map[ii][jj] = cnt
	id_to_file_row[cnt] = (ii, jj)
	cnt += 1
    
  pickle.dump({'file_row_to_id_map':file_row_to_id_map, 'id_to_file_row':id_to_file_row}, open(output_file,'w')) 
  
  
def find_ind(time_stamps, time):
  ind = np.argmin(np.abs(time_stamps-time))
  return ind

  
  
def get_transients(fileList, map_file, segExt = '.seg', pitchExt = '.pitchSilIntrpPP', tonicExt = '.tonicFine'):
  """
  """
  map_data = pickle.load(open(map_file,'r'))
  lines = open(fileList,'r').readlines()
  ids_data = []
  
  cnt = 0
  for ii, line in enumerate(lines):
    seg_filename = line.strip() + segExt
    seg_file = np.loadtxt(seg_filename)
    
    pitch, time, Hop = BO.readPitchFile(line.strip() + pitchExt)
    tonic = np.loadtxt(line.strip()  + tonicExt)
    pcents = BO.PitchHz2Cents(pitch, tonic)
    #print pitch
    
    for jj in range(seg_file.shape[0]):
      if seg_file[jj][2] == -20000:
	ids_data.append(map_data['file_row_to_id_map'][ii][jj])
	start_time = seg_file[jj][0]
	end_time = seg_file[jj][1]
	#trans_id = map_data[][ii][jj]
	
	start_ind = find_ind(time, start_time)
	end_ind = find_ind(time, end_time)
	
	segment = pitch[start_ind:end_ind]
	#print len(segment)
	if len(segment) >= 60:
	  segment_norm = polyfit_shapes_norm(segment)

	if cnt == 0:
	  aggregate = np.array([segment_norm])
	else:
	  aggregate = np.vstack((aggregate, segment_norm))
	cnt += 1
  
  print aggregate.shape
  #plt.show()
  
  np.save('transientIds',np.array(ids_data))
  np.save('transientShapes',aggregate)
  


def polyfit_shapes_norm(segment):
  """
  """
  segment = (segment - np.min(segment)) / np.ptp(segment, axis=0)
  #print segment
  samples = np.arange(len(segment))
  
  # calculate polynomial
  z = np.polyfit(samples,segment, 3)
  f = np.poly1d(z)
  
  # calculate new x's and y's
  x_new = np.linspace(samples[0], samples[-1], 100)
  y_new = f(x_new)
  
  #plt.plot(samples,segment,'o', x_new, y_new)
  plt.plot(np.arange(100), y_new)
  #plt.show()
  
  return y_new
      
  
def clustering_scipy_kmeans(features, n_clust = 8):
  """
  """
  whitened = whiten(features)
  print whitened.shape
  
  #initial = [kmeans(whitened,i) for i in range(8)]
  #plt.plot([var for (cent,var) in initial])
  #plt.show()
  
  #cent, var = initial[3]
  ##use vq() to get as assignment for each obs.
  #assignment,cdist = vq(whitened,cent)
  #plt.scatter(whitened[:,0], whitened[:,1], c=assignment)
  #plt.show()
  
  codebook, distortion = kmeans(whitened, n_clust)
  print codebook, distortion
  assigned_label, dist = vq(whitened, codebook)
  for ii in range(8):
    plt.subplot(4,2,ii)
    plt.plot(codebook[ii])
  plt.show()
  
  centroid, label = kmeans2(whitened, n_clust, minit = 'points')
  print centroid, label
  for ii in range(8):
    plt.subplot(4,2,ii)
    plt.plot(centroid[ii])
  plt.show()
  
  
  
def clustering_scipy_dendrogram(features, metric='euclidean', method = 'complete'):
  """
  """
  #x = pdist(features, metric)
  z = hac.linkage(features, method = method)
  d = hac.dendrogram(z, p=30, truncate_mode=None, color_threshold=None, get_leaves=True, orientation='top', labels=None, count_sort=False, distance_sort=False, show_leaf_counts=True, no_plot=False, no_labels=False, color_list=None, leaf_font_size=None, leaf_rotation=None, leaf_label_func=None, no_leaves=False, show_contracted=False, link_color_func=None)
  #plt.show()
  
  num_col = d['color_list']
  cnt = Counter(num_col)
  print cnt
  
  n_clust = 7
  clusters = hac.fcluster(z, n_clust, criterion='maxclust')
  #print clusters
  
  num_elem = Counter(clusters)
  print num_elem
  
  centroids = to_codebook(features, clusters)
  #print temp
  #for i in range(len(temp)):
    #plt.plot(temp[i])
  
  fig = plt.figure()
  for ii in range(len(centroids)):
    plt.subplot(4,2,ii)
    plt.plot(centroids[ii])
    plt.ylabel(np.array(ii+1))
  plt.show()
  
  np.save('centroids',np.array(centroids))
  
  return clusters, centroids
  
  
def to_codebook(X, part):
  """
  """
  codebook = []
  #temp = []
  
  for i in range(part.min(), part.max()+1):
    codebook.append(X[part == i].mean(0))
    
  #for i in [2,6]:
    #temp.append(X[part == i])

  return np.vstack(codebook)
  
  
  
def cluster_data(featureFile, id_file, map_file, output_file):
  """
  """
  features = np.load(featureFile)
  #print features
  
  #clustering_scipy_kmeans(features)
  cluster_ids, centroids = clustering_scipy_dendrogram(features)
  
  ids_data = np.load(id_file)
  
  map_data = pickle.load(open(map_file, 'r'))
  
  file_row_to_clusterid_map = {}
  for ii, c in enumerate(cluster_ids):
    id_transient = ids_data[ii]
    file_id, row_id = map_data['id_to_file_row'][id_transient]
    if not file_row_to_clusterid_map.has_key(file_id):
      file_row_to_clusterid_map[file_id]={}
    file_row_to_clusterid_map[file_id][row_id] = c
  
  pickle.dump(file_row_to_clusterid_map, open(output_file, 'w'))
  
  
  
def generate_full_transcription(fileList, clusterMap_file, segExt = '.seg', fullTransExt = '.fullTrans'):
  """
  """
  clusterMap_data = pickle.load(open(clusterMap_file, 'r'))
  
  lines = open(fileList,'r').readlines()
  
  for ii, line in enumerate(lines):
    seg_filename = line.strip() + segExt
    full_transcription_filename = line.strip() + fullTransExt
    seg_file = np.loadtxt(seg_filename)
    
    full_transcription_file = seg_file
    
    for jj in range(seg_file.shape[0]):
      if seg_file[jj][2] == -20000:
	full_transcription_file[jj][2] = clusterMap_data[ii][jj]
	
    #np.savetxt(full_transcription_filename, full_transcription_file)
    
    fid = open(full_transcription_filename,'w')
    for ii in full_transcription_file:
      fid.write("%f\t%f\t%d"%(ii[0],ii[1],ii[2]))
      fid.write('\n')
    fid.close()