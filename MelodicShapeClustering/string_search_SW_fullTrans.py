import numpy as np
import scipy.signal as sig
from scipy.cluster.vq import vq, kmeans, kmeans2, whiten
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hac
from scipy.interpolate import interp1d
from collections import Counter
import math
eps = np.finfo(np.float).eps
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_PythonNew/batchProcessing'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Library_PythonNew/melodyProcessing'))

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../ISMIR-2015_Systems/NoteBasedMethod'))

import basicOperations as BO
import batchProcessing as BP
import pitchHistogram as PH
import segmentation as seg
import transcription as ts
import smith_expts_N4 as SW
import copy
import pickle
import string


LEVELS = range(-800, 2501, 100)
NOTES = ['S','r','R','g','G','m','M','P','d','D','n','N']*3
CODES = string.ascii_letters[:len(NOTES)] 


def readFullTransFile(fileList, fullTransExt = '.fullTrans'):
  """
  """
  lines = open(fileList,'r').readlines()
  
  for ii, line in enumerate(lines):
    full_transcription_filename = line.strip() + fullTransExt
    full_transcription_file = np.loadtxt(full_transcription_filename)
    
    song_str = full_transcription_file[:,2].astype(int)
    st_seg = full_transcription_file[:,0]
    en_seg = full_transcription_file[:,1]
    
    spl_arr = np.split(song_str, np.where(song_str == -100000)[0])
    
    for jj in range(len(spl_arr)):
      if spl_arr[jj].size != 0:
	spl_arr[jj] = spl_arr[jj][1:]
	
      count = 0
      for ii in range(len(spl_arr[jj])):
	if spl_arr[jj][ii]%100 == 0:
	  count += 1     
    
  return song_str, st_seg, en_seg, spl_arr


def find_ind(time_stamps, time):
  ind = np.argmin(np.abs(time_stamps-time))
  return ind


def visualizeGroundTruth(fileList):
  """
  """
  song_str, st_seg, en_seg, spl_arr = readFullTransFile(fileList, fullTransExt = '.fullTrans')  
  segment, time_stamps, st, en, Hop = readGroundTruth(fileList)
  centroids = readCentroids(centroids_file = 'centroids.npy')
  
  st_gt, en_gt, str_gt = [0.0]*len(st), [0.0]*len(st), [0.0]*len(st)
  for ii in range(len(st)):
    st_gt[ii] = st_seg[find_ind(st_seg, st[ii])]
    en_gt[ii] = en_seg[find_ind(st_seg, en[ii])+1]
    str_gt[ii] = song_str[find_ind(st_seg, st[ii]):find_ind(st_seg, en[ii])]
  #print str_gt
  
  recons = get_quantized_ts(st_seg/Hop, en_seg/Hop, song_str, centroids)
  #plt.plot(recons)
  #plt.show()
  
  for ii in range(len(segment)):
    ax = plt.subplot(2,3,ii)
    plt.plot(time_stamps[ii]*Hop, segment[ii])
    plt.plot(time_stamps[ii]*Hop, recons[time_stamps[ii]], 'r', linewidth=3)
    ax.set_ylim([-200,900])
  #plt.show()
  
  return st_gt, en_gt, str_gt
  
  
  
def get_quantized_ts(st, en, song_str, centroids):
  """
  """
  qts = np.array([None]*(max(en)))
  for ii in range(len(st)):
    if song_str[ii] != -100000 and song_str[ii]%100 == 0:
      qts[st[ii]:en[ii]] = song_str[ii]
      #print st[ii], en[ii], song_str[ii]
    elif song_str[ii] != -100000 and song_str[ii]%100 != 0:
      time = np.arange(st[ii],en[ii])
      segment = centroids[song_str[ii]-1]
      if len(time) >= 60:
	qts[int(round(st[ii]))+np.arange(len(time))] = polyfit_shapes_denorm(time,segment)[:len(time)]
  return qts  


def polyfit_shapes_denorm(time,segment):
  """
  """
  #segment = (segment - np.min(segment)) / np.ptp(segment, axis=0)
  #print segment
  x = np.linspace(segment[0], segment[-1], num=len(segment))
  y = segment
  
  # calculate interpolated shape
  f = interp1d(x, y, kind='cubic', bounds_error=False)
  
  # calculate new x's and y's
  x_new = np.linspace(segment[0], segment[-1], num=len(time))
  y_new = f(x_new)
  
  #print len(x_new), len(y_new)
  #plt.plot(x, y, 'o', x_new, y_new, '-')
  #plt.plot(np.arange(len(y_new)), y_new)
  #plt.show()
  
  return y_new*200
  
  
def readGroundTruth(fileList, gtExt = '.gtruth', pitchExt = '.pitchSilIntrpPP', tonicExt = '.tonicFine'):
  """
  Returns the start and end time of ground truth phrases in seconds
  along with the label representing the annotated name of the phrase
  """
  lines = open(fileList,'r').readlines()
  
  for ii, line in enumerate(lines):
    gt_filename = line.strip() + gtExt
    gt_file = np.loadtxt(gt_filename)
    
    pitch, time, Hop = BO.readPitchFile(line.strip() + pitchExt)
    tonic = np.loadtxt(line.strip()  + tonicExt)
    pcents = BO.PitchHz2Cents(pitch, tonic)
    
    segment = []
    time_stamps = []
    count = 5
    st, en = [0.0]*count, [0.0]*count
    for ii in range(count):
      s, e = gt_file[ii][0], gt_file[ii][1]
      st[ii], en[ii] = s, e
      
      start_ind = find_ind(time, s)
      end_ind = find_ind(time, e)
      #print start_ind, end_ind
      
      time_stamp = np.arange(start_ind, end_ind)
      pitch_vals = pcents[start_ind:end_ind]
      
      time_stamps.append(time_stamp)
      segment.append(pitch_vals)	
	
  return segment, time_stamps, st, en, Hop


def readCentroids(centroids_file = 'centroids.npy'):
  """
  """
  centroids = np.load(centroids_file)

  #for ii in range(len(centroids)):
    #plt.plot(range(100), centroids[ii])
  ##plt.show()
  
  return centroids
  
  
def getQuertString(fileList):
  """
  """
  st_gt, en_gt, str_gt = visualizeGroundTruth(fileList) 
  
  note_sym = str_gt
  for ii in range(len(str_gt)):
    for jj in range(len(str_gt[ii])):
      if str_gt[ii][jj] != -100000 and str_gt[ii][jj]%100 == 0:
	temp = CODES[LEVELS.index(str_gt[ii][jj])]
	note_sym[ii][jj] = ord(temp)

  #print zip(st_gt, en_gt, note_sym)     
  print str_gt, note_sym