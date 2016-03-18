import numpy as np
import scipy.signal as sig
from scipy.cluster.vq import vq, kmeans, kmeans2, whiten
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hac
from scipy.interpolate import interp1d
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
from test_SW_with_transients import *
import copy
import pickle
import string


LEVELS = range(-800, 2501, 100)
NOTES = ['G','m','M','P','d','D','n','N','S','r','R','g']*3
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


def visualizeGroundTruth(fileList, gtExt = '.gtruth', pitchExt = '.pitch', tonicExt = '.tonic'):
  """
  """
  song_str, st_seg, en_seg, spl_arr = readFullTransFile(fileList, fullTransExt = '.fullTrans')  
  segment, time_stamps, st, en, Hop, pcents = readGroundTruth(fileList, gtExt = gtExt, pitchExt = pitchExt, tonicExt = tonicExt)
  centroids = readCentroids(centroids_file = 'centroids.npy')
  
  st_gt, en_gt, str_gt = [0.0]*len(st), [0.0]*len(st), [0.0]*len(st)
  for ii in range(len(st)):
    st_gt[ii] = st_seg[find_ind(st_seg, st[ii])]
    en_gt[ii] = en_seg[find_ind(st_seg, en[ii])+1]
    str_gt[ii] = song_str[find_ind(st_seg, st[ii]):find_ind(st_seg, en[ii])]
  #print str_gt
  
  recons = get_quantized_ts(st_seg/Hop, en_seg/Hop, song_str, centroids, pcents)
  #plt.plot(recons,linewidth=2)
  #plt.ylim((-500,1700))
  #plt.show()
  
  stylized_pitch = np.array(recons, dtype=np.float)
  
  lines = open(fileList,'r').readlines()
  for ii, line in enumerate(lines):
    filename = line.strip()
    stylizedContourFilename = filename + '.stylizedPitch'
  
    fid = open(stylizedContourFilename,'w')
    for ii in range(len(stylized_pitch))[::2]:
      #print ii, (Hop*ii), stylized_pitch[ii]
      fid.write("%f\t%f"%((Hop*ii),stylized_pitch[ii]))
      fid.write('\n')
    fid.close()
  
  #plt.plot(time_stamps[1]*Hop, segment[1], linewidth=2)
  #plt.plot(time_stamps[1]*Hop, recons[time_stamps[1]], 'r', linewidth=3)
  #plt.ylim((-400,1200))
  #plt.show()
  
  gt_rec = []
  for ii in range(len(segment)):
    plt.subplot(2,2,ii+1)
    plt.plot(time_stamps[ii]*Hop, segment[ii])
    plt.plot(time_stamps[ii]*Hop, recons[time_stamps[ii]], 'r', linewidth=3)
    plt.ylim((-300,1100))
    
    gt_rec.append(recons[time_stamps[ii]])
    
  plt.show()
  
  return st_gt, en_gt, str_gt, gt_rec, recons
  
  
  
def get_quantized_ts(st, en, song_str, centroids, pcents):
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
	qts[int(round(st[ii]))+np.arange(len(time))] = polyfit_shapes_denorm(time,segment,song_str,ii)[:len(time)]
      else:
        qts[int(round(st[ii]))+np.arange(len(time))] = pcents[st[ii]:en[ii]]	
  return qts  


def polyfit_shapes_denorm(time,segment,song_str,ind):
  """
  """
  x = np.linspace(0, 0.99, num=len(segment))
  y = segment
  
  # calculate interpolated shape
  f = interp1d(x, y, kind='cubic', bounds_error=False)
  
  # calculate new x's and y's
  x_new = np.linspace(0, .99, num=len(time))
  y_new = f(x_new)
  
  y_sc = (y_new - y_new[0]) / (y_new[-1] - y_new[0])
  
  if song_str[ind-1] == -100000:
    song_str[ind-1] = song_str[ind-2]
  if song_str[ind+1] == -100000:
    song_str[ind+1] = song_str[ind+2]
  
  extent = song_str[ind+1] - song_str[ind-1]
  y_sc_norm = (y_sc*extent) + song_str[ind-1]
  
  #plt.subplot(2,2,1)
  #plt.plot(y)
  #plt.subplot(2,2,2)
  #plt.plot(y_new)
  #plt.subplot(2,2,3)
  #plt.plot(y_sc)
  #plt.subplot(2,2,4)
  #plt.plot(y_sc_norm)
  #plt.show()  
  
  return y_sc_norm
  
  
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
    count = 4
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
	
  return segment, time_stamps, st, en, Hop, pcents


def readCentroids(centroids_file = 'centroids.npy'):
  """
  """
  centroids = np.load(centroids_file)

  #for ii in range(len(centroids)):
    #plt.plot(range(100), centroids[ii])
  #plt.show()
  
  return centroids



def getDistanceBetweenCentroids(centroids_file = 'centroids.npy', metric = 'euclidean'):
  """
  """
  centroids = np.load(centroids_file)
  #print len(centroids)
  
  for ii in range(len(centroids)):
    plt.plot(range(100), centroids[ii], label = str(ii))
    
  plt.legend(loc = 'best')
  plt.show()
  
  d = pdist(centroids, metric)  
  X = squareform(d)
  #print X[0]
  print np.argsort(X)
  
  plt.imshow(X, interpolation = 'nearest', origin = 'lower', cmap = plt.get_cmap('OrRd'))
  plt.show()
  
  Y = X
  for ii in range(len(X)):
    Y[ii] = X[ii] / np.max(X[ii])
  #print Y
  
    hist = np.histogram(Y[ii], bins = 3, normed=0)[0]
    #print hist
  
  #plt.imshow(Y, interpolation = 'nearest', origin = 'lower', cmap = plt.get_cmap('OrRd'))
  #plt.show() 
  
  
  
def getQuertString(fileList):
  """
  """
  st_gt, en_gt, str_gt, gt_rec, recons = visualizeGroundTruth(fileList) 
  
  note_sym = str_gt
  for ii in range(len(str_gt)):
    for jj in range(len(str_gt[ii])):
      if str_gt[ii][jj] != -100000 and str_gt[ii][jj]%100 == 0:
	temp = CODES[LEVELS.index(str_gt[ii][jj])]
	note_sym[ii][jj] = ord(temp)
  #print note_sym	

  #print zip(st_gt, en_gt, note_sym)     
  #print str_gt, note_sym
  return note_sym
  
  
def getSearchString(fileList):
  """
  """
  song_str, st_seg, en_seg, spl_arr = readFullTransFile(fileList, fullTransExt = '.fullTrans')
  
  search_str = song_str
  for ii in range(len(song_str)):
    if song_str[ii] != -100000 and song_str[ii]%100 == 0:
      temp = CODES[LEVELS.index(song_str[ii])]
      search_str[ii] = ord(temp)
  #print search_str	
  
  return search_str
	

def getAlignment(fileList):
  """
  """
  song_str = getSearchString(fileList)
  note_sym = getQuertString(fileList)
  
  aligned = []
  for ii in range(len(note_sym)):
    query_str = note_sym[ii]
    print "Query index:", ii+1
    
    matches = testSW(query_str, song_str)
    aligned.append(matches)

  return aligned


def plotFoundMatches(fileList, pitchExt = '.pitchSilIntrpPP', tonicExt = '.tonicFine'):
  """
  """
  song_str, st_seg, en_seg, spl_arr = readFullTransFile(fileList, fullTransExt = '.fullTrans')
  centroids = readCentroids(centroids_file = 'centroids.npy')
  aligned = getAlignment(fileList)
  
  lines = open(fileList,'r').readlines()
  
  for ii, line in enumerate(lines):
    pitch, time, Hop = BO.readPitchFile(line.strip() + pitchExt)
    tonic = np.loadtxt(line.strip()  + tonicExt)
    pcents = BO.PitchHz2Cents(pitch, tonic)
  
  #recons = get_quantized_ts(st_seg/Hop, en_seg/Hop, song_str, centroids, pcents)
  
  for ii in range(len(aligned)):
    matches = aligned[ii]
    print "Query index:", ii+1
    count = 0
    for s, e in matches:
      st = st_seg[s]
      en = en_seg[e]
      #print st, en
      contour = pcents[st/Hop:en/Hop]
      plt.plot(np.arange(len(contour))*Hop, contour)
      plt.ylim((-300,1100))
      #plt.show()
      
      count += 1
    print "# motifs found: ", count  
    
    
    
    
def get_quantized_ts_onlySteadyNotes(fileList, pitchExt = '.pitch'):
  """
  """
  lines = open(fileList,'r').readlines()
  
  for ii, line in enumerate(lines):
    pitch, time, Hop = BO.readPitchFile(line.strip() + pitchExt)
    #tonic = np.loadtxt(line.strip()  + tonicExt)
    #pcents = BO.PitchHz2Cents(pitch, tonic)
  
  song_str, st_seg, en_seg, spl_arr = readFullTransFile(fileList, fullTransExt = '.fullTrans')
  
  st, en = st_seg/Hop, en_seg/Hop
  
  qts = np.array([None]*(max(en)))
  for ii in range(len(st)):
    if song_str[ii] != -100000 and song_str[ii]%100 == 0:
      qts[st[ii]:en[ii]] = song_str[ii]
      #print st[ii], en[ii], song_str[ii]
      
  #plt.plot(qts,linewidth=2)
  #plt.ylim((-500,1700))
  #plt.show()  
  
  #segment, time_stamps, st, en, Hop, pcents = readGroundTruth(fileList)
  
  #for ii in range(len(segment)):
    #plt.subplot(2,2,ii+1)
    #plt.plot(time_stamps[ii]*Hop, segment[ii])
    #plt.plot(time_stamps[ii]*Hop, qts[time_stamps[ii]], 'r', linewidth=3)
    #plt.ylim((-300,1100))
  #plt.show()
  
  stylized_pitch = np.array(qts, dtype=np.float)
  
  lines = open(fileList,'r').readlines()
  for ii, line in enumerate(lines):
    filename = line.strip()
    steadyContourFilename = filename + '.steadyPitch'
    
    fid = open(steadyContourFilename,'w')
    for ii in range(len(stylized_pitch))[::2]:
      #print ii, (Hop*ii), stylized_pitch[ii]
      fid.write("%f\t%f"%((Hop*ii),stylized_pitch[ii]))
      fid.write('\n')
    fid.close()
  
  return qts     