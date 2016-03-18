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

import basicOperations as BO
import batchProcessing as BP
import pitchHistogram as PH
import segmentation as seg
import transcription as ts
from test_SW_with_transients import *
import copy
import pickle
import string
from mutagen import easyid3
from mutagen.mp3 import MP3


def get_mbid_from_mp3(mp3_file):
   """
   fetch MBID form an mp3 file
   """
   try:
       mbid = easyid3.ID3(mp3_file)['UFID:http://musicbrainz.org'].data
   except:
       print "problem reading mbid for file %s\n" % mp3_file
       raise
   return mbid


def get_raga_id(mbid_raga_mapping):
  """
  """
  lines = open(mbid_raga_mapping,'r').readlines()
  
  mbid_to_ragaId_map = {}
  
  for ii, line in enumerate(lines):
    sline = line.split(',')
    sline = [s.strip() for s in sline]
    if not mbid_to_ragaId_map.has_key(sline[0]):
      mbid_to_ragaId_map[sline[0]] = sline[1]
  
  return mbid_to_ragaId_map



LEVELS = range(-800, 2501, 100)
NOTES = ['G','m','M','P','d','D','n','N','S','r','R','g',]*3
CODES = string.ascii_letters[:len(NOTES)] 




def batchProc(fileList, centroids_file = 'centroids.npy', audioExt = '.mp3', pitchExt = '.pitchSilIntrpPP', tonicExt = '.tonicFine', fullTransExt = '.fullTrans', gtExt = '.gtruth'):
  """
  """
  centroids = readCentroids(centroids_file)
    
  lines = open(fileList,'r').readlines()
  
  for ii, line in enumerate(lines):
    
    filename = line.strip()
    print "Processing file: %s" %filename
    
    # Read pitch data
    #----------------
    pitch, time, Hop = BO.readPitchFile(filename + pitchExt)
    tonic = np.loadtxt(line.strip()  + tonicExt)
    pcents = BO.PitchHz2Cents(pitch, tonic)
    
    # Read transcription
    #-------------------
    song_str, st_seg, en_seg = readFullTransFile(filename, fullTransExt = fullTransExt)
    
    # Read ground truth
    #------------------
    st_gt, en_gt, str_gt = visualizeGroundTruth(filename, pcents, time, Hop, song_str, st_seg, en_seg)
    
    # Get query strings
    #------------------
    note_sym = getQuertString(str_gt)
    
    # Get song string
    #----------------
    search_str = getSearchString(song_str)
    
    # Get aligned contour indices by SW
    #----------------------------------
    aligned = getAlignment(song_str, note_sym)
    
    # Get contour segments
    #---------------------
    plotFoundMatches(aligned, st_seg, en_seg, pcents, Hop)
    
    
    print "-------\nDone !!\n-------"
    
    
    
    
    

def batchProc_evalSet(fileList, centroids_file = 'centroids_16.npy', pitchExt = '.tpe', tonicExt = '.tonic', fullTransExt = '.fullTrans', gtExt = '.gtruth'):
  """
  """
  centroids = readCentroids(centroids_file)
    
  lines = open(fileList,'r').readlines()
  
  for ii, line in enumerate(lines):
    
    filename = line.strip()
    print "Processing file: %s" %filename
    
    # Read pitch data
    #----------------
    pitch, time, Hop = BO.readPitchFile(filename + pitchExt)
    tonic = np.loadtxt(line.strip()  + tonicExt)
    pcents = BO.PitchHz2Cents(pitch, tonic)
    
    # Read transcription
    #-------------------
    song_str, st_seg, en_seg = readFullTransFile(filename, fullTransExt = fullTransExt)
    
    # Read ground truth
    #------------------
    st_gt, en_gt, str_gt = visualizeGroundTruth(filename, pcents, time, Hop, song_str, st_seg, en_seg)
    
    # Get query strings
    #------------------
    note_sym = getQuertString(str_gt)
    
    # Get song string
    #----------------
    search_str = getSearchString(song_str)
    
    # Get aligned contour indices by SW
    #----------------------------------
    aligned = getAlignment(song_str, note_sym)
    
    # Get contour segments
    #---------------------
    plotFoundMatches(aligned, st_seg, en_seg, pcents, Hop)
    
    
    print "-------\nDone !!\n-------"    
        
        
        


def readCentroids(centroids_file = 'centroids.npy'):
  """
  """
  centroids = np.load(centroids_file)

  #for ii in range(len(centroids)):
    #plt.plot(range(100), centroids[ii])
  #plt.show()
  
  return centroids


def readFullTransFile(filename, fullTransExt = '.fullTrans'):
  """
  """
  full_transcription_filename = filename + fullTransExt
  full_transcription_file = np.loadtxt(full_transcription_filename)
  
  song_str = full_transcription_file[:,2].astype(int)
  st_seg = full_transcription_file[:,0]
  en_seg = full_transcription_file[:,1]     
  
  return song_str, st_seg, en_seg


def readGroundTruth(filename, pcents, time, gtExt = '.gtruth'):
  """
  Returns the start and end time of ground truth phrases in seconds
  """
  gt_filename = filename + gtExt
  gt_file = np.loadtxt(gt_filename)
  
  segment = []
  time_stamps = []
  count = 4
  st, en = [0.0]*count, [0.0]*count
  for ii in range(count):
    s, e = gt_file[ii][0], gt_file[ii][1]
    st[ii], en[ii] = s, e
    
    start_ind = find_ind(time, s)
    end_ind = find_ind(time, e)
    
    time_stamp = np.arange(start_ind, end_ind)
    pitch_vals = pcents[start_ind:end_ind]
    
    time_stamps.append(time_stamp)
    segment.append(pitch_vals)
      
  return segment, time_stamps, st, en


def find_ind(time_stamps, time):
  ind = np.argmin(np.abs(time_stamps-time))
  return ind


def visualizeGroundTruth(filename, pcents, time, Hop, song_str, st_seg, en_seg):
  """
  """
  segment, time_stamps, st, en = readGroundTruth(filename, pcents, time)
  
  st_gt, en_gt, str_gt = [0.0]*len(st), [0.0]*len(st), [0.0]*len(st)
  for ii in range(len(st)):
    st_gt[ii] = st_seg[find_ind(st_seg, st[ii])]
    en_gt[ii] = en_seg[find_ind(st_seg, en[ii])+1]
    str_gt[ii] = song_str[find_ind(st_seg, st[ii]):find_ind(st_seg, en[ii])]

  #for ii in range(len(segment)):
    #plt.subplot(2,2,ii+1)
    #plt.plot(time_stamps[ii]*Hop, segment[ii])
    #plt.ylim((-300,1100))
  #plt.show()
  
  return st_gt, en_gt, str_gt


def getQuertString(str_gt):
  """
  """  
  note_sym = str_gt
  for ii in range(len(str_gt)):
    for jj in range(len(str_gt[ii])):
      if str_gt[ii][jj] != -100000 and str_gt[ii][jj]%100 == 0:
	temp = CODES[LEVELS.index(str_gt[ii][jj])]
	note_sym[ii][jj] = ord(temp)

  return note_sym


def getSearchString(song_str):
  """
  """
  search_str = song_str
  for ii in range(len(song_str)):
    if song_str[ii] != -100000 and song_str[ii]%100 == 0:
      temp = CODES[LEVELS.index(song_str[ii])]
      search_str[ii] = ord(temp)	
  
  return search_str


def getAlignment(song_str, note_sym):
  """
  """
  aligned = []
  for ii in range(len(note_sym)):
    query_str = note_sym[ii]
    print "Query index:", ii+1
    
    matches = testSW(query_str, song_str)
    aligned.append(matches)

  return aligned


def plotFoundMatches(aligned, st_seg, en_seg, pcents, Hop):
  """
  """
  for ii in range(len(aligned)):
    matches = aligned[ii]
    print "Query index:", ii+1
    count = 0
    contours = []
    times = []
    for s, e in matches:
      st = st_seg[s]
      en = en_seg[e]
      #print st, en
      time = np.arange(int(st/Hop), int(en/Hop))
      contour = pcents[st/Hop:en/Hop]
      #plt.plot(np.arange(len(contour))*Hop, contour)
      #plt.ylim((-300,1100))
      #plt.show()
      contours.append(contour)
      times.append(time)
      
      count += 1
    print "# motifs found: ", count  
    
    for ii in range(8):
      plt.subplot(4,2,ii+1)
      plt.plot(times[ii]*Hop, contours[ii], linewidth = 2)
      plt.ylim((-300,1500))
    plt.show()  