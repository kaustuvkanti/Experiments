import numpy as np
import os
import os.path
import json

#song_str = np.array(([2, 107,   6, 105,   3, 105,   3, 104, 2, 107,   1, 112,   6, 107,   5, 105, 105,   3, 104,   5, 102,   7,  2, 107,   5, 105,   3, 105,   3, 104, 3, 111,   5, 112,   5, 107,   4, 105,   4, 104]))
#query_str = np.array(([2, 107, 6, 105, 3, 105, 3, 104]))

def testSW(query_str, song_str):
  """
  Test SW with previous settings with notes and transients as same
  """
  matches = smith_waterman(query_str, song_str, gap=affine_gap(.8,.8), sim=level_sim(3,1,-1,3))
  #print matches
  
  found = rem_overlap(matches)
  print "Found matches in:"
  print found

  #for s, e in matches:
    #subseq = song_str[s:e] 
    #print subseq
    
  return found  



################################################
# Similarity functions
# These functions take functions as input and output new functions based on 
# these parameters. The output functions are then passed to smith_waterman.
# The output functions take two inputs, each of which is a (symbol, duration)
# tuple

def const_sim(sim_weight, diff_weight):
  return lambda a, b : sim_weight if a[0] == b[0] else -diff_weight

def level_sim(same = 3, close = 1, far = -1, gap = 2): 
  def sim(a,b):
    i1 = a
    i2 = b
    #print i1, i2
    if i1 > 15 and i2 > 15:
      if i1 == i2:
	#print same
	return same
      elif abs(i1-i2) <= gap:
	#print close
	return close
      else:
	#print far
	return far
    elif i1 < 15 and i2 < 15 and i1 > 0 and i2 > 0:
      if i1 == i2:
	#print same
	return same
      elif abs(i1-i2) <= gap:
	#print close
	return close
      else:
	#print far
        return far
    elif i1 == -100000 or i2 == -100000:
      return -0.5
    else:
      #print "0"
      return 0 
  return sim 

################################################
# Gap functions
# These functions take functions as input and output new functions based on 
# these parameters. The output functions are then passed to smith_waterman.
# The output functions take a list of (symbol, duration) tuples as input

def linear_gap(m):
    return lambda x : - m * len(x)
    
def affine_gap(m, c):
    return lambda x : - (m * len(x) + c)

def thresh_gap(thres, c1, c2):
    '''
    Function gives a gap cost of c1 if duation is greater than thres else c2
    '''
    def func(symbs):
        return sum(map(lambda x:-c1 if x[1] > thres else -c2, symbs))
    return func

# def exp_gap

# def quad_gap

 
#################################################
# Smith waterman function taking time series symbol lengths into account

def _find_maxima(array):
    array = np.array(array)
    is_gt_left = np.concatenate(([True], array[1:] > array[:-1]))
    is_gt_right = np.concatenate((array[:-1] > array[1:], [True]))
    return np.where(is_gt_left * is_gt_right)[0]

def smith_waterman(s, t, thres = float('inf'), gap=linear_gap(1), sim=const_sim(2,1)):
    # inputs s and t are list of (symbol, duration)
    # d is a matrix of (score, prev_position)
    d = np.array([[(0,None) for j in xrange(len(t)+1)] for i in xrange(len(s)+1)])
    for i in xrange(1, len(s)+1):
        for j in xrange(1, len(t)+1):
            max_row = max((d[i-k,j][0] + gap(s[i-k:i]), (i-k,j)) \
                      for k in xrange(1, min(i, len(t))+1))
            max_col = max((d[i,j-k][0] + gap(t[j-k:j]), (i, j-k)) \
                      for k in xrange(1, min(j, len(s))+1))
            d[i,j] = max((0, None), 
                  (d[i-1, j-1][0] + sim(s[i-1], t[j-1]), (i-1,j-1)),
                  max_row, max_col)
    scores = [x[0] for x in d[-1]]
    indices = _find_maxima(scores)
    indices = filter(lambda x:scores[x]>max(scores)-thres, indices)
    en = [x-1 for x in indices]
    st = [0]*len(en)
    for i, k in enumerate(indices):
        start = (len(s), k)
        while d[start][1] is not None:
            start = d[start][1]
        st[i] = start[1]
    scrs = [scores[i] for i in indices]
    motifs = [(st[i], en[i]) for i, s in sorted(enumerate(scrs), key=lambda (i,s):(-s,i))]
    return motifs


#####################################################
# Function to remove overlapping motifs

def rem_overlap(motifs):
    st, en = zip(*motifs)
    times = sorted(st+en)
    res = []
    bmap = [0]*len(times)
    for s,e in motifs:
        i1, i2 = times.index(s),times.index(e)
        if sum(bmap[i1:i2+1]) == 0:
            res.append((s,e))
        for i in range(i1,i2+1):
            bmap[i] = 1
    return res  