import glob 
import sys, os
import re

def fetch_pickle(path):
  files = glob.glob(os.path.join(path,"*.pikle"))
  files = [os.path.split(file)[1] for file in files]
  saved_steps = [re.search(r'step[0-9]+', file).group() for file in files]
  saved_steps = [int(step.lstrip("step")) for step in saved_steps]
  if len(saved_steps)!=0:
    saved_steps, files = zip(*sorted(zip(saved_steps, files)))
  
  return saved_steps, files 