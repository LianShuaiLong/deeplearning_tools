#-*-coding:utf-8-*-
import sys
import os
import json
import numpy as np
import multiprocessing
import datetime
import pynvml
from pynvml import *

nvmlInit()

MEMORY_THESHOLD = 15 # GB
def get_aviliable_gpus():
    print ("Driver Version:", nvmlSystemGetDriverVersion())
    deviceCount = nvmlDeviceGetCount()
    GPU_AVILIABLE=[]
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memo_total = meminfo.total/(1024*1024*1024)
        memo_used = meminfo.used/(1024*1024*1024)
        if memo_total>=MEMORY_THESHOLD and memo_used/memo_total<=0.2:
           GPU_AVILIABLE.append(i)
    if len(GPU_AVILIABLE)==0: 
       print('No GPU Is Avilable!')
       sys.exit(0)
    else:
       print('Avilable GPUS:',GPU_AVILIABLE)
       return GPU_AVILIABLE

def get_search_space(cfg_path='search_space.json'):
    if not os.path.isfile(cfg_path):
        print('Cannot find search space file:{}'.format(cfg_path))
        sys.exit(0)
    search_space = json.loads(open(cfg_path,'r').read())
    print('search space\n:',search_space)
    return search_space

def get_params(search_space):
    params={}
    for k,v in search_space.items():
        if v["type"] == "uniform":
           value = np.random.uniform(v["value"][0],v["value"][1])
        elif v["type"] == "choice":
           value = np.random.choice(v["value"],1)
        params[k] = value[0] if isinstance(value,np.ndarray) else value
    return params
        
def start_running(*args,**kwargs):
    GPU_ID = int(args[0])
    print('start run train.py on GPU_ID:',GPU_ID)
    learning_rate = kwargs['learning_rate']
    batch_size = kwargs['batch_size']
    optimizer = kwargs['optimizer']
    max_number_of_steps = kwargs['max_number_of_steps']
    learning_rate_decay_type = kwargs['learning_rate_decay_type']
    print('pid:',os.getpid(),'running config:\nlearning rate:',learning_rate,'batch_size:',batch_size,'optimizer:',optimizer,'max_num_of_steps:',max_number_of_steps,'learning_rate_decay_type:',learning_rate_decay_type)
    today = datetime.date.today()
    checkpoint_dir = '{}/{}'.format(today,os.getpid())
    try: 
       os.makedirs(checkpoint_dir)
    except OSError:
       if not os.path.isdir(checkpoint_dir):
          raise
    os.popen('CUDA_VISIBLE_DEVICES={} python train.py --batch_size={} --max_number_of_steps={} --learning_rate={} --optimizer={} --checkpoint_dir={} --learning_rate_decay_type={}'.format(GPU_ID,batch_size,max_number_of_steps,learning_rate,optimizer,checkpoint_dir,learning_rate_decay_type),mode='w')

if __name__=='__main__':

   GPU_AVILIABLE = get_aviliable_gpus()
   search_space = get_search_space('search_space.json')
   pp1=[multiprocessing.Process(target = start_running,args=(str(GPU_ID)),kwargs=get_params(search_space)) for i,GPU_ID in enumerate(GPU_AVILIABLE)]

   for p in pp1:
       p.start()

   for p in pp1:
       p.join()
   
   

    

