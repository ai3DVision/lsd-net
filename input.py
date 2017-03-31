import cv2
import random
import numpy as np
import time
import multiprocessing as mp
import globals as g_

W = H = 256

class Shape:
    def __init__(self, list_file):
        with open(list_file) as f:
            self.label = int(f.readline())
            self.V = int(f.readline())
            view_files = [l.strip() for l in f.readlines()]
        
        self.views = self._load_views(view_files, self.V)
        self.done_mean = False
        

    def _load_views(self, view_files, V):
        views = []
        for f in view_files:
            im = cv2.imread(f)
            #print(f)
            im = cv2.resize(im, (W, H))
            # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) #BGR!!
            assert im.shape == (W,H,3), 'BGR!'
            im = im.astype('float32')
            views.append(im)
        views = np.asarray(views)
        return views
    
    def subtract_mean(self):
        if not self.done_mean:
            mean_bgr = (104., 116., 122.)
            for i in range(3):
                self.views[:,:,:,i] -= mean_bgr[i]
            
            self.done_mean = True
    
    def crop_center(self, size=(224,224)):
        w, h = self.views.shape[1], self.views.shape[2]
        wn, hn = size
        left = round(w / 2 - wn / 2)
        top = round(h / 2 - hn / 2)
        right = left + wn
        bottom = top + hn
        self.views = self.views[:, left:right, top:bottom, :]
    
    
class Dataset:
    def __init__(self, listfiles, labels, subtract_mean, V):
        self.listfiles = listfiles
        self.labels = labels
        self.shuffled = False
        self.subtract_mean = subtract_mean
        self.V = V
        print('dataset inited')
        print('  total size:', len(listfiles))
    
    def shuffle(self):
        z = zip(self.listfiles, self.labels)
        if type(z) == zip:
            z = list(z)
        random.shuffle(z)
        self.listfiles, self.labels = [list(l) for l in zip(*z)]
        self.shuffled = True

    def batches(self, batch_size):
        for x,y in self._batches_simple(self.listfiles, batch_size):
            yield x,y
        
    def sample_batches(self, batch_size, n):
        listfiles = random.sample(self.listfiles, n)
        for x,y in self._batches_fast(listfiles, batch_size):
            yield x,y

    def _batches_simple(self,listfiles,batch_size):
    	n = len(listfiles) 
    	for i in range(0,n,batch_size):
            starttime = time.time()
            lists = listfiles[i : i+batch_size]
            x = np.zeros((batch_size, 224, 224, 3))
            y = np.zeros(batch_size)

            for j,l in enumerate(lists):
                s = Shape(l)
                s.crop_center()
                if self.subtract_mean:
                    s.subtract_mean()
                    x[j,...] = s.views[1,...]
                    y[j] = s.label
            yield x,y

    def _batches(self, listfiles, batch_size):
        n = len(listfiles)
        for i in range(0, n, batch_size):
            starttime = time.time()

            lists = listfiles[i : i+batch_size]
            x = np.zeros((batch_size,self.V, 224, 224, 3)) 
            y = np.zeros(batch_size)

            for j,l in enumerate(lists):
                s = Shape(l)
                s.crop_center()
                if self.subtract_mean:
                    s.subtract_mean()
                x[j, ...] = s.views
                y[j] = s.label
            
            print('load batch time:', time.time()-starttime, 'sec')
            yield x, y
    
    def _load_shape(self, listfile):
        s = Shape(listfile)
        s.crop_center()
        if self.subtract_mean:
            s.subtract_mean()
        return s 

    def _batches_fast(self, listfiles, batch_size):
        subtract_mean = self.subtract_mean
        n = len(listfiles)

        def load(listfiles, q):                    
            for l in listfiles:
                q.put(self._load_shape(l))

            # indicate that I'm done
            q.put(None)
            q.close()

        # This must be larger than twice the batch_size
        q = mp.Queue(maxsize=g_.INPUT_QUEUE_SIZE)

        # background loading Shapes process
        p = mp.Process(target=load, args=(listfiles, q))
        # daemon child is killed when parent exits
        p.daemon = True
        p.start()


        x = np.zeros((batch_size, self.V, 224, 224, 3)) 
        y = np.zeros(batch_size)

        for i in xrange(0, n, batch_size):
            starttime = time.time()
            
            # print 'q size', q.qsize() 

            for j in xrange(batch_size):
                s = q.get()

                # queue is done
                if s == None: 
                    break
                
                x[j, ...] = s.views
                y[j] = s.label 
            
            # print 'load batch time:', time.time()-starttime, 'sec'
            yield x, y

    def size(self):
        """ size of listfiles (if splitted, only count 'train', not 'val')"""
        return len(self.listfiles)



