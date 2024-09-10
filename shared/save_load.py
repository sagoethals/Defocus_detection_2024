#----------------------------------------------------------------------------------------
#SAVING BINARY OBJECTS DATA
#need to automate data folder creation
#----------------------------------------------------------------------------------------
import pickle as pickle
import os as os

__all__ = ['save_obj', 'load_obj', 'load_py2_obj', 'getfiles']

def save_obj(obj, name ):
    if name[-4:]=='.pkl':
        name = name[:-4]
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    if name[-4:]=='.pkl':
        name = name[:-4]
    #~ try:
        #~ return pk5.dumps(name+'pkl', protocol=5)
    #~ except:
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_py2_obj(name ):
    if name[-4:]=='.pkl':
        name = name[:-4]
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f,encoding = 'latin1')
        


def getfiles(startdir, phrase, function) :
    files = []
    for dirpath, dirnames, filenames in os.walk(startdir) :
        if function == 'find' :
            for filename in [f for f in filenames if f.find(phrase)!=-1]:
                files.append(os.path.join(dirpath, filename))
        elif function == 'endswith' :
            for filename in [f for f in filenames if f.endswith(phrase)]:
                files.append(os.path.join(dirpath, filename))
        else :
            assert function=='endswith' or function=='find'
    return files
