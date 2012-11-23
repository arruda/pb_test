'''
Created on 22/11/2012

@author: arruda
'''

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection

def simple_ffn():
    "make a simple feed forward network"
    
    n = FeedForwardNetwork()
    
    inLayer = LinearLayer(2)
    hiddenLayer = SigmoidLayer(3)
    outLayer = LinearLayer(1)
    
    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addOutputModule(outLayer)
    
    n.addConnection(FullConnection(inLayer,hiddenLayer))
    n.addConnection(FullConnection(hiddenLayer,outLayer))
    
    n.sortModules()
    
#    print n
    
#    print n.activate([1, 2])
    for k,v in n.connections.items():
        for c in v:
            print c.name, c.params 
    
    print n.params
    
if __name__ == '__main__':
    simple_ffn()
    
    