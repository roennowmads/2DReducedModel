
import os, os.path
import struct

def processFile(filename):
    dir = "x/"
    #filename = "u_00229.txt"
    fIn = open(dir + "" + filename)
    filenameNoExt = os.path.splitext(filename)[0]

    fOut = open(dir + "output/" + filenameNoExt + ".bytes", "wb+")

    for i, line in enumerate(fIn):
        #print line
        x = float(line)
        dataX = struct.pack('f',x)
        fOut.write(dataX)
        
    fIn.close()
    fOut.close()


dir = os.getcwd() + "/x"
filenames = [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]
for filename in filenames:
    processFile(filename)