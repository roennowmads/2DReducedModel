
import os, os.path
import struct

def processFile(filenameU, filenameV):
    dir = "input/"
    #filename = "u_00229.txt"
    fInU = open(dir + 'u/' + filenameU)
    fInV = open(dir + 'v/' + filenameV)
    filenameNoExt = os.path.splitext(filenameU)[0]

    fOut = open("output/" + filenameNoExt[2:] + ".bytes", "wb+")

    for line1, line2 in zip(fInU, fInV):
    #for i, line in enumerate(fIn):
        #print line
        x = float(line1)
        y = float(line2)
        
        #print str(x) + " " + str(y)
        
        dataX = struct.pack('f',x)
        dataY = struct.pack('f',y)
        fOut.write(dataX)
        fOut.write(dataY)
        
    fInU.close()
    fInV.close()
    fOut.close()


dirU = os.getcwd() + "/input/u"
dirV = os.getcwd() + "/input/v"
filenamesU = [name for name in os.listdir(dirU) if os.path.isfile(os.path.join(dirU, name))]
filenamesV = [name for name in os.listdir(dirV) if os.path.isfile(os.path.join(dirV, name))]
for filenameU, filenameV in zip(filenamesU, filenamesV):
    processFile(filenameU, filenameV)