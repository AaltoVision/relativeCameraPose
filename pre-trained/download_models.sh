rm hybrid*
# HybridCNN .prototxt and .caffemodel files
wget http://places.csail.mit.edu/model/hybridCNN_upgraded.tar.gz
gunzip < hybridCNN_upgraded.tar.gz | tar xvf -
rm *.csv *.binaryproto *.tar.gz
