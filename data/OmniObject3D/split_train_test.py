import os
import glob
import numpy as np

def write( samplelist, filename ):
    f = open( filename, 'w' )
    for i in range( len( samplelist ) ):
        f.write( samplelist[i][0] + " " + str( samplelist[i][1] ) + "\n" )
    f.close()

root="1024_ply"
dirlist = glob.glob( root + "/*" )
dirlist = sorted( dirlist )
num_classes = len( dirlist )

P = 0.8
train_samples = []
test_samples = []

for i in range( num_classes ):
    class_id = i

    samplelist = glob.glob( dirlist[ i ] + "/*" )
    samplelist = sorted( samplelist )
    num_samples = len( samplelist )
    samplelist_new = []

    # Some directories do not contain a ply file.
    # Thus, such empty directories are excluded from dataset creation.
    for j in range( num_samples ):
        exist = os.path.exists( samplelist[ j ] + "/pcd_1024.ply" )
        if( exist ):
            samplelist_new.append( samplelist[ j ] )

    samplelist = samplelist_new
    num_samples = len( samplelist_new )

    # The first "P*num_samples" samples are included in the training dataset
    # The remaining samples are included in the testing dataset
    num_train = int( np.floor( P * num_samples ) )
    trainlist = samplelist[ 0:num_train ]
    testlist = samplelist[ num_train: ]
    for j in range( len( trainlist ) ):
        train_samples.append( ( trainlist[ j ], class_id ) )
    for j in range( len( testlist ) ):
        test_samples.append( ( testlist[ j ], class_id ) )

# write to files
write( train_samples, "omniobject_train.txt" )
write( test_samples, "omniobject_test.txt" )

quit()
