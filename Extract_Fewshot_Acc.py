import numpy as np
import sys
import glob

ways = [ 5, 10 ]
shots = [ 10, 20 ]
folds = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

if( __name__ == '__main__' ):
    if( len(sys.argv) != 2 ):
        print( "Usage: python " + sys.argv[0] + " <base_path>" )
        print( "E.g., python " + sys.argv[0] + "./experiments/finetune_fewshot" )
        quit()

    base_path = sys.argv[1]

    for w in ways:
        for s in shots:
            accs = []
            for f in folds:
                log_dir = base_path + "_w" + str(w) + "_s" + str(s) + "_f" + str(f)
                log_file = list( glob.glob( log_dir + "/*.log" ) )
                if( len( log_file ) != 1 ):
                    print( "Error: each log directory must contain only one log file." )
                    print( "But, the following directory has " + str( len( log_file ) ) + " log files." )
                    print( log_dir )
                    quit()
                log_file = log_file[0]
                with open( log_file ) as f:
                    lines = f.readlines()
                acc = lines[-1].split()[11]
                accs.append( float( acc ) )
            accs = np.asarray( accs )
            print( str(w) + " way, " + str(s) + " shot" )
            print( str( np.mean( accs ) ) + " +- " + str( np.std( accs ) ) )

    quit()
