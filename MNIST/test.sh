pkl() (
    python -c 'import pickle,sys;d=pickle.load(open(sys.argv[1],"rb"));print(d)'
    "$1"
)

pkl ./mnist.pkl
