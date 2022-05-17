
cd dataset

if [ ! -f DanceTrack.zip ] 
then
    gdown 1mOb1g-ptPX9h9Djlj-xVvn7MdEerqWLX
    unzip DanceTrack.zip -d .
fi

if [ ! -f PoseTrack.zip ] 
then
    gdown 1NdI9KHaQggylQGdCO7t-uIQJg7DjJ5g5
    unzip PoseTrack.zip -d .
fi

if [ ! -f MOT20.zip ] 
then
    gdown 1xkpnUaM54dzwBfakVUQMlG5qaQdyVQZc
    unzip MOT20.zip -d .
fi

if [ ! -f MOT17.zip ] 
then
    gdown 1AjiqAP2AGR_Qk8M0t2y388LvH7EDpZok
    unzip MOT17.zip -d .
fi

