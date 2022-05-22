
track_eval_path=${PWD}/../TrackEval

track_n=DAGv2_2Points #TDTR_2Points
dataset=DanceTrack

org_data=output/MOTDataset/${dataset}/${track_n}/

for d in $org_data/*/ ; do
    track_basename=$(basename "$d")
    track_name=${track_n}_${track_basename}
    track_path=${track_eval_path}/data/trackers/mot_challenge/${dataset}-train/${track_name}
    if [ ! -d $track_path ] 
    then
        mkdir -p $track_path
        cp -r $d $track_path/data
    fi
done

cd $track_eval_path
python scripts/run_mot_challenge.py --BENCHMARK ${dataset} #--TRACKERS_TO_EVAL
