
track_eval_path=/Users/yaliu/Documents/git/TrackEval

track_n=TDTR_IOU #TDTR_2Points

org_data=output/MOTDataset/MOT20/${track_n}/

for d in $org_data/*/ ; do
    track_basename=$(basename "$d")
    track_name=${track_n}_${track_basename}
    track_path=${track_eval_path}/data/trackers/mot_challenge/MOT20-train/${track_name}
    if [ ! -d $track_path ] 
    then
        mkdir -p $track_path
        cp -r $d $track_path/data
    fi
done

# cd $track_eval_path
# python scripts/run_mot_challenge.py --BENCHMARK MOT20 #--TRACKERS_TO_EVAL
