
for d in output/PoseTrackDataset/trainval/TDTR_Points/*_0.1/ ; do
    echo "$d"
    i=$((${#d}-1))
    python -m poseval.evaluate \
        --groundTruth=dataset/PoseTrack/trainval \
        --predictions=$d \
        --evalPoseTracking --evalPoseEstimation \
        > ${d:0:$i}.log
done
