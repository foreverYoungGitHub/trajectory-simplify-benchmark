org_data=dataset/MOT20-noisy-bbox
output_path=output/MOTDataset/MOT20-noisy-bbox

for d in $org_data/*/ ; do
    track_basename=$(basename "$d")
    python visual_mot.py \
        "tracks_pattern='${org_data}/${track_basename}/{seq_name}.txt'" \
        "output_pattern='${output_path}/${track_basename}_imgs/{seq_name}'"
done