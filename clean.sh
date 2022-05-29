org_data=dataset/MOT20-noisy-idswitch
# output_path=output/MOTDataset/MOT20-noisy-bbox

for d in $org_data/*_iou_0.50/ ; do
    # track_basename=$(basename "$d")
    python clean.py \
        "dataset.path='${d}'" \
        algo=tdtr_iou #tdtr_iou #uniform
done