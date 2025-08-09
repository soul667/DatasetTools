export HDF5_USE_FILE_LOCKING=FALSE
export RAY_DEDUP_LOGS=0
uv run agibot_h5.py \
    --src-path /data/axgu/dataset/pack_in_the_supermarket\
    --output-path /data/axgu/dataset/pack_in_the_supermarket/lerobot \
    --eef-type sim \
    --cpus-per-task 3 \
    --debug 
