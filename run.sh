docker run \
--shm-size 30G \
--log-driver=none \
--gpus all \
-v $(realpath /storage_research):/storage_research \
--name garsiya_template \
-it \
--entrypoint \
/bin/bash garsiya_base
# garsiya_base

