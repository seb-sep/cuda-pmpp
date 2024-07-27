
REMOTE_HOST=home-desktop
REMOTE_DIR=~/CUDA/
LOCAL_DIR=~/Code/CUDA/pmpp

RSYNC_OPTS=-avz --progress

.PHONY: to-remote from-remote

to-remote:
	rsync $(RSYNC_OPTS) $(LOCAL_DIR) $(REMOTE_HOST):$(REMOTE_DIR)

from-remote:
	rsync $(RSYNC_OPTS) $(REMOTE_HOST):$(REMOTE_DIR) $(LOCAL_DIR)
