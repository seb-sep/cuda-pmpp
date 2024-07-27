
REMOTE_HOST=home-desktop
REMOTE_DIR=~/CUDA/pmpp/
LOCAL_DIR=~/Code/CUDA/pmpp/

RSYNC_OPTS=-avz --progress --exclude '.git/'

.PHONY: to-remote from-remote

to-remote:
	rsync $(RSYNC_OPTS) $(LOCAL_DIR) $(REMOTE_HOST):$(REMOTE_DIR)

from-remote:
	rsync $(RSYNC_OPTS) --exclude 'Makefile' $(REMOTE_HOST):$(REMOTE_DIR) $(LOCAL_DIR)
