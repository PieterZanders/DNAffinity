#!/bin/bash
cd "$(dirname "$0")"; while pgrep -f queue_shape.sh >/dev/null; do sleep 20; done; ./queue_perm.sh
