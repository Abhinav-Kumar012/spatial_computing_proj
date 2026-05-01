#!/bin/bash
while true; do
    python3 -u main.py | tee -a out_retry.log
    [ "${PIPESTATUS[0]}" -eq 0 ] && break
    echo "retry"
done
echo "Done"