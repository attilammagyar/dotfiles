#!/bin/bash

WORKDIR="/home/athos"
OUTPUT_DIR="$WORKDIR/Pictures/ci"
WIDTH=1600
HEIGHT=900

URLS=$(cat <<URLS
syslogng35 https://travis-ci.org/balabit/syslog-ng-3.5
angularui https://travis-ci.org/angular-ui/bootstrap
URLS
)

function main()
{
    local url=""
    local output=""

    echo "$URLS" \
        | while read
          do
              url=$(echo "$REPLY" | cut -d' ' -f2-)
              output=$(echo "$REPLY" | cut -d' ' -f1)
              output="$OUTPUT_DIR/$output.png"
              echo "Saving $url to $output"
              xvfb-run --server-args="-screen 0, 640x480x24" \
                  /home/athos/bin/webshot.py "$url" "$output" "$WIDTH" "$HEIGHT"
          done
}

main >"$WORKDIR/.update_jenkins_screenshot.log" 2>&1
