#!/bin/bash

function main()
{
    local file_name="$1"

    local file_size=$(du -hs "$file_name" | cut -f1)

    local video_info=$(
        mplayer -frames 0 -identify "$file_name" 2>/dev/null \
            | grep '^ID_\(\(VIDEO_\(WIDTH\|HEIGHT\|FPS\)\)\|LENGTH\)='
    )
    local video_width=$(echo "$video_info" | grep WIDTH | cut -d= -f2)
    local video_height=$(echo "$video_info" | grep HEIGHT | cut -d= -f2)
    local video_fps=$(echo "$video_info" | grep FPS | cut -d= -f2)
    local video_length=$(echo "$video_info" | grep LENGTH | cut -d= -f2)

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$video_fps" \
        "$video_width" \
        "$video_height" \
        "$file_size" \
        $(format_length "$video_length") \
        "$file_name"
}

function format_length()
{
    local length=$(printf "%d" "$1" 2>/dev/null)
    local hours=$(($length/3600))
    local minutes=$((($length%3600)/60))
    local seconds=$(($length%60))

    printf "%02d:%02d:%02d\n" "$hours" "$minutes" "$seconds"
}

main "$@"
