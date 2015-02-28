#!/bin/bash

function error()
{
    local ret=$1
    shift
    echo "$@" >&2
    exit $ret
}

function usage()
{
    cat <<USAGE
Usage:

    `basename $0` <input_video> <output_video> \\
        [max_width:max_height [vbitrate [extra_mecoder_args]]]

USAGE

    exit 1
}

function calculate_scaling_args()
{
    local input="$1"
    local max_dimensions="$3"

    local video_info=$(
        mplayer -frames 0 -identify "$input" 2>/dev/null \
            | grep '^ID_VIDEO_\(WIDTH\|HEIGHT\)='
    )
    local max_width=$(echo "$2" | cut -d: -f1)
    local max_height=$(echo "$2" | cut -d: -f2)
    local video_width=$(echo "$video_info" | grep WIDTH | cut -d= -f2)
    local video_height=$(echo "$video_info" | grep HEIGHT | cut -d= -f2)
    local scaled_width=$video_width
    local scaled_height=$video_height

    if [[ $scaled_width -gt $max_width ]]
    then
        scaled_height=$((($scaled_height*$max_width)/$scaled_width))
        scaled_width=$max_width
    fi

    if [[ $scaled_height -gt $max_height ]]
    then
        scaled_width=$((($scaled_width*$max_height)/$scaled_height))
        scaled_height=$max_height
    fi

    echo "$scaled_width:$scaled_height"
}

function main()
{
    local input="$1"
    local output="$2"
    local max_dimensions="$3"
    local vbitrate="$4"

    shift
    shift
    shift
    shift
    local extra_args="$@"

    local scale=""

    if [[ "x$input" = "x" || "x$output" = "x" || "$input" = "x--help" ]]
    then
        usage
    fi

    if [[ "x$max_dimensions" != "x" ]]
    then
        scale="-vf scale="$(calculate_scaling_args "$input" "$max_dimensions")
    fi

    if [[ "x$vbitrate" != "x" ]]
    then
        vbitrate=":vbitrate=$vbitrate"
    fi

    mencoder \
        "$input" \
        -ovc lavc \
        -lavcopts "vcodec=mpeg4$vbitrate:autoaspect:vpass=1" \
        -oac mp3lame \
        -lameopts vbr=3 \
        $scale \
        -o /dev/null \
        $extra_args
    mencoder \
        "$input" \
        -ovc lavc \
        -lavcopts "vcodec=mpeg4$vbitrate:mbd=2:trell:autoaspect:vpass=2" \
        -oac mp3lame \
        -lameopts vbr=3 \
        $scale \
        -o "$output" \
        $extra_args

    rm divx2pass.log
}

main "$@"
