#!/bin/bash

function error()
{
	ret=$1
	shift
	echo "$@" >&2
	exit $ret
}

function usage()
{
	echo "Usage: `basename '$0'` <input_video_file> <output_video_file> [max_width:max_height [vbitrate]]"
	exit 1
}

function calculate_scaling_args()
{
	video_info=`mplayer -frames 0 -identify "$1" 2>/dev/null | grep '^ID_VIDEO_\(WIDTH\|HEIGHT\)='`
	max_width=`echo "$2" | cut -d: -f1`
	max_height=`echo "$2" | cut -d: -f2`
	video_width=`echo "$video_info" | grep WIDTH | cut -d= -f2`
	video_height=`echo "$video_info" | grep HEIGHT | cut -d= -f2`
	scaled_width=$video_width
	scaled_height=$video_height
	if [ $scaled_width -gt $max_width ]
	then
		scaled_height=$((($scaled_height*$max_width)/$scaled_width))
		scaled_width=$max_width
	fi
	if [ $scaled_height -gt $max_height ]
	then
		scaled_width=$((($scaled_width*$max_height)/$scaled_height))
		scaled_height=$max_height
	fi
	echo "$scaled_width:$scaled_height"
}

function main()
{
	input_video_file="$1"
	output_video_file="$2"
	max_dimensions="$3"
	vbitrate="$4"

	if [ "x$input_video_file" = "x" -o "x$output_video_file" = "x" -o "$input_video_file" = "x--help" ]
	then
		usage
	fi

	scale=""
	if [ "x$max_dimensions" != "x" ]
	then
		scale="-vf scale="$(calculate_scaling_args "$input_video_file" "$max_dimensions")
	fi

	if [ "x$vbitrate" != "x" ]
	then
		vbitrate=":vbitrate=$vbitrate"
	fi

	mencoder "$input_video_file" \
		-ovc lavc -lavcopts vcodec=mpeg4$vbitrate:vpass=1 \
		-oac mp3lame -lameopts vbr=3 \
		$scale -o /dev/null
	mencoder "$input_video_file" \
		-ovc lavc -lavcopts vcodec=mpeg4$vbitrate:mbd=2:trell:vpass=2 \
		-oac mp3lame -lameopts vbr=3 \
		$scale -o "$output_video_file"
}

main "$@"
