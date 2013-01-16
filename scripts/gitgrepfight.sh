#!/bin/bash

SCRIPT="`basename \"$0\"`"

function main()
{
    flags="-E"
    add_random_samples=0

    parsed_options=$(parse_options "$@")
    if [[ $? -ne 0 ]]
    then
        print_usage >&2
        return 1
    fi

    eval set -- "$parsed_options"

    while true
    do
        option="$1"
        shift
        case "$option" in
            -i|--case-insensitive) flags="-Ei" ;;
            -s|--samples) add_random_samples=1 ;;
            -h|--help) print_usage ; return 0 ;;
            --) break ;;
        esac
    done

    validate_arguments "$@" || return 2

    count_matches "$flags" "$@" \
        | sort -nr \
        | format_output "$add_random_samples" "$flags" \
        | print_winner
}

function parse_options()
{
    getopt -o his --long help,case-insensitive,samples -n "$SCRIPT" -- "$@"
    return $?
}

function print_usage()
{
    cat <<USAGE
Usage: $SCRIPT [options] <regexp1> [regexp2 [...]]

  At least 1 extended regular expression is required.

  Options:

    -i, --case-insensitive      Search for case insensitive matches
    -s, --samples               Print random samples of matches
    -h, --help                  Print this usage information

USAGE
}

function validate_arguments()
{
    if [ $# -lt 1 ]
    then
        echo "At least 1 extended regular expression is required!" >&2
        print_usage >&2
        return 1
    fi
    return 0
}

function count_matches()
{
    flags="$1"

    shift

    while [[ $# -gt 0 ]]
    do
        regexp="$1"
        matches=$(git_grep_count "$flags" "$regexp")
        echo "$matches $regexp"
        shift
    done
}

function git_grep_count()
{
    flags="$1"
    regexp="$2"

    git grep "$flags" --no-color -- "$regexp" 2>/dev/null \
        | wc -l
}

function format_output()
{
    add_random_samples="$1"
    flags="$2"

    if [[ $add_random_samples -eq 0 ]]
    then
        sed "s/ / matches for /"
    else
        while read
        do
            matches=$(echo "$REPLY" | cut -d" " -f1)
            regexp=$(echo "$REPLY" | cut -d" " -f2-)

            echo -n "$matches matches for $regexp"

            if [[ $matches -gt 0 ]]
            then
                echo ", samples:"
                print_random_samples "$flags" "$regexp"
            else
                echo ""
            fi
        done
    fi
}

function print_random_samples()
{
    flags="$1"
    regexp="$2"

    git grep -C1 "$flags" -- "$regexp" \
        | tail -n$((($RANDOM%100)+10)) \
        | head -n12 \
        | sed "s/^/  | /g"
}

function print_winner()
{
    winner=""
    lines=0
    while read
    do
        echo "$REPLY"
        lines=$(($lines+1))
        if [[ "x$winner" = "x" && "x$REPLY" =~ ^x[0-9]*\ matches\ for\  ]]
        then
            winner="$REPLY"
        fi
    done
    if [[ $lines -gt 1 ]]
    then
        echo ""
        echo "And the winner is: $winner"
    fi
}

main "$@"
exit $?
