#!/bin/bash

function main()
{
    command="$1"
    source_file="$2"

    shift 2

    retval=0
    base_name="`basename \"$source_file\"`"

    cd "`dirname \"$source_file\"`"
    case "$command" in
        grep) git_grep "$@" ;;
        log) git_log "$base_name" "$@" ;;
        blame) git_blame "$base_name" "$@" ;;
        *) retval=1 ;;
    esac
    cd -

    return $retval
}

function git_grep()
{
    word="$1"

    cd "`git rev-parse --show-cdup`"
    git grep -C3 "$word" \
        | less -r
}

function git_log()
{
    source_file="$1"

    git log --patch-with-stat --follow "$source_file" \
        | less -r
}

function git_blame()
{
    source_file="$1"
    line="$2"

    git blame -L "$line,+1" "$source_file"
}

main "$@"
exit $?
