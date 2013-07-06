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
        commit_all) git_commit_all ;;
        commit_selection) git_commit_selection ;;
        *) retval=1 ;;
    esac
    cd -

    return $retval
}

function git_grep()
{
    word="$1"

    git_cd_to_root
    git grep -C3 -n "$word" \
        | less -r
}

function git_cd_to_root()
{
    cd "`git rev-parse --show-cdup`"
}

function git_log()
{
    source_file="$1"

    git log --decorate --patch-with-stat --follow "$source_file" \
        | less -r
}

function git_blame()
{
    source_file="$1"
    line="$2"

    git blame -L "$line,+1" "$source_file"
}

function git_commit_all()
{
    git_cd_to_root
    git add .
    git commit -a --verbose
}

function git_commit_selection()
{
    git_cd_to_root
    git add --interactive
    git commit --verbose
}

main "$@"
exit $?
