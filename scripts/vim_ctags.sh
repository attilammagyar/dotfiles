#!/bin/bash

CTAGS="ctags"

function main()
{
    tag_file="$1"
    source_file="$2"

    [[ -z "$tag_file" ]] && return 0
    [[ -z "$source_file" ]] && return 0
    [[ ! -f "$tag_file" ]] && return 0
    [[ ! -f "$source_file" ]] && return 0
    which "$CTAGS" || return 0

    if can_be_updated "$tag_file"
    then
        update_tag_file "$tag_file" "$source_file"
    else
        rebuild_tag_file "$tag_file"
    fi
}

function can_be_updated()
{
    tag_file="$1"

    [[ -s "$tag_file" ]] || return 1

    last_modified_timestamp="`stat -c\"%Y\" \"$tag_file\"`"
    now="`date \"+%s\"`"
    too_old=$(($now-7200))

    [[ $last_modified_timestamp -gt $too_old ]] || return 2
    return 0
}

function update_tag_file()
{
    tag_file="$1"
    source_file="$2"

    "$CTAGS" --append --sort=yes -f "$tag_file" "$source_file"
}

function rebuild_tag_file()
{
    tag_file="$1"

    "$CTAGS" --sort=yes -Rf "$tag_file" *
}

main "$@" >/dev/null 2>/dev/null
exit 0
