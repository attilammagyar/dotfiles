#!/bin/bash

CTAGS="ctags"
TAGFILE_NAME=""

pwd="`pwd`"

main()
{
    TAGFILE_NAME="$1"
    source_file="$2"

    [ -z "$TAGFILE_NAME" ] && return 0
    [ -z "$source_file" ] && return 0
    [ ! -f "$source_file" ] && return 0
    which "$CTAGS" || return 0
    cd_to_tagfile || return 0

    if can_be_updated
    then
        update_tagfile "$source_file"
    else
        rebuild_tagfile
    fi
}

cd_to_tagfile()
{
    while [ ! -f "$TAGFILE_NAME" -a "`pwd`" != "/" ]
    do
        cd .. || return 1
    done
    [ "x`pwd`" = "x/" ] && return 2
    return 0
}

can_be_updated()
{
    [ -s "$TAGFILE_NAME" ] || return 1

    last_modified_timestamp="`stat -c\"%Y\" \"$TAGFILE_NAME\"`"
    now="`date \"+%s\"`"
    too_old=$(($now-7200))

    [ $last_modified_timestamp -gt $too_old ] || return 2
    return 0
}

update_tagfile()
{
    source_file="$1"

    # Ugly hack to overcome Vim's behavior when using project directories that
    # contain symlinks to actual git repos. In that case, $source_file
    # contains a resolved path that may be outside of the current directory.
    # This will lead to the tag file containing absolute paths outside of the
    # project directory making hard to navigate the source code staying inside
    # the project directory.
    basename="`basename \"$source_file\"`"
    relative_path_to_source="`find -L . -type f -name \"$basename\" 2>/dev/null | head -n 1`"
    [ -f "$relative_path_to_source" ] || return 0
    "$CTAGS" --append --sort=yes -f "$TAGFILE_NAME" "$relative_path_to_source"
}

rebuild_tagfile()
{
    "$CTAGS" --sort=yes -Rf "$TAGFILE_NAME" *
}

main "$@" >/dev/null 2>/dev/null
exit_code=$?
cd "$pwd"
exit $exit_code
