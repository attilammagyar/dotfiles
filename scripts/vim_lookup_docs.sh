#!/bin/bash

GLIB_DOCS="/usr/share/doc/libglib2.0-doc/glib/api-index-full.html"
PHP_DOCS=~/docs/php-5.6/indexes.functions.html
LINKS="links"

main()
{
    source_file="$1"
    search_keyword="$2"

    if [ "x$source_file" = "x" ]
    then
        echo "Usage: $0 source_file [search_keyword]" >&2
        return 1
    fi

    language=$(guess_language "$source_file")
    [ "x$language" != "x" ] || return 1

    if [ "x$search_keyword" != "x" ]
    then
        case "$language" in
            "c"|"cpp")
                lookup_in_man || lookup_in_glib_docs
                return $?
                ;;
            "php")
                lookup_in_php_docs
                return $?
                ;;
            *)
                google_it
                return $?
                ;;
        esac
    else
        case "$language" in
            "c"|"cpp")
                open_glib_docs
                return $?
                ;;
            "php")
                open_php_docs
                return $?
                ;;
            *)
                search_keyword="docs"
                google_it
                return $?
                ;;
        esac
    fi
}

guess_language()
{
    filename="$1"
    extension=$(echo "$filename" | rev | cut -d'.' -f1 | rev)
    case "$extension" in
        "cpp"|"c++"|"hpp"|"h++"|"cc"|"hh")
            echo "cpp"
            ;;
        "c"|"h")
            echo "c"
            ;;
        "php"|"php3")
            echo "php"
            ;;
        "js")
            echo "javascript"
            ;;
        "sh"|"bashrc")
            echo "bash"
            ;;
        "py")
            echo "python"
            ;;
        "rb")
            echo "ruby"
            ;;
        "vimrc")
            echo "vim"
            ;;
        "pl"|"pm"|"pmk")
            echo "perl"
            ;;
        "java")
            echo "java"
            ;;
    esac
}

lookup_in_man()
{
    man 2 "$search_keyword" 2>/dev/null || man 3 "$search_keyword" 2>/dev/null || return 1
}

lookup_in_glib_docs()
{
    [ -f "$GLIB_DOCS" ] || return 1
    doc_dir=$(dirname "$GLIB_DOCS")
    doc_url=$(find_doc_url "$GLIB_DOCS")
    doc_file=$(echo "$doc_url" | cut -d'#' -f1)
    [ -f "$doc_dir/$doc_file" ] || return 1
    $LINKS "file://$doc_dir/$doc_url" || return 1
    return 0

}

find_doc_url()
{
    index="$1"
    grep -iFwm1 "$search_keyword" "$index" \
        | cut -c-100 \
        | sed -r 's/.*href="([^\"]*)".*/\1/g'
}

open_glib_docs()
{
    open_docs "$GLIB_DOCS"
    return $?
}

lookup_in_php_docs()
{
    [ -f "$PHP_DOCS" ] || return 1
    doc_dir=$(dirname "$PHP_DOCS")
    doc_file=$(find_doc_url "$PHP_DOCS" | cut -d '#' -f1)
    [ -f "$doc_dir/$doc_file" ] || return 1
    $LINKS "file://$doc_dir/$doc_file" || return 1
    return 0
}

open_php_docs()
{
    open_docs "$PHP_DOCS"
    return $?
}

open_docs()
{
    doc_file="$1"
    [ -f "$doc_file" ] || return 1
    $LINKS "$doc_file" || return 1
    return 0
}

google_it()
{
    $LINKS "http://google.com/search?q=$language+$search_keyword"
    return $?
}

main "$@"
exit $?
